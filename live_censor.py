"""Live mic censor for a Zoom audio call.

Captures the microphone, transcribes each short window with faster-whisper,
and replaces banned phrases with static in place. The other samples pass
through. Point Zoom at the virtual cable that this process writes to.

Setup:
    pip install "faster-whisper>=1.2.1" "sounddevice>=0.5.6" "numpy>=2.2"
    Windows: install VB-Audio Virtual Cable ("CABLE Input" / "CABLE Output")
    Mac: brew install blackhole-2ch

Zoom > Settings > Audio:
    Microphone = "CABLE Output" on Windows, or "BlackHole 2ch" on Mac
    Uncheck "Automatically adjust microphone volume"
    Background noise suppression = Low

Usage:
    python live_censor.py --list-devices
    python live_censor.py
    python live_censor.py --words "secret project"
    python live_censor.py --out-device "CABLE Input"
"""

import argparse
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

# * Device and censor settings. 48 kHz is what mics and virtual cables share.
DEFAULT_TOPICS = Path(__file__).resolve().parent / "demo" / "private_topics.txt"
DEFAULT_MODEL = "tiny.en"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"
DEV_SR = 48000
WHISPER_SR = 16000
BLOCKSIZE = 1024
PAD = 0.08
NOISE_VOL = 0.15
NO_SPEECH_PROB = 0.6
# * Used when a hit is split across two windows. The live loop censors the
# * whole chunk and does not wait on a second window.
OVERLAP_SECONDS = 0.25
# * Audio waiting beyond this many chunks is dropped so the delay cannot grow.
MAX_PENDING_CHUNKS = 2
BACKLOG_WARN_SECONDS = 8.0
BACKLOG_WARN_INTERVAL = 15.0


@dataclass(frozen=True)
class HeardWord:
    """One transcribed word and its time span, in seconds."""

    word: str
    start: float
    end: float


@dataclass(frozen=True)
class CensorSpan:
    """Audio interval that matches one banned phrase."""

    start: float
    end: float
    topic: str


def normalize_token(word: str) -> str:
    """Reduce a word to lowercase letters and digits.

    Args:
        word: Raw token, including any punctuation from the recognizer.

    Returns:
        The comparable form of the word. An empty string means the token
        had no letters or digits.
    """
    return re.sub(r"[^a-z0-9]", "", word.lower())


def tokens_match(heard: str, banned: str) -> bool:
    """Return whether two tokens are the same word, ignoring a simple plural.

    Args:
        heard: Normalized token from the transcript.
        banned: Normalized token from a topic phrase.

    Returns:
        True when the tokens are equal, or one is the other plus "s" or "es".
    """
    if not heard or not banned:
        return False
    if heard == banned:
        return True
    return heard in (banned + "s", banned + "es") or banned in (
        heard + "s",
        heard + "es",
    )


def load_topics(path: Path) -> list[str]:
    """Load banned phrases from a topics file.

    Args:
        path: Text file with one phrase per line. Blank lines and lines that
            start with "#" are ignored.

    Returns:
        Phrases in file order.

    Raises:
        FileNotFoundError: The topics file does not exist.
        ValueError: The file contains no phrases.
    """
    text = path.read_text(encoding="utf-8-sig")
    phrases = []
    for line in text.splitlines():
        phrase = line.strip()
        if not phrase or phrase.startswith("#"):
            continue
        phrases.append(phrase)
    if not phrases:
        raise ValueError(f"No topics in {path}")
    return phrases


def collect_topics(path: Path | None, extra: list[str]) -> list[str]:
    """Combine the topics file with extra phrases from the command line.

    Args:
        path: Topics file. None skips the file.
        extra: Additional phrases. Each entry is one phrase.

    Returns:
        The combined phrase list.

    Raises:
        FileNotFoundError: The topics file was requested and is missing.
        ValueError: Neither source produced a phrase.
    """
    phrases = load_topics(path) if path is not None else []
    phrases.extend(phrase.strip() for phrase in extra if phrase.strip())
    if not phrases:
        raise ValueError("No banned phrases. Add a topics file or --words.")
    return phrases


def matched_topics(words: list[str], topics: list[str]) -> list[str]:
    """Return banned phrases that appear in order in the transcript.

    Args:
        words: Transcript tokens, without times.
        topics: Banned phrases. A phrase matches only when every token lines
            up in order.

    Returns:
        Matching phrases, in topic-list order, once each.
    """
    heard = [normalize_token(word) for word in words]
    heard = [token for token in heard if token]
    found = []
    for topic in topics:
        tokens = [normalize_token(part) for part in topic.split()]
        tokens = [token for token in tokens if token]
        if not tokens or len(tokens) > len(heard):
            continue
        width = len(tokens)
        for index in range(len(heard) - width + 1):
            window = heard[index : index + width]
            if all(tokens_match(got, want) for got, want in zip(window, tokens)):
                found.append(topic)
                break
    return found


def find_banned_spans(words: list[HeardWord], topics: list[str]) -> list[CensorSpan]:
    """Find whole banned phrases inside a timestamped transcript.

    Args:
        words: Words from one audio window, in time order.
        topics: Banned phrases. A phrase matches only when every token lines
            up in order.

    Returns:
        Spans in start-time order.
    """
    heard = [word for word in words if normalize_token(word.word)]
    topic_tokens = []
    for topic in topics:
        tokens = [normalize_token(part) for part in topic.split()]
        tokens = [token for token in tokens if token]
        if tokens:
            topic_tokens.append((topic, tokens))

    spans = []
    for topic, tokens in topic_tokens:
        width = len(tokens)
        for index in range(len(heard) - width + 1):
            window = heard[index : index + width]
            if all(
                tokens_match(normalize_token(word.word), token)
                for word, token in zip(window, tokens)
            ):
                spans.append(
                    CensorSpan(
                        start=window[0].start,
                        end=window[-1].end,
                        topic=topic,
                    )
                )
    spans.sort(key=lambda span: (span.start, span.end))
    return spans


def tokens_from_segments(segments) -> list[str]:
    """Collect transcript tokens and drop silent segments.

    Args:
        segments: faster-whisper segments.

    Returns:
        Tokens in order. Times are ignored.
    """
    tokens = []
    for segment in segments:
        if getattr(segment, "no_speech_prob", 0.0) >= NO_SPEECH_PROB:
            continue
        tokens.extend((getattr(segment, "text", "") or "").split())
    return tokens


def words_from_segments(segments) -> list[HeardWord]:
    """Collect timestamped words and drop silent segments.

    Args:
        segments: faster-whisper segments. Each segment may carry words.

    Returns:
        Words that have a start and an end.
    """
    heard = []
    for segment in segments:
        if getattr(segment, "no_speech_prob", 0.0) >= NO_SPEECH_PROB:
            continue
        for word in segment.words or []:
            if word.start is None or word.end is None:
                continue
            heard.append(HeardWord(word.word, float(word.start), float(word.end)))
    return heard


def replace_with_static(samples):
    """Replace a whole chunk with static.

    Args:
        samples: One-dimensional float audio.

    Returns:
        Static of the same length. An empty chunk stays empty.
    """
    import numpy as np

    out = np.array(samples, dtype=np.float32, copy=True)
    if len(out) == 0:
        return out
    out[:] = np.random.uniform(-NOISE_VOL, NOISE_VOL, len(out)).astype(np.float32)
    return out


def apply_static(samples, spans: list[CensorSpan], sample_rate: int):
    """Replace each span with static and leave the surrounding audio in place.

    Args:
        samples: One-dimensional float audio at sample_rate.
        spans: Intervals in seconds, relative to the start of samples.
        sample_rate: Sample rate of samples.

    Returns:
        A copy of samples with static written over each span.
    """
    import numpy as np

    out = np.array(samples, dtype=np.float32, copy=True)
    for span in spans:
        start = max(0, int((span.start - PAD) * sample_rate))
        end = min(len(out), int((span.end + PAD) * sample_rate))
        if end <= start:
            continue
        out[start:end] = np.random.uniform(-NOISE_VOL, NOISE_VOL, end - start).astype(
            np.float32
        )
    return out


def resample(window, source_rate: int, target_rate: int):
    """Change the sample rate without changing the duration.

    Args:
        window: One-dimensional samples at source_rate.
        source_rate: Rate of window, in Hz.
        target_rate: Desired rate, in Hz.

    Returns:
        Float32 samples at target_rate.
    """
    import numpy as np

    audio = np.asarray(window, dtype=np.float32)
    if source_rate == target_rate or len(audio) == 0:
        return audio
    target_n = max(1, int(round(len(audio) * target_rate / source_rate)))
    source_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
    target_x = np.linspace(0.0, 1.0, num=target_n, endpoint=False)
    return np.interp(target_x, source_x, audio).astype(np.float32)


def to_whisper_rate(window, sample_rate: int):
    """Resample captured audio to the 16 kHz rate Whisper expects.

    Args:
        window: One-dimensional samples at sample_rate.
        sample_rate: Rate of window, in Hz.

    Returns:
        Float32 samples at 16 kHz, with the same duration as window.
    """
    return resample(window, sample_rate, WHISPER_SR)


def playback_audio(samples, capture_rate: int, playback_rate: int):
    """Match captured audio to the output device clock.

    Args:
        samples: Audio at capture_rate.
        capture_rate: Rate of samples, in Hz.
        playback_rate: Rate the output device expects, in Hz.

    Returns:
        Samples at playback_rate with the same duration.
    """
    return resample(samples, capture_rate, playback_rate)


def overlap_sample_count(chunk_n: int, sample_rate: int) -> int:
    """Return how many samples of the held chunk are checked again.

    Args:
        chunk_n: Samples in one chunk.
        sample_rate: Rate of those samples, in Hz.

    Returns:
        Overlap length, shorter than the chunk.
    """
    overlap = int(OVERLAP_SECONDS * sample_rate)
    if overlap <= 0 or overlap >= chunk_n:
        overlap = max(1, chunk_n // 3)
    return overlap


def drop_stale_audio(pending, chunk_n: int, sample_rate: int):
    """Drop queued audio that would push the live delay too far.

    Args:
        pending: Samples waiting to be transcribed, oldest first.
        chunk_n: Samples in one chunk.
        sample_rate: Rate of those samples, in Hz.

    Returns:
        The pending buffer, and how many seconds were discarded. Discarded
        audio is never played.
    """
    limit = chunk_n * MAX_PENDING_CHUNKS
    if chunk_n <= 0 or len(pending) <= limit:
        return pending, 0.0
    dropped = len(pending) - chunk_n
    return pending[-chunk_n:].copy(), dropped / sample_rate


def absorb_audio(pending, blocks, chunk_n: int, sample_rate: int):
    """Pull every queued block in, then drop audio that is too old to play.

    Args:
        pending: Samples already pulled off the microphone queue.
        blocks: Newer blocks waiting on the queue, oldest first.
        chunk_n: Samples in one chunk.
        sample_rate: Rate of those samples, in Hz.

    Returns:
        The pending buffer and the seconds of audio discarded.
    """
    import numpy as np

    if len(blocks):
        pending = np.concatenate([pending, *blocks])
    return drop_stale_audio(pending, chunk_n, sample_rate)


def split_boundary_spans(
    spans: list[CensorSpan], overlap_samples: int, sample_rate: int
) -> tuple[list[CensorSpan], list[CensorSpan]]:
    """Split spans into the held tail and the new chunk.

    Args:
        spans: Spans whose times start at the beginning of tail plus new audio.
        overlap_samples: Length of the tail that belongs to the held chunk.
        sample_rate: Sample rate shared by both pieces.

    Returns:
        Tail spans rebased to the start of the tail, and new-chunk spans
        rebased to the start of the new chunk. A word across the boundary
        is clipped into both sides.
    """
    overlap_s = overlap_samples / sample_rate
    tail_spans = []
    new_spans = []
    for span in spans:
        if span.start < overlap_s and span.end > 0:
            tail_spans.append(
                CensorSpan(
                    start=max(0.0, span.start),
                    end=min(overlap_s, span.end),
                    topic=span.topic,
                )
            )
        if span.end > overlap_s:
            new_spans.append(
                CensorSpan(
                    start=max(0.0, span.start - overlap_s),
                    end=span.end - overlap_s,
                    topic=span.topic,
                )
            )
    return tail_spans, new_spans


def should_report_backlog(
    backlog: float, chunk: float, now: float, last_report: float
) -> bool:
    """Return whether the queue is far enough behind to mention.

    Args:
        backlog: Seconds of audio waiting to be transcribed.
        chunk: Chunk length in seconds.
        now: Current clock time, in seconds.
        last_report: Clock time of the previous warning. Zero means none yet.

    Returns:
        True when the delay is past the live buffer and the last warning
        was long enough ago.
    """
    limit = max(BACKLOG_WARN_SECONDS, chunk * 4)
    if backlog <= limit:
        return False
    return now - last_report >= BACKLOG_WARN_INTERVAL


def fresh_status(status, seen: set[str]) -> str | None:
    """Return a stream status the first time it appears.

    Args:
        status: sounddevice callback status. Empty means the stream is fine.
        seen: Status texts already printed.

    Returns:
        The status text, or None when it should stay quiet.
    """
    if not status:
        return None
    text = str(status)
    if text in seen:
        return None
    seen.add(text)
    return text


def find_device(sd, name: str | None, kind: str) -> int | None:
    """Resolve a device substring to a sounddevice index.

    Args:
        sd: The sounddevice module.
        name: Substring of the device name. None selects the system default.
        kind: "input" or "output".

    Returns:
        The matching device index, or None for the default device.

    Raises:
        SystemExit: No device of that kind contains the substring.
    """
    if name is None:
        return None
    channel_key = f"max_{kind}_channels"
    for index, device in enumerate(sd.query_devices()):
        if name.lower() in device["name"].lower() and device[channel_key] > 0:
            return index
    raise SystemExit(f"No {kind} device matching '{name}'. Run with --list-devices.")


def device_rate(sd, device: int | None, kind: str) -> int:
    """Return the sample rate the device is already using.

    Args:
        sd: The sounddevice module.
        device: Device index, or None for the default of that kind.
        kind: "input" or "output".

    Returns:
        The device default rate in Hz.
    """
    if device is None:
        info = sd.query_devices(kind=kind)
    else:
        info = sd.query_devices(device)
    rate = int(info["default_samplerate"])
    if rate <= 0:
        return DEV_SR
    return rate


def output_channels(sd, device: int | None) -> int:
    """Use stereo when the output device has two channels.

    Args:
        sd: The sounddevice module.
        device: Output device index, or None for the default output.

    Returns:
        2 when the device accepts stereo, otherwise 1.
    """
    if device is None:
        info = sd.query_devices(kind="output")
    else:
        info = sd.query_devices(device)
    if int(info["max_output_channels"]) >= 2:
        return 2
    return 1


def import_runtime():
    """Import the audio and speech packages used by the live loop.

    Returns:
        The numpy module, the sounddevice module, and the WhisperModel class.

    Raises:
        SystemExit: A required package is not installed.
    """
    try:
        import numpy as np
        import sounddevice as sd
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with: "
            'pip install "faster-whisper>=1.2.1" "sounddevice>=0.5.6" "numpy>=2.2"'
        ) from exc
    return np, sd, WhisperModel


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the live-censor command line.

    Args:
        argv: Arguments, excluding the program name. None reads sys.argv.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Censor banned phrases on the microphone in place."
    )
    parser.add_argument(
        "--topics",
        default=str(DEFAULT_TOPICS),
        help="topics file, one phrase per line",
    )
    parser.add_argument(
        "--words",
        nargs="*",
        default=[],
        help='extra phrases, quoted if they contain spaces, e.g. "customer list"',
    )
    parser.add_argument("--in-device", help="substring of the microphone name")
    parser.add_argument(
        "--out-device",
        help='substring of the output name, e.g. "CABLE Input" or BlackHole',
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="faster-whisper model")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="cpu or cuda")
    parser.add_argument(
        "--compute-type",
        default=DEFAULT_COMPUTE_TYPE,
        help="int8 on CPU, float16 on CUDA",
    )
    parser.add_argument(
        "--chunk",
        type=float,
        default=0.6,
        help="chunk length in seconds. The voice is delayed by one chunk plus transcription",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print every transcript line, not only bleeps",
    )
    return parser.parse_args(argv)


def transcribe_window(model, window, topics: list[str], sample_rate: int) -> list[str]:
    """Transcribe one captured window.

    Args:
        model: Loaded WhisperModel.
        window: Samples at sample_rate.
        topics: Phrases passed to the recognizer as hints.
        sample_rate: Rate of window, in Hz.

    Returns:
        Transcript tokens. A banned phrase censors the whole window.
    """
    # * Hotwords bias spelling. A full sentence prompt was transcribed back as
    # * speech, which bleeped "tacos" when nobody said it and slowed the decode.
    # * Word alignment stays off. The chunk is censored as a whole.
    segments, _info = model.transcribe(
        to_whisper_rate(window, sample_rate),
        language="en",
        word_timestamps=False,
        beam_size=1,
        temperature=0.0,
        hotwords=", ".join(topics),
        condition_on_previous_text=False,
        no_repeat_ngram_size=3,
        vad_filter=False,
    )
    return tokens_from_segments(segments)


def _report_censor(heard: list[str], topics_hit: list[str], verbose: bool) -> None:
    """Print each bleep, and the transcript when verbose mode is on.

    Args:
        heard: Tokens from the window that was just transcribed.
        topics_hit: Banned phrases found in those tokens.
        verbose: When True, print the transcript as well as each bleep.
    """
    for topic in topics_hit:
        print(f"   BLEEPED: '{topic}'", flush=True)
    if verbose and heard:
        print("heard:", " ".join(heard), flush=True)


def load_model(whisper_model, model_name: str, device: str, compute_type: str):
    """Load a local model, and download it only when it is not cached.

    Args:
        whisper_model: The WhisperModel class.
        model_name: faster-whisper model id, such as "base.en".
        device: "cpu" or "cuda".
        compute_type: Quantization, such as "int8".

    Returns:
        A loaded model.
    """
    try:
        return whisper_model(
            model_name,
            device=device,
            compute_type=compute_type,
            local_files_only=True,
        )
    except Exception:
        print(f"'{model_name}' is not on disk yet. Downloading once...", flush=True)
        return whisper_model(
            model_name,
            device=device,
            compute_type=compute_type,
            local_files_only=False,
        )


def run(args: argparse.Namespace) -> None:
    """Run the microphone loop until Ctrl+C.

    Args:
        args: Parsed command-line arguments.

    Raises:
        SystemExit: Devices, topics, or dependencies are missing.
        ValueError: The chunk length is too short to downsample.
    """
    np, sd, whisper_model = import_runtime()
    if args.list_devices:
        print(sd.query_devices())
        return

    topics_path = Path(args.topics) if args.topics else None
    try:
        topics = collect_topics(topics_path, args.words)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"Loading '{args.model}' on {args.device} ({args.compute_type})...",
        flush=True,
    )
    model = load_model(whisper_model, args.model, args.device, args.compute_type)

    if args.chunk <= 0:
        raise ValueError("--chunk must be a positive number of seconds.")
    in_dev = find_device(sd, args.in_device, "input")
    out_dev = find_device(sd, args.out_device, "output")
    capture_rate = device_rate(sd, in_dev, "input")
    playback_rate = device_rate(sd, out_dev, "output")
    chunk_n = int(args.chunk * capture_rate)
    if chunk_n == 0:
        raise ValueError("--chunk is too short to transcribe.")

    in_q: queue.Queue = queue.Queue()
    out_buf = np.zeros(0, dtype=np.float32)
    lock = threading.Lock()
    stop = threading.Event()
    seen_status: set[str] = set()

    def on_input(indata, frames, time_info, status):
        text = fresh_status(status, seen_status)
        if text:
            print(text, file=sys.stderr)
        in_q.put(indata[:, 0].copy())

    def on_output(outdata, frames, time_info, status):
        nonlocal out_buf
        text = fresh_status(status, seen_status)
        if text:
            print(text, file=sys.stderr)
        outdata.fill(0)
        with lock:
            count = min(frames, len(out_buf))
            if count:
                outdata[:count, :] = out_buf[:count, np.newaxis]
            out_buf = out_buf[count:]

    def emit(samples) -> None:
        nonlocal out_buf
        with lock:
            out_buf = np.concatenate([out_buf, np.asarray(samples, dtype=np.float32)])

    def worker() -> None:
        pending = np.zeros(0, dtype=np.float32)
        last_report = 0.0
        while not stop.is_set():
            blocks = []
            while True:
                try:
                    blocks.append(in_q.get_nowait())
                except queue.Empty:
                    break
            if not blocks and len(pending) < chunk_n:
                try:
                    blocks.append(in_q.get(timeout=0.2))
                except queue.Empty:
                    continue
            pending, dropped = absorb_audio(pending, blocks, chunk_n, capture_rate)
            if dropped:
                print(
                    f"   [warn] skipped {dropped:.1f}s to stay live.",
                    flush=True,
                )
            if len(pending) < chunk_n:
                continue
            current = pending[:chunk_n].copy()
            pending = pending[chunk_n:]
            try:
                # * The chunk plays as soon as its text is ready. A banned
                # * phrase replaces the whole chunk, so word times are unused.
                heard = transcribe_window(model, current, topics, capture_rate)
                hits = matched_topics(heard, topics)
                _report_censor(heard, hits, args.verbose)
                audio = replace_with_static(current) if hits else current
                emit(playback_audio(audio, capture_rate, playback_rate))
            except Exception as exc:
                print(f"   [warn] transcription failed: {exc}", file=sys.stderr)
                emit(playback_audio(current, capture_rate, playback_rate))
                continue
            backlog = (in_q.qsize() * BLOCKSIZE + len(pending)) / capture_rate
            now = time.monotonic()
            if should_report_backlog(backlog, args.chunk, now, last_report):
                last_report = now
                print(
                    f"   [warn] audio is about {backlog:.0f}s behind.",
                    flush=True,
                )

    channels = output_channels(sd, out_dev)
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    with (
        sd.InputStream(
            device=in_dev,
            samplerate=capture_rate,
            channels=1,
            dtype="float32",
            blocksize=BLOCKSIZE,
            callback=on_input,
        ),
        sd.OutputStream(
            device=out_dev,
            samplerate=playback_rate,
            channels=channels,
            dtype="float32",
            blocksize=BLOCKSIZE,
            callback=on_output,
        ),
    ):
        print(
            "Live. Censoring: "
            + ", ".join(topics)
            + f". Mic {capture_rate} Hz, cable {playback_rate} Hz. Ctrl+C to stop."
        )
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            stop.set()
            worker_thread.join(timeout=3)
            with lock:
                remaining = len(out_buf) / playback_rate
            if remaining > 0:
                try:
                    sd.sleep(int(remaining * 1000) + 100)
                except KeyboardInterrupt:
                    pass
            print("\nStopped.", flush=True)


def main(argv: list[str] | None = None) -> None:
    """Start the live censor from the command line.

    Args:
        argv: Arguments, excluding the program name.
    """
    run(parse_args(argv))


if __name__ == "__main__":
    main()
