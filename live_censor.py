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
DEFAULT_MODEL = "base.en"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"
DEV_SR = 48000
WHISPER_SR = 16000
DOWN = DEV_SR // WHISPER_SR
BLOCKSIZE = 1024
PAD = 0.08
NOISE_VOL = 0.15
NO_SPEECH_PROB = 0.6
# * A couple of chunks of delay is the live buffer. Warn only past this.
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


def downsample(window):
    """Average 48 kHz audio down to the 16 kHz rate Whisper expects.

    Args:
        window: One-dimensional 48 kHz samples. Length must be divisible by 3.

    Returns:
        Float32 samples at 16 kHz.

    Raises:
        ValueError: The buffer length is not divisible by the downsample factor.
    """
    import numpy as np

    audio = np.asarray(window, dtype=np.float32)
    if len(audio) % DOWN != 0:
        raise ValueError(f"Audio length {len(audio)} is not divisible by {DOWN}.")
    return audio.reshape(-1, DOWN).mean(axis=1).astype(np.float32)


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


def build_prompt(topics: list[str]) -> str:
    """Build a short recognizer hint from the banned phrases.

    Args:
        topics: Phrases that should be spelled the way the topics file spells them.

    Returns:
        A one-sentence prompt.
    """
    return "Words that may come up: " + ", ".join(topics) + "."


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
        default=1.5,
        help="chunk length in seconds",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print every transcript line, not only bleeps",
    )
    return parser.parse_args(argv)


def transcribe_window(model, window, topics: list[str]) -> list[HeardWord]:
    """Transcribe one 48 kHz window.

    Args:
        model: Loaded WhisperModel.
        window: 48 kHz samples whose length is divisible by 3.
        topics: Phrases passed to the recognizer as hints.

    Returns:
        Timestamped words. Times are relative to the start of window.
    """
    hint = build_prompt(topics)
    segments, _info = model.transcribe(
        downsample(window),
        language="en",
        word_timestamps=True,
        beam_size=1,
        temperature=0.0,
        initial_prompt=hint,
        hotwords=hint,
        condition_on_previous_text=False,
        vad_filter=False,
    )
    return words_from_segments(segments)


def censor_window(window, heard: list[HeardWord], topics: list[str], verbose: bool):
    """Apply static to banned phrases inside one window.

    Args:
        window: 48 kHz samples.
        heard: Words whose times line up with window.
        topics: Banned phrases.
        verbose: When True, print the transcript as well as each bleep.

    Returns:
        The censored samples and the spans that were covered.
    """
    spans = find_banned_spans(heard, topics)
    for span in spans:
        print(f"   BLEEPED: '{span.topic}'", flush=True)
    if verbose and heard:
        print("heard:", " ".join(word.word.strip() for word in heard), flush=True)
    return apply_static(window, spans, DEV_SR), spans


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
    chunk_n = (int(args.chunk * DEV_SR) // DOWN) * DOWN
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
        prev = None
        last_report = 0.0
        while not stop.is_set():
            try:
                block = in_q.get(timeout=0.2)
            except queue.Empty:
                continue
            pending = np.concatenate([pending, block])
            if len(pending) < chunk_n:
                continue
            current = pending[:chunk_n].copy()
            pending = pending[chunk_n:]
            try:
                if prev is None:
                    heard = transcribe_window(model, current, topics)
                    prev, _spans = censor_window(current, heard, topics, args.verbose)
                    continue
                # * The overlap keeps a phrase that starts in the previous chunk.
                combined = np.concatenate([prev, current])
                heard = transcribe_window(model, combined, topics)
                censored, _spans = censor_window(combined, heard, topics, args.verbose)
                emit(censored[: len(prev)])
                prev = censored[len(prev) :]
            except Exception as exc:
                print(f"   [warn] transcription failed: {exc}", file=sys.stderr)
                if prev is not None:
                    emit(prev)
                prev = current
                continue
            backlog = in_q.qsize() * BLOCKSIZE / DEV_SR
            now = time.monotonic()
            if should_report_backlog(backlog, args.chunk, now, last_report):
                last_report = now
                print(
                    f"   [warn] audio is about {backlog:.0f}s behind.",
                    flush=True,
                )
        if prev is not None:
            emit(prev)

    in_dev = find_device(sd, args.in_device, "input")
    out_dev = find_device(sd, args.out_device, "output")
    channels = output_channels(sd, out_dev)
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    with (
        sd.InputStream(
            device=in_dev,
            samplerate=DEV_SR,
            channels=1,
            dtype="float32",
            blocksize=BLOCKSIZE,
            callback=on_input,
        ),
        sd.OutputStream(
            device=out_dev,
            samplerate=DEV_SR,
            channels=channels,
            dtype="float32",
            blocksize=BLOCKSIZE,
            callback=on_output,
        ),
    ):
        print(
            "Live. Censoring: "
            + ", ".join(topics)
            + ". Delay is about two chunks. Ctrl+C to stop."
        )
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            stop.set()
            worker_thread.join(timeout=3)
            with lock:
                remaining = len(out_buf) / DEV_SR
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
