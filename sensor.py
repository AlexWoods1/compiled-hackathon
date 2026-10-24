"""
sensor.py  (live microphone censor)

Listens to your mic, transcribes with Whisper, and plays the audio back a few
seconds late with censored words replaced by static.

A live censor MUST delay the audio: a word can't be bleeped until it has been
spoken and recognised (same idea as the "7-second delay" on live TV).

Install:
    python -m pip install faster-whisper sounddevice numpy

Usage:
    python sensor.py --words monday taco
    python sensor.py --words monday --model base --window 3 --hold 0.6
    python sensor.py --list-devices
    python sensor.py --words monday --device 1 --out-device 4

USE HEADPHONES, otherwise your speakers feed back into the mic.
Stop with Ctrl+C.
"""

import argparse
import queue
import re
import threading

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SR = 16000  # sample rate Whisper expects


# --------------------------------------------------------------------------- #
# Word matching
# --------------------------------------------------------------------------- #
def normalize(word):
    return re.sub(r"[^\w']", "", word.lower())


def matches(word, censored):
    w = normalize(word)
    return any(w in (c, c + "s", c + "es") for c in censored)


# --------------------------------------------------------------------------- #
# Transcribe + censor one window of audio
# --------------------------------------------------------------------------- #
def transcribe(model, audio, prompt=None):
    segments, _ = model.transcribe(
        audio,
        language="en",                   # stop Whisper guessing the wrong language
        initial_prompt=prompt,           # nudges Whisper toward the banned words
        beam_size=1,                     # faster
        word_timestamps=True,
        vad_filter=True,                 # skip silence (reduces made-up words)
        condition_on_previous_text=False,
    )
    return [(w.word.strip(), w.start, w.end) for s in segments for w in (s.words or [])]


def process_window(model, window, censored, hold, pad, volume, prompt=None):
    """
    Returns (audio_ready_to_play, audio_to_keep_buffered).

    The last `hold` seconds are held back and re-checked with the next window, so a
    word that gets cut in half by a window boundary is still caught.
    """
    n = len(window)
    words = transcribe(model, window, prompt)

    # Choose where to cut. Never cut through the middle of a word.
    cut = n - int(hold * SR)
    for _, s, e in words:
        s_i, e_i = int(s * SR), int(e * SR)
        if s_i < cut < e_i:
            cut = s_i
    cut = max(cut, 0)

    out = window.copy()
    found = []
    for word, s, e in words:
        if matches(word, censored):
            a = max(0, int((s - pad) * SR))
            b = min(n, int((e + pad) * SR))
            out[a:b] = np.random.uniform(-volume, volume, b - a).astype(np.float32)
            found.append(normalize(word))

    if found:
        print("  censored:", ", ".join(found))
    elif words:
        print("  heard:", " ".join(w for w, _, _ in words))

    return out[:cut], window[cut:]


# --------------------------------------------------------------------------- #
# Live loop
# --------------------------------------------------------------------------- #
def run(args):
    censored = {normalize(w) for w in args.words}
    prompt = "Words: " + ", ".join(sorted(censored)) + "."

    print(f"Loading Whisper model '{args.model}'...")
    model = WhisperModel(args.model, device="auto", compute_type="int8")

    in_q, out_q = queue.Queue(), queue.Queue()

    def mic_callback(indata, frames, time_info, status):
        if status:
            print("mic status:", status)
        in_q.put(indata[:, 0].copy())

    def player():
        with sd.OutputStream(samplerate=SR, channels=1, dtype="float32",
                             device=args.out_device) as out:
            while True:
                chunk = out_q.get()
                if chunk is None:
                    break
                out.write(chunk.reshape(-1, 1))

    threading.Thread(target=player, daemon=True).start()
    out_q.put(np.zeros(int(0.5 * SR), dtype=np.float32))  # small cushion so playback doesn't stutter

    print(f"Listening... censoring: {sorted(censored)}  (Ctrl+C to stop)")
    print("Audio plays back with a delay of a few seconds. Wear headphones!\n")

    buf = np.zeros(0, dtype=np.float32)
    try:
        with sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                            blocksize=4096, latency="high",
                            device=args.device, callback=mic_callback):
            while True:
                buf = np.concatenate([buf, in_q.get()])
                if len(buf) >= int(args.window * SR):
                    ready, buf = process_window(
                        model, buf, censored, args.hold, args.pad, args.volume, prompt
                    )
                    if len(ready):
                        out_q.put(ready)
    except KeyboardInterrupt:
        print("\nStopping...")
        if len(buf):  # flush whatever is left
            ready, _ = process_window(model, buf, censored, 0, args.pad, args.volume, prompt)
            out_q.put(ready)
        out_q.put(None)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Live microphone word censor.")
    p.add_argument("--words", nargs="+", help="censored words, e.g. monday")
    p.add_argument("--model", default="small", help="tiny/base/small (smaller = faster)")
    p.add_argument("--window", type=float, default=4.0, help="seconds of audio per pass (= delay)")
    p.add_argument("--hold", type=float, default=0.8, help="seconds held back for word-boundary safety")
    p.add_argument("--pad", type=float, default=0.08, help="extra seconds censored around each word")
    p.add_argument("--volume", type=float, default=0.25, help="static loudness 0-1")
    p.add_argument("--device", type=int, default=None, help="input (mic) device number")
    p.add_argument("--out-device", type=int, default=None, help="output (headphones) device number")
    p.add_argument("--list-devices", action="store_true", help="show audio devices and exit")
    args = p.parse_args()

    if args.list_devices:
        print(sd.query_devices())
    elif not args.words:
        p.error("--words is required")
    else:
        run(args)