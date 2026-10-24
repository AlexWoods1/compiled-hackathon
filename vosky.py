"""
sensor_vosk.py  (live microphone censor, Vosk streaming version)

Same idea as sensor.py, but uses Vosk, a streaming recognizer. Instead of waiting
for a 4-second window, audio is fed to the recognizer in 0.1 s blocks and words
are recognised while you are still speaking. Your voice is played back through a
short delay line (default 1.5 s); if a banned word is recognised before it leaves
the delay line, it is replaced with static.

Install:
    python -m pip install -U vosk sounddevice numpy

Usage:
    python sensor_vosk.py --words monday taco
    python sensor_vosk.py --words monday --delay 2.0
    python sensor_vosk.py --words monday --only-banned
    python sensor_vosk.py --words monday --model-path C:\\models\\vosk-model-en-us-0.22-lgraph
    python sensor_vosk.py --list-devices

The first run downloads the small English model (about 40 MB) automatically.
USE HEADPHONES, otherwise your speakers feed back into the mic.
Stop with Ctrl+C.
"""

import argparse
import json
import queue
import re
import threading

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model, SetLogLevel

SR = 16000
BLOCK = 1600  # 0.1 seconds per block


# --------------------------------------------------------------------------- #
# Word matching
# --------------------------------------------------------------------------- #
def normalize(word):
    return re.sub(r"[^\w']", "", word.lower())


def matches(word, censored):
    w = normalize(word)
    return any(w in (c, c + "s", c + "es") for c in censored)


# --------------------------------------------------------------------------- #
# Censor bookkeeping
# --------------------------------------------------------------------------- #
def register(words, censored, intervals, rel_start, pad):
    """Add static regions (in absolute sample positions) for any banned words."""
    for w in words:
        if not matches(w["word"], censored):
            continue
        a = max(0, int((w["start"] - pad) * SR))
        b = int((w["end"] + pad) * SR)

        overlapping = [iv for iv in intervals if a < iv[1] and b > iv[0]]
        if overlapping:  # same word seen again (partial -> final): just widen it
            for iv in overlapping:
                iv[0] = min(iv[0], a)
                iv[1] = max(iv[1], b)
            continue

        intervals.append([a, b])
        late = "  (late - part already played, raise --delay)" if a < rel_start else ""
        print(f"  censored: {normalize(w['word'])} at {w['start']:.1f}s{late}")


def release(hist, rel_start, end_abs, intervals, volume, out_q):
    """Send audio up to end_abs to the speakers, with static over banned regions."""
    if end_abs <= rel_start:
        return hist, rel_start

    n = end_abs - rel_start
    chunk = hist[:n].copy()
    for a, b in intervals:
        lo, hi = max(a, rel_start), min(b, end_abs)
        if lo < hi:
            chunk[lo - rel_start:hi - rel_start] = np.random.uniform(
                -volume, volume, hi - lo
            ).astype(np.float32)
    out_q.put(chunk)

    hist = hist[n:]
    rel_start = end_abs
    intervals[:] = [iv for iv in intervals if iv[1] > rel_start - 10 * SR]  # forget old ones
    return hist, rel_start


# --------------------------------------------------------------------------- #
# Live loop
# --------------------------------------------------------------------------- #
def run(args):
    censored = {normalize(w) for w in args.words}

    SetLogLevel(-1)
    print("Loading Vosk model...")
    model = Model(model_path=args.model_path) if args.model_path else Model(lang="en-us")

    if args.only_banned:
        vocab = sorted(censored | {c + "s" for c in censored}) + ["[unk]"]
        rec = KaldiRecognizer(model, SR, json.dumps(vocab))
    else:
        rec = KaldiRecognizer(model, SR)
    rec.SetWords(True)
    try:
        rec.SetPartialWords(True)  # word timings in partial results = faster detection
    except AttributeError:
        print("Warning: your vosk is old; run  python -m pip install -U vosk  "
              "for faster detection. Using final results only (raise --delay to ~3).")

    in_q, out_q = queue.Queue(), queue.Queue()

    def mic_callback(indata, frames, time_info, status):
        if status:
            print("mic status:", status)
        in_q.put(bytes(indata))

    def player():
        with sd.OutputStream(samplerate=SR, channels=1, dtype="float32",
                             device=args.out_device) as out:
            while True:
                chunk = out_q.get()
                if chunk is None:
                    break
                out.write(chunk.reshape(-1, 1))

    player_thread = threading.Thread(target=player)
    player_thread.start()
    out_q.put(np.zeros(int(0.2 * SR), dtype=np.float32))

    print(f"Listening... censoring: {sorted(censored)}  delay: {args.delay}s  (Ctrl+C to stop)")
    print("Wear headphones!\n")

    delay = int(args.delay * SR)
    hist = np.zeros(0, dtype=np.float32)  # audio not yet played
    rel_start = 0                         # absolute sample index of hist[0]
    fed = 0                               # total samples received from the mic
    intervals = []                        # [start, end] sample ranges to bleep

    try:
        with sd.RawInputStream(samplerate=SR, blocksize=BLOCK, dtype="int16",
                               channels=1, device=args.device, callback=mic_callback):
            while True:
                data = in_q.get()
                block = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                hist = np.concatenate([hist, block])
                fed += len(block)

                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    if res.get("text"):
                        print("  heard:", res["text"])
                    words = res.get("result", [])
                else:
                    words = json.loads(rec.PartialResult()).get("partial_result", [])

                register(words, censored, intervals, rel_start, args.pad)
                hist, rel_start = release(hist, rel_start, fed - delay,
                                          intervals, args.volume, out_q)
    except KeyboardInterrupt:
        print("\nStopping...")
        words = json.loads(rec.FinalResult()).get("result", [])
        register(words, censored, intervals, rel_start, args.pad)
        release(hist, rel_start, fed, intervals, args.volume, out_q)  # play what's left
    finally:
        out_q.put(None)
        player_thread.join()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Live microphone word censor (Vosk).")
    p.add_argument("--words", nargs="+", help="censored words, e.g. monday")
    p.add_argument("--delay", type=float, default=1.5,
                   help="seconds the audio is held back before playing (bigger = safer)")
    p.add_argument("--pad", type=float, default=0.1, help="extra seconds censored around each word")
    p.add_argument("--volume", type=float, default=0.25, help="static loudness 0-1")
    p.add_argument("--model-path", help="folder of a downloaded Vosk model (default: small English)")
    p.add_argument("--only-banned", action="store_true",
                   help="listen ONLY for the banned words (catches more, but more false bleeps)")
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