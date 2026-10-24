"""
live_censor.py - near-real-time censor:
    your mic -> Whisper (word timestamps) -> static over banned words -> virtual mic -> Zoom

Setup:
    pip install faster-whisper sounddevice numpy
    Mac:     brew install blackhole-2ch          (creates device "BlackHole 2ch")
    Windows: install VB-Audio Virtual Cable      (creates "CABLE Input" / "CABLE Output")

    In Zoom > Settings > Audio:
      - Microphone = "BlackHole 2ch" (Mac) or "CABLE Output" (Windows)
      - Uncheck "Automatically adjust microphone volume"
      - Background noise suppression = Low  (otherwise Zoom may strip out the static)

Usage:
    python live_censor.py --list-devices
    python live_censor.py --words taco burrito                      # self-test: plays to your speakers/headphones
    python live_censor.py --topics topics.txt                       # one word/phrase per line
    python live_censor.py --words taco --out-device "BlackHole"     # Mac -> Zoom
    python live_censor.py --words taco --out-device "CABLE Input"   # Windows -> Zoom

How it works:
    Audio is cut into CHUNK-second pieces. Each step transcribes [previous chunk + current chunk]
    so words that straddle a chunk boundary still get caught, then plays the previous chunk.
    Delay heard by the other person is about 2 x chunk + processing time (~3-4 s at defaults).
"""

import argparse
import queue
import sys
import threading

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from censor_match import find_matches, load_topics, parse_topics

DEV_SR = 48000  # device sample rate (widely supported by mics and virtual cables)
DOWN = 3  # 48 kHz -> 16 kHz for Whisper
PAD = 0.08  # seconds of padding around each bleeped word
NOISE_VOL = 0.15


def find_device(name, kind):
    if name is None:
        return None
    for i, d in enumerate(sd.query_devices()):
        if name.lower() in d["name"].lower() and d[f"max_{kind}_channels"] > 0:
            return i
    sys.exit(f"No {kind} device matching '{name}'. Run with --list-devices.")


def main():
    ap = argparse.ArgumentParser(description="Live-bleep banned words from your mic.")
    ap.add_argument("--words", nargs="+", help="banned words, e.g. taco burrito")
    ap.add_argument("--topics", help="file with one banned word/phrase per line")
    ap.add_argument("--in-device", help="substring of mic name (default: system mic)")
    ap.add_argument(
        "--out-device", help="substring of output name, e.g. BlackHole / 'CABLE Input'"
    )
    ap.add_argument("--model", default="base.en", help="tiny.en / base.en / small.en")
    ap.add_argument("--chunk", type=float, default=1.5, help="chunk length in seconds")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return
    phrases = parse_topics(args.words or [])
    if args.topics:
        phrases += [p for p in load_topics(args.topics) if p not in phrases]
    if not phrases:
        sys.exit("Pass --words taco and/or --topics topics.txt")

    labels = [" ".join(p) for p in phrases]
    prompt = (
        "Words that may come up: " + ", ".join(labels) + "."
    )  # biases Whisper to spell them right

    print(f"Loading Whisper '{args.model}'...")
    model = WhisperModel(args.model, device="auto", compute_type="auto")  # load ONCE

    chunk_n = (int(args.chunk * DEV_SR) // DOWN) * DOWN
    in_q = queue.Queue()
    out_buf = np.zeros(0, dtype=np.float32)
    lock = threading.Lock()

    # ---------------- audio callbacks ----------------
    def on_input(indata, frames, t, status):
        in_q.put(indata[:, 0].copy())

    def on_output(outdata, frames, t, status):
        nonlocal out_buf
        outdata[:] = 0
        with lock:
            n = min(frames, len(out_buf))
            outdata[:n, 0] = out_buf[:n]
            out_buf = out_buf[n:]

    def emit(samples):
        nonlocal out_buf
        with lock:
            out_buf = np.concatenate([out_buf, samples])

    # ---------------- censoring ----------------
    def censor(window):
        audio16 = window.reshape(-1, DOWN).mean(axis=1)
        segments, _ = model.transcribe(
            audio16,
            language="en",
            word_timestamps=True,
            beam_size=1,
            initial_prompt=prompt,
            condition_on_previous_text=False,
        )
        out = window.copy()
        words = [(w.word, w.start, w.end) for seg in segments for w in seg.words or []]
        if words:
            print("heard:", " ".join(w[0].strip() for w in words))
        for text, start, end in find_matches(words, phrases):
            s = max(0, int((start - PAD) * DEV_SR))
            e = min(len(out), int((end + PAD) * DEV_SR))
            noise = np.random.uniform(-NOISE_VOL, NOISE_VOL, e - s)
            out[s:e] = noise.astype(np.float32)
            print(f"   BLEEPED: '{text}'")
        return out

    def worker():
        pending = np.zeros(0, dtype=np.float32)
        prev = None
        while True:
            pending = np.concatenate([pending, in_q.get()])
            if len(pending) < chunk_n:
                continue
            cur, pending = pending[:chunk_n].copy(), pending[chunk_n:]

            if prev is None:
                prev = censor(cur)
                continue

            window = censor(np.concatenate([prev, cur]))
            emit(window[: len(prev)])  # play previous chunk (now checked twice)
            prev = window[len(prev) :]  # keep current chunk (already censored once)

            backlog = in_q.qsize() * 1024 / DEV_SR
            if backlog > args.chunk:
                print(
                    f"   [warn] falling behind by {backlog:.1f}s - try --model tiny.en"
                )

    in_dev = find_device(args.in_device, "input")
    out_dev = find_device(args.out_device, "output")
    threading.Thread(target=worker, daemon=True).start()

    with sd.InputStream(
        device=in_dev,
        samplerate=DEV_SR,
        channels=1,
        dtype="float32",
        blocksize=1024,
        callback=on_input,
    ), sd.OutputStream(
        device=out_dev,
        samplerate=DEV_SR,
        channels=1,
        dtype="float32",
        blocksize=1024,
        callback=on_output,
    ):
        print(f"Live. Censoring: {', '.join(labels)}. Ctrl+C to stop.")
        try:
            while True:
                sd.sleep(1000)
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()