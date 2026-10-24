"""
censor_audio.py

1. Transcribes an audio file with word-level timestamps (Whisper).
2. Looks for censored words in the transcript.
3. If none found -> returns the original audio untouched.
4. If found -> replaces just those words with static noise and saves a new file.

Install:
    pip install faster-whisper pydub numpy
    # plus ffmpeg on your system (brew install ffmpeg / apt install ffmpeg / choco install ffmpeg)

Usage:
    python censor_audio.py input.mp3 --words taco burrito
    python censor_audio.py input.wav --words taco --out clean.wav --model small
"""

import argparse
import os
import re
import shutil

import numpy as np
from faster_whisper import WhisperModel
from pydub import AudioSegment


# --------------------------------------------------------------------------- #
# 1. Transcription
# --------------------------------------------------------------------------- #
def transcribe_words(audio_path, model_size="base"):
    """Return (full_text, [(word, start_sec, end_sec), ...])."""
    model = WhisperModel(model_size, device="auto", compute_type="auto")
    segments, _info = model.transcribe(audio_path, word_timestamps=True)

    words, text_parts = [], []
    for seg in segments:
        text_parts.append(seg.text.strip())
        for w in seg.words or []:
            words.append((w.word.strip(), w.start, w.end))
    return " ".join(text_parts), words


# --------------------------------------------------------------------------- #
# 2. Finding censored words
# --------------------------------------------------------------------------- #
def normalize(word):
    return re.sub(r"[^\w']", "", word.lower())


def matches(word, censored):
    """Match the word itself plus simple plurals (taco -> tacos)."""
    w = normalize(word)
    return any(w in (c, c + "s", c + "es") for c in censored)


def find_censor_spans(words, censored, pad=0.08):
    """Return merged [(start_ms, end_ms), ...] regions to censor."""
    censored = {normalize(c) for c in censored}
    spans = []
    for word, start, end in words:
        if matches(word, censored):
            spans.append([max(0, int((start - pad) * 1000)), int((end + pad) * 1000)])

    # merge overlapping / touching spans
    merged = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


# --------------------------------------------------------------------------- #
# 3. Static noise + splicing
# --------------------------------------------------------------------------- #
def make_static(duration_ms, like, volume=0.25):
    """White noise matching the sample rate / channels of `like` (16-bit)."""
    n_frames = int(like.frame_rate * duration_ms / 1000)
    amp = int(32767 * volume)
    noise = np.random.randint(-amp, amp, size=n_frames * like.channels, dtype=np.int16)
    return AudioSegment(
        data=noise.tobytes(),
        sample_width=2,
        frame_rate=like.frame_rate,
        channels=like.channels,
    )


def apply_censor(audio, spans):
    audio = audio.set_sample_width(2)
    out = AudioSegment.empty()
    cursor = 0
    for start, end in spans:
        end = min(end, len(audio))
        out += audio[cursor:start]
        out += make_static(end - start, audio)
        cursor = end
    out += audio[cursor:]
    return out


# --------------------------------------------------------------------------- #
# 4. Main pipeline
# --------------------------------------------------------------------------- #
def censor_audio(input_path, censored_words, output_path=None, model_size="base"):
    base, ext = os.path.splitext(input_path)
    output_path = output_path or f"{base}_censored{ext}"

    text, words = transcribe_words(input_path, model_size)
    print("Transcript:", text)

    spans = find_censor_spans(words, censored_words)

    if not spans:
        print("No censored words found - returning original audio unedited.")
        if os.path.abspath(output_path) != os.path.abspath(input_path):
            shutil.copyfile(input_path, output_path)
        return output_path

    print(f"Found {len(spans)} censored section(s):")
    for s, e in spans:
        print(f"  {s / 1000:.2f}s -> {e / 1000:.2f}s")

    audio = AudioSegment.from_file(input_path)
    censored = apply_censor(audio, spans)
    censored.export(output_path, format=ext.lstrip(".") or "wav")
    print("Saved:", output_path)
    return output_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bleep out censored words in an audio file.")
    p.add_argument("input", help="path to audio file (mp3, wav, m4a, ...)")
    p.add_argument("--words", nargs="+", required=True, help="censored words, e.g. taco")
    p.add_argument("--out", help="output path (default: <input>_censored.<ext>)")
    p.add_argument("--model", default="base", help="whisper size: tiny/base/small/medium/large-v3")
    args = p.parse_args()

    censor_audio(args.input, args.words, args.out, args.model)