"""Generate a sample WAV file for quick audio testing."""

import os
import numpy as np
import soundfile as sf


def generate_sample(output_path="sample.wav", duration=2.5, samplerate=44100):
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    # Generate pleasant 2-tone melodic chime
    tone1 = np.sin(2 * np.pi * 523.25 * t)  # C5
    tone2 = np.sin(2 * np.pi * 659.25 * t)  # E5
    envelope = np.exp(-t * 1.5)
    audio = ((tone1 + tone2) * 0.4 * envelope * 32767).astype(np.int16)

    # 2 channels (stereo)
    stereo_audio = np.column_stack([audio, audio])
    sf.write(output_path, stereo_audio, samplerate, subtype="PCM_16")
    print(f"Generated sample WAV at: {os.path.abspath(output_path)}")


if __name__ == "__main__":
    generate_sample()
