# mic_audio_output

Stream masked `.wav` audio files directly to a virtual microphone input on your system so that Zoom, Discord, Teams, or any calling application can receive the audio as if it was spoken into a real microphone.

---

## Quick Start (with Pixi)

Navigate to the `mic_audio_output` folder:

```bash
cd mic_audio_output
```

### 1. List Available Audio Devices
Inspect all system audio devices, their indices, and auto-detect virtual cables:

```bash
pixi run list-devices
```

### 2. Stream Audio to Microphone
Stream a `.wav` file (automatically matches the sample rate, channel count, and bit depth of the file):

```bash
pixi run play -f path/to/masked.wav
```

### 3. Specify Target Device or Loop
```bash
# Explicit device by index or name
pixi run play -f path/to/masked.wav -d "CABLE Input"

# Loop audio continuously (great for testing Zoom levels)
pixi run play -f path/to/masked.wav --loop

# Inspect WAV specifications without playing
pixi run play -f path/to/masked.wav --info
```

### 4. Run Tests
All unit tests are located in `mic_audio_output/tests/`:

```bash
pixi run test
```

---

## Zoom Configuration Guide for Demos

1. **Install Virtual Cable** (if not already installed):
   - **Windows**: Install [VB-Audio Virtual Cable](https://vb-audio.com/Cable/) (free).
   - **macOS**: Install [BlackHole](https://existential.audio/blackhole/) (`brew install blackhole-2ch`).
2. **In Zoom**:
   - Open **Settings** -> **Audio**.
   - Under **Microphone**, select **CABLE Output (VB-Audio Virtual Cable)** (or **BlackHole 2ch**).
   - **Audio Profile**: Select **Original sound for musicians** (or set Background Noise Suppression to **Low**). This prevents Zoom from accidentally suppressing the static noise / beep mask applied to censored words.
3. **Run your masked audio**:
   ```bash
   pixi run play -f sample.wav
   ```
   Zoom's microphone input meter will react and the other participants in the call will hear the masked audio.
