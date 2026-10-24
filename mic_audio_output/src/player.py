"""Audio player that matches incoming WAV specifications and streams to virtual microphone."""

from dataclasses import dataclass
import os
import sys
import time
from typing import Callable, Optional
import numpy as np
import sounddevice as sd
import soundfile as sf

from .device import get_output_device


@dataclass
class AudioSpecs:
    """Audio specifications extracted directly from the audio file."""

    samplerate: int
    channels: int
    subtype: str
    format: str
    frames: int
    duration_seconds: float
    dtype: str

    def __str__(self) -> str:
        ch_label = "Mono (1ch)" if self.channels == 1 else f"Stereo ({self.channels}ch)"
        return (
            f"Sample Rate: {self.samplerate} Hz | Channels: {ch_label} | "
            f"Subtype: {self.subtype} ({self.dtype}) | "
            f"Duration: {self.duration_seconds:.2f}s"
        )


def subtype_to_dtype(subtype: str) -> str:
    """Map soundfile subtype to sounddevice compatible numpy dtype."""
    subtype_upper = subtype.upper()
    if subtype_upper == "PCM_16":
        return "int16"
    if subtype_upper in ("PCM_24", "PCM_32"):
        return "int32"
    if subtype_upper in ("FLOAT", "DOUBLE"):
        return "float32"
    if subtype_upper == "PCM_U8":
        return "uint8"
    # Default to float32 for maximum compatibility and precision
    return "float32"


def get_audio_specs(filepath: str) -> AudioSpecs:
    """Inspect and return the exact audio specifications of a sound file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Audio file not found: {filepath}")

    with sf.SoundFile(filepath) as f:
        duration = float(f.frames) / float(f.samplerate) if f.samplerate > 0 else 0.0
        dtype = subtype_to_dtype(f.subtype)
        return AudioSpecs(
            samplerate=f.samplerate,
            channels=f.channels,
            subtype=f.subtype,
            format=f.format,
            frames=f.frames,
            duration_seconds=duration,
            dtype=dtype,
        )


def play_file(
    filepath: str,
    device: Optional[str] = None,
    volume: float = 1.0,
    blocksize: int = 2048,
    loop: bool = False,
    progress_callback: Optional[Callable[[float, float], None]] = None,
) -> None:
    """Stream a WAV file to the target audio output device matching WAV specifications.

    Args:
        filepath: Path to the .wav file to stream.
        device: Device index or substring name (None for virtual cable auto-detect).
        volume: Volume multiplier (1.0 = unchanged).
        blocksize: Number of frames per chunk.
        loop: Whether to loop playback continuously until interrupted.
        progress_callback: Optional callback(current_sec, total_sec).
    """
    specs = get_audio_specs(filepath)
    target_idx, target_name, is_virtual = get_output_device(device)

    print(f"[*] Audio File: {os.path.abspath(filepath)}")
    print(f"[*] Incoming Specs: {specs}")
    print(f"[*] Target Output Device [{target_idx}]: {target_name}")
    if is_virtual:
        print(
            "    -> Virtual audio cable detected! Audio will route to virtual microphone."
        )
    else:
        print(
            "    -> Warning: Not a recognized virtual cable. Audio will play to standard output."
        )

    while True:
        with sf.SoundFile(filepath) as f:
            # We attempt streaming with the exact sample rate and channel count
            stream_channels = specs.channels
            needs_channel_duplication = False

            try:
                out_stream = sd.OutputStream(
                    samplerate=specs.samplerate,
                    channels=stream_channels,
                    dtype=specs.dtype,
                    device=target_idx,
                    blocksize=blocksize,
                )
            except Exception as e:
                # If the device driver strictly enforces stereo (2 channels) while WAV is mono
                if specs.channels == 1:
                    print(
                        "[*] Driver requested adaptation: attempting stereo streaming "
                        "with duplicated channels..."
                    )
                    stream_channels = 2
                    needs_channel_duplication = True
                    out_stream = sd.OutputStream(
                        samplerate=specs.samplerate,
                        channels=2,
                        dtype=specs.dtype,
                        device=target_idx,
                        blocksize=blocksize,
                    )
                else:
                    raise e

            with out_stream:
                frames_played = 0
                total_frames = specs.frames

                while True:
                    data = f.read(blocksize, dtype=specs.dtype)
                    if len(data) == 0:
                        break

                    # Apply volume scaling if modified
                    if volume != 1.0:
                        data = data * volume
                        if "int" in specs.dtype:
                            # Clamp integer ranges to prevent overflow
                            info = np.iinfo(data.dtype)
                            data = np.clip(data, info.min, info.max).astype(specs.dtype)

                    # Adapt mono to stereo if the hardware driver requires 2 channels
                    if needs_channel_duplication:
                        if data.ndim == 1:
                            data = np.column_stack([data, data])
                        elif data.shape[1] == 1:
                            data = np.repeat(data, 2, axis=1)

                    out_stream.write(data)
                    frames_played += len(data)

                    current_sec = min(
                        specs.duration_seconds, frames_played / specs.samplerate
                    )
                    if progress_callback:
                        progress_callback(current_sec, specs.duration_seconds)
                    else:
                        pct = (
                            (frames_played / total_frames) * 100
                            if total_frames > 0
                            else 100
                        )
                        sys.stdout.write(
                            f"\rStreaming: {pct:5.1f}% "
                            f"[{current_sec:5.1f}s / {specs.duration_seconds:5.1f}s]"
                        )
                        sys.stdout.flush()

                sys.stdout.write("\n")
                sys.stdout.flush()

        if not loop:
            break
        print("[*] Looping playback... (Press Ctrl+C to stop)")
        time.sleep(0.1)

    print("[✓] Playback complete.")
