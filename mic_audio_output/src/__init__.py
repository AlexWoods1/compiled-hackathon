"""mic_audio_output: Stream masked audio files directly to virtual microphone devices."""

from .device import get_output_device, list_devices
from .player import AudioSpecs, play_file, get_audio_specs

__all__ = [
    "get_output_device",
    "list_devices",
    "AudioSpecs",
    "play_file",
    "get_audio_specs",
]
