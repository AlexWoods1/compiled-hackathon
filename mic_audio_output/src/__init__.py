"""Stream masked audio files to virtual microphone devices."""

from .device import get_output_device, list_devices
from .player import AudioSpecs, get_audio_specs, play_file

__all__ = [
    "get_output_device",
    "list_devices",
    "AudioSpecs",
    "play_file",
    "get_audio_specs",
]
