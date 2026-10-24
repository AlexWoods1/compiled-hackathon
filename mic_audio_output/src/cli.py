"""Command-line interface for mic_audio_output."""

import argparse
import sys
from .device import format_device_table, list_devices
from .player import get_audio_specs, play_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mic_audio_output",
        description=(
            "Stream audio files directly to a virtual microphone for Zoom and voice apps."
        ),
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="filepath",
        type=str,
        help="Path to the .wav audio file to play.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=None,
        help=(
            "Target output device index or substring name "
            "(default: auto-detect virtual cable)."
        ),
    )
    parser.add_argument(
        "-l",
        "--list-devices",
        action="store_true",
        help="List all detected audio devices with indices and exit.",
    )
    parser.add_argument(
        "-v",
        "--volume",
        type=float,
        default=1.0,
        help="Volume multiplier (default: 1.0).",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Continuously loop audio playback until interrupted (Ctrl+C).",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Inspect and print incoming WAV specifications without playing.",
    )
    return parser


def main(args=None):
    parser = build_parser()
    parsed = parser.parse_args(args)

    if parsed.list_devices:
        print("\nAvailable Audio Devices:")
        print(format_device_table(list_devices()))
        return 0

    if not parsed.filepath:
        parser.print_help()
        print(
            "\nError: Please specify an audio file using --file / -f, "
            "or use --list-devices."
        )
        return 1

    try:
        specs = get_audio_specs(parsed.filepath)
        if parsed.info:
            print(f"File: {parsed.filepath}")
            print(specs)
            return 0

        play_file(
            filepath=parsed.filepath,
            device=parsed.device,
            volume=parsed.volume,
            loop=parsed.loop,
        )
        return 0
    except KeyboardInterrupt:
        print("\n[!] Playback stopped by user.")
        return 0
    except Exception as e:
        print(f"\n[!] Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
