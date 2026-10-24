"""Audio device discovery, listing, and virtual audio cable matching."""

from typing import Any, Dict, List, Optional, Tuple
import sounddevice as sd

# Common keywords for virtual audio cables across platforms
VIRTUAL_DEVICE_KEYWORDS = [
    "cable input",  # VB-Audio Cable (Windows)
    "vb-audio",  # VB-Audio Cable alternative name
    "blackhole",  # BlackHole (macOS)
    "virtual audio",  # General virtual audio drivers
    "loopback",  # ALSA / Pulse / Jack loopbacks
    "virtual cable",  # Virtual Audio Cable (VAC)
    "steam streaming",  # Steam virtual mic sink
    "voicemeeter",  # Voicemeeter aux/vaio
]


def list_devices() -> List[Dict[str, Any]]:
    """Return a list of all audio devices with their index and metadata."""
    raw_devices = sd.query_devices()
    devices = []
    for idx, d in enumerate(raw_devices):
        device_dict = dict(d)
        device_dict["index"] = idx
        devices.append(device_dict)
    return devices


def format_device_table(devices: Optional[List[Dict[str, Any]]] = None) -> str:
    """Format available audio devices into a readable text table."""
    if devices is None:
        devices = list_devices()

    lines = [
        f"{'Idx':<4} {'Type':<8} {'Channels (In/Out)':<18} "
        f"{'Sample Rate':<12} {'Device Name'}",
        "-" * 75,
    ]

    for d in devices:
        in_ch = d.get("max_input_channels", 0)
        out_ch = d.get("max_output_channels", 0)
        if in_ch > 0 and out_ch > 0:
            dev_type = "In/Out"
        elif out_ch > 0:
            dev_type = "Output"
        else:
            dev_type = "Input"
        sr = f"{int(d.get('default_samplerate', 0))} Hz"
        ch_str = f"{in_ch} in / {out_ch} out"
        lines.append(
            f"{d['index']:<4} {dev_type:<8} {ch_str:<18} "
            f"{sr:<12} {d.get('name', 'Unknown')}"
        )

    return "\n".join(lines)


def find_virtual_cable_device(
    devices: Optional[List[Dict[str, Any]]] = None,
) -> Optional[int]:
    """Find the index of a virtual audio cable output endpoint.

    Returns the device index if found, or None.
    """
    if devices is None:
        devices = list_devices()

    for d in devices:
        if d.get("max_output_channels", 0) <= 0:
            continue
        name_lower = d.get("name", "").lower()
        for kw in VIRTUAL_DEVICE_KEYWORDS:
            if kw in name_lower:
                return d["index"]

    return None


def get_output_device(
    device_query: Optional[str] = None,
    devices: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[int, str, bool]:
    """Resolve an output device target from a user query or auto-detection.

    Returns:
        (device_index, device_name, is_virtual_cable)
    """
    if devices is None:
        devices = list_devices()

    output_devices = [d for d in devices if d.get("max_output_channels", 0) > 0]
    if not output_devices:
        raise RuntimeError("No audio output devices found on the system.")

    # 1. User specified explicit device index or name
    if device_query is not None and str(device_query).strip() != "":
        query_str = str(device_query).strip()

        # Check if query is an integer index
        if query_str.isdigit():
            target_idx = int(query_str)
            match = next((d for d in output_devices if d["index"] == target_idx), None)
            if match:
                is_virtual = any(
                    kw in match["name"].lower() for kw in VIRTUAL_DEVICE_KEYWORDS
                )
                return match["index"], match["name"], is_virtual
            raise ValueError(f"Device index {target_idx} is not a valid output device.")

        # Substring search in device name
        matches = [d for d in output_devices if query_str.lower() in d["name"].lower()]
        if matches:
            match = matches[0]
            is_virtual = any(
                kw in match["name"].lower() for kw in VIRTUAL_DEVICE_KEYWORDS
            )
            return match["index"], match["name"], is_virtual

        raise ValueError(
            f"No output audio device matched query '{device_query}'. "
            "Run with --list-devices to see options."
        )

    # 2. Auto-detect virtual audio cable
    virtual_idx = find_virtual_cable_device(output_devices)
    if virtual_idx is not None:
        match = next(d for d in output_devices if d["index"] == virtual_idx)
        return match["index"], match["name"], True

    # 3. Fallback to default output device
    default_out_idx = sd.default.device[1]
    if default_out_idx is not None and default_out_idx >= 0:
        match = next((d for d in output_devices if d["index"] == default_out_idx), None)
        if match:
            return match["index"], match["name"], False

    # First available output device
    first = output_devices[0]
    return first["index"], first["name"], False
