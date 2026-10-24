"""Tests for mic_audio_output audio spec matching, device detection, and streaming."""

import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import soundfile as sf

try:
    from src import device as device_mod
    from src import player as player_mod
    from src import cli as cli_mod
    from src.device import (
        find_virtual_cable_device,
        format_device_table,
        get_output_device,
    )
    from src.player import (
        AudioSpecs,
        get_audio_specs,
        play_file,
        subtype_to_dtype,
    )
    from src.cli import main
except ImportError:
    from mic_audio_output.src import device as device_mod
    from mic_audio_output.src import player as player_mod
    from mic_audio_output.src import cli as cli_mod
    from mic_audio_output.src.device import (
        find_virtual_cable_device,
        format_device_table,
        get_output_device,
    )
    from mic_audio_output.src.player import (
        AudioSpecs,
        get_audio_specs,
        play_file,
        subtype_to_dtype,
    )
    from mic_audio_output.src.cli import main


# --------------------------------------------------------------------------- #
# Fixtures for synthetic WAV files with varying specifications
# --------------------------------------------------------------------------- #
@pytest.fixture
def wav_16k_mono(tmp_path):
    """Generate a 16kHz mono 16-bit PCM WAV file."""
    path = str(tmp_path / "mono_16k.wav")
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    sf.write(path, data, sr, subtype="PCM_16")
    return path, sr, 1, "PCM_16", "int16"


@pytest.fixture
def wav_44k_stereo(tmp_path):
    """Generate a 44.1kHz stereo 16-bit PCM WAV file."""
    path = str(tmp_path / "stereo_44k.wav")
    sr = 44100
    duration = 0.5
    frames = int(sr * duration)
    t = np.linspace(0, duration, frames, endpoint=False)
    left = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    right = (np.cos(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    data = np.column_stack([left, right])
    sf.write(path, data, sr, subtype="PCM_16")
    return path, sr, 2, "PCM_16", "int16"


@pytest.fixture
def wav_48k_stereo_float(tmp_path):
    """Generate a 48kHz stereo 32-bit float WAV file."""
    path = str(tmp_path / "stereo_48k_float.wav")
    sr = 48000
    duration = 0.5
    frames = int(sr * duration)
    t = np.linspace(0, duration, frames, endpoint=False)
    left = np.sin(2 * np.pi * 880 * t)
    right = np.cos(2 * np.pi * 880 * t)
    data = np.column_stack([left, right]).astype(np.float32)
    sf.write(path, data, sr, subtype="FLOAT")
    return path, sr, 2, "FLOAT", "float32"


# --------------------------------------------------------------------------- #
# Audio Specification Matching Tests
# --------------------------------------------------------------------------- #
def test_audio_specs_matching_16k_mono(wav_16k_mono):
    path, expected_sr, expected_ch, expected_sub, expected_dtype = wav_16k_mono
    specs = get_audio_specs(path)
    assert specs.samplerate == expected_sr
    assert specs.channels == expected_ch
    assert specs.subtype == expected_sub
    assert specs.dtype == expected_dtype
    assert specs.duration_seconds == pytest.approx(0.5, rel=1e-2)


def test_audio_specs_matching_44k_stereo(wav_44k_stereo):
    path, expected_sr, expected_ch, expected_sub, expected_dtype = wav_44k_stereo
    specs = get_audio_specs(path)
    assert specs.samplerate == expected_sr
    assert specs.channels == expected_ch
    assert specs.subtype == expected_sub
    assert specs.dtype == expected_dtype


def test_audio_specs_matching_48k_float(wav_48k_stereo_float):
    path, expected_sr, expected_ch, expected_sub, expected_dtype = (
        wav_48k_stereo_float
    )
    specs = get_audio_specs(path)
    assert specs.samplerate == expected_sr
    assert specs.channels == expected_ch
    assert specs.subtype == expected_sub
    assert specs.dtype == expected_dtype


def test_subtype_to_dtype_mappings():
    assert subtype_to_dtype("PCM_16") == "int16"
    assert subtype_to_dtype("PCM_24") == "int32"
    assert subtype_to_dtype("PCM_32") == "int32"
    assert subtype_to_dtype("FLOAT") == "float32"
    assert subtype_to_dtype("PCM_U8") == "uint8"
    assert subtype_to_dtype("UNKNOWN_SUBTYPE") == "float32"


# --------------------------------------------------------------------------- #
# Device Detection & Virtual Cable Resolution Tests
# --------------------------------------------------------------------------- #
MOCK_DEVICES = [
    {
        "index": 0,
        "name": "Realtek High Definition Audio",
        "max_input_channels": 2,
        "max_output_channels": 0,
        "default_samplerate": 48000.0,
    },
    {
        "index": 1,
        "name": "Speakers (Realtek Audio)",
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 48000.0,
    },
    {
        "index": 2,
        "name": "CABLE Input (VB-Audio Virtual Cable)",
        "max_input_channels": 0,
        "max_output_channels": 2,
        "default_samplerate": 44100.0,
    },
    {
        "index": 3,
        "name": "BlackHole 2ch",
        "max_input_channels": 2,
        "max_output_channels": 2,
        "default_samplerate": 44100.0,
    },
]


def test_find_virtual_cable_detected():
    idx = find_virtual_cable_device(MOCK_DEVICES)
    assert idx == 2  # Matches CABLE Input


def test_get_output_device_auto_virtual():
    idx, name, is_virtual = get_output_device(
        device_query=None, devices=MOCK_DEVICES
    )
    assert idx == 2
    assert "VB-Audio" in name
    assert is_virtual is True


def test_get_output_device_explicit_index():
    idx, name, is_virtual = get_output_device(
        device_query="1", devices=MOCK_DEVICES
    )
    assert idx == 1
    assert "Speakers" in name
    assert is_virtual is False


def test_get_output_device_explicit_name_substring():
    idx, name, is_virtual = get_output_device(
        device_query="blackhole", devices=MOCK_DEVICES
    )
    assert idx == 3
    assert "BlackHole" in name
    assert is_virtual is True


def test_get_output_device_invalid_index():
    with pytest.raises(ValueError, match="not a valid output device"):
        get_output_device(device_query="99", devices=MOCK_DEVICES)


def test_format_device_table():
    table = format_device_table(MOCK_DEVICES)
    assert "Speakers" in table
    assert "CABLE Input" in table
    assert "48000 Hz" in table


# --------------------------------------------------------------------------- #
# Playback Stream Exact Spec Matching Verification
# --------------------------------------------------------------------------- #
def test_play_file_passes_exact_audio_specs(wav_16k_mono):
    path, expected_sr, expected_ch, _, expected_dtype = wav_16k_mono

    mock_stream_instance = MagicMock()
    mock_stream_class = MagicMock(return_value=mock_stream_instance)

    with patch.object(player_mod.sd, "OutputStream", mock_stream_class), patch.object(
        player_mod,
        "get_output_device",
        return_value=(2, "CABLE Input", True),
    ):
        play_file(path, device=None, blocksize=512)

        # Assert OutputStream was configured with exact specifications from the WAV file
        mock_stream_class.assert_called_once()
        _, kwargs = mock_stream_class.call_args
        assert kwargs["samplerate"] == expected_sr
        assert kwargs["channels"] == expected_ch
        assert kwargs["dtype"] == expected_dtype
        assert kwargs["device"] == 2

        # Assert data was written to stream
        assert mock_stream_instance.write.called


# --------------------------------------------------------------------------- #
# CLI Tests
# --------------------------------------------------------------------------- #
def test_cli_info_flag(wav_44k_stereo, capsys):
    path, _, _, _, _ = wav_44k_stereo
    exit_code = main(["--file", path, "--info"])
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "44100 Hz" in captured
    assert "Stereo (2ch)" in captured


def test_cli_list_devices(capsys):
    with patch.object(cli_mod, "list_devices", return_value=MOCK_DEVICES):
        exit_code = main(["--list-devices"])
        assert exit_code == 0
        captured = capsys.readouterr().out
        assert "VB-Audio Virtual Cable" in captured
