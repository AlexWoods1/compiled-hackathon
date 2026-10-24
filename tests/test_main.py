"""Tests for the project entry point."""

import live_censor
import main


def test_entry_point_runs_the_censor():
    """python main.py starts the live censor."""
    assert main.main is live_censor.main
