"""Tests for the hackathon entry point."""

from main import main


def test_main_prints_greeting(capsys):
    """main writes the project greeting to stdout."""
    main()
    captured = capsys.readouterr()
    assert captured.out == "Hello from compiled-hackathon!\n"
