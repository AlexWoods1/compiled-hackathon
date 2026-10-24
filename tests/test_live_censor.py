"""Tests for phrase matching and in-place censoring."""

from pathlib import Path

import numpy as np
import pytest

from live_censor import (
    CensorSpan,
    HeardWord,
    apply_static,
    collect_topics,
    find_banned_spans,
    fresh_status,
    load_topics,
    should_report_backlog,
    tokens_match,
    words_from_segments,
)

DEMO_TOPICS = Path(__file__).resolve().parents[1] / "demo" / "private_topics.txt"


def test_demo_topics_include_the_dinner_word():
    """The demo topics file bans the dinner buzz word and a multi-word secret."""
    phrases = load_topics(DEMO_TOPICS)
    assert "tacos" in phrases
    assert "production database password" in phrases


def test_tokens_match_simple_plurals():
    """A trailing s or es still counts as the banned word."""
    assert tokens_match("tacos", "taco")
    assert tokens_match("taco", "tacos")
    assert tokens_match("boxes", "box")
    assert not tokens_match("customer", "list")


def test_phrase_must_match_as_a_whole():
    """A single word from a longer topic stays in the clear."""
    words = [
        HeardWord("the", 0.0, 0.2),
        HeardWord("customer", 0.2, 0.6),
        HeardWord("unreleased", 0.7, 1.1),
        HeardWord("earnings", 1.1, 1.5),
        HeardWord("Tacos,", 1.6, 2.0),
    ]
    spans = find_banned_spans(
        words,
        ["tacos", "unreleased earnings", "customer list"],
    )
    assert [(span.start, span.end, span.topic) for span in spans] == [
        (0.7, 1.5, "unreleased earnings"),
        (1.6, 2.0, "tacos"),
    ]


def test_three_word_topic_matches_in_order():
    """The password phrase matches only as three consecutive words."""
    words = [
        HeardWord("production", 0.0, 0.4),
        HeardWord("database", 0.4, 0.8),
        HeardWord("password", 0.8, 1.2),
    ]
    spans = find_banned_spans(words, ["production database password"])
    assert len(spans) == 1
    assert spans[0].start == 0.0
    assert spans[0].end == 1.2


def test_load_topics_skips_blanks_and_comments(tmp_path):
    """Blank lines and comments are not phrases."""
    path = tmp_path / "topics.txt"
    path.write_text("# note\n\ntacos\n", encoding="utf-8")
    assert load_topics(path) == ["tacos"]


def test_collect_topics_adds_command_line_phrases(tmp_path):
    """Extra --words phrases are appended to the file."""
    path = tmp_path / "topics.txt"
    path.write_text("tacos\n", encoding="utf-8")
    assert collect_topics(path, ["customer list"]) == ["tacos", "customer list"]


def test_collect_topics_rejects_an_empty_source(tmp_path):
    """An empty topics file with no extra words is an error."""
    path = tmp_path / "topics.txt"
    path.write_text("# nothing\n", encoding="utf-8")
    with pytest.raises(ValueError):
        collect_topics(path, [])


def test_words_from_segments_skip_silence():
    """A segment the recognizer marks as silence contributes no words."""

    class Word:
        def __init__(self, word, start, end):
            self.word = word
            self.start = start
            self.end = end

    class Segment:
        def __init__(self, words, no_speech_prob):
            self.words = words
            self.no_speech_prob = no_speech_prob

    segments = [
        Segment([Word("tacos", 0.1, 0.4)], 0.9),
        Segment([Word("hello", None, 0.2)], 0.1),
        Segment([Word("list", 0.5, 0.8)], 0.1),
    ]
    assert words_from_segments(segments) == [HeardWord("list", 0.5, 0.8)]


def test_backlog_warning_stays_quiet_for_the_live_buffer():
    """A few seconds of delay is the censor buffer, not a failure."""
    assert should_report_backlog(2.5, 1.5, now=20.0, last_report=0.0) is False
    assert should_report_backlog(10.0, 1.5, now=20.0, last_report=0.0) is True
    assert should_report_backlog(10.0, 1.5, now=25.0, last_report=20.0) is False


def test_stream_status_prints_once():
    """Repeated overflow messages do not flood the console."""
    seen = set()
    assert fresh_status("input overflow", seen) == "input overflow"
    assert fresh_status("input overflow", seen) is None
    assert fresh_status("output underflow", seen) == "output underflow"


def test_apply_static_changes_only_the_matched_span():
    """Static covers the banned interval and the samples outside it stay put."""
    samples = np.linspace(-0.2, 0.2, 48000, dtype=np.float32)
    span = CensorSpan(start=1.0, end=1.2, topic="tacos")
    censored = apply_static(samples, [span], 48000)
    start = int((1.0 - 0.08) * 48000)
    end = int((1.2 + 0.08) * 48000)
    assert not np.array_equal(censored[start:end], samples[start:end])
    assert np.array_equal(censored[:start], samples[:start])
    assert np.array_equal(censored[end:], samples[end:])
