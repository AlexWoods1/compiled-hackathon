"""Tests for phrase matching and in-place censoring."""

from pathlib import Path

import numpy as np
import pytest

from live_censor import (
    CensorSpan,
    HeardWord,
    NOISE_VOL,
    apply_static,
    collect_topics,
    absorb_audio,
    drop_stale_audio,
    find_banned_spans,
    fresh_status,
    load_topics,
    matched_topics,
    overlap_sample_count,
    replace_with_static,
    should_report_backlog,
    split_boundary_spans,
    tokens_from_segments,
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


def test_drop_stale_audio_keeps_only_the_newest_chunk():
    """Audio older than the live cap is discarded instead of stretching the delay."""
    pending = np.arange(48000 * 5, dtype=np.float32)
    kept, dropped = drop_stale_audio(pending, 48000, 48000)
    assert dropped == 4.0
    assert len(kept) == 48000
    assert kept[0] == pending[-48000]


def test_absorb_audio_drops_the_queue_down_to_one_chunk():
    """A backed-up microphone queue keeps the newest chunk and drops the rest."""
    pending = np.zeros(0, dtype=np.float32)
    blocks = [np.full(48000, index, dtype=np.float32) for index in range(5)]
    kept, dropped = absorb_audio(pending, blocks, 48000, 48000)
    assert dropped == 4.0
    assert kept[0] == 4


def test_short_queue_is_not_dropped():
    """A normal chunk of pending audio stays intact."""
    pending = np.ones(48000, dtype=np.float32)
    kept, dropped = drop_stale_audio(pending, 48000, 48000)
    assert dropped == 0.0
    assert len(kept) == 48000


def test_boundary_word_is_split_across_the_overlap():
    """A phrase that crosses the chunk boundary censors both sides."""
    overlap = overlap_sample_count(48000, 48000)
    spans = [CensorSpan(start=0.10, end=0.50, topic="tacos")]
    tail, new = split_boundary_spans(spans, overlap, 48000)
    assert tail[0].start == 0.10
    assert tail[0].end == pytest.approx(overlap / 48000)
    assert new[0].start == 0.0
    assert new[0].end == pytest.approx(0.50 - overlap / 48000)


def test_phrase_match_ignores_word_times():
    """A banned phrase is found from the tokens alone."""
    hits = matched_topics(
        ["the", "customer", "unreleased", "earnings", "Tacos,"],
        ["tacos", "unreleased earnings", "customer list"],
    )
    assert hits == ["tacos", "unreleased earnings"]


def test_tokens_from_segments_skip_silence():
    """A segment the recognizer marks as silence contributes no tokens."""

    class Segment:
        def __init__(self, text, no_speech_prob):
            self.text = text
            self.no_speech_prob = no_speech_prob

    segments = [
        Segment("tacos", 0.9),
        Segment("hello list", 0.1),
    ]
    assert tokens_from_segments(segments) == ["hello", "list"]


def test_banned_chunk_is_replaced_with_static():
    """A hit censors every sample in the chunk."""
    samples = np.linspace(-0.2, 0.2, 800, dtype=np.float32)
    censored = replace_with_static(samples)
    assert not np.array_equal(censored, samples)
    assert len(censored) == len(samples)
    assert np.max(np.abs(censored)) <= NOISE_VOL


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
