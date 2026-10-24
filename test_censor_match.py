"""Tests for the censor's word/phrase matching."""

from censor_match import find_matches, parse_topics


def words_from(text):
    """Fake Whisper output: one second per word."""
    return [(" " + w, float(i), i + 1.0) for i, w in enumerate(text.split())]


def test_parse_skips_comments_and_blanks():
    topics = parse_topics(["# secret stuff", "", "Taco", "Project Falcon!", "taco"])
    assert topics == [("taco",), ("project", "falcon")]


def test_single_word_and_plural():
    hits = find_matches(words_from("I'm making Tacos tonight."), [("taco",)])
    assert hits == [("Tacos", 2.0, 3.0)]


def test_phrase_spans_all_its_words():
    topics = parse_topics(["project falcon"])
    hits = find_matches(words_from("the project falcon launch"), topics)
    assert hits == [("project falcon", 1.0, 3.0)]


def test_partial_phrase_is_not_censored():
    topics = parse_topics(["project falcon"])
    assert find_matches(words_from("this project is late"), topics) == []


def test_no_match_leaves_nothing():
    assert find_matches(words_from("hello everyone"), [("taco",)]) == []