"""Pure-Python matching logic for the censor (no audio deps, so CI can test it).

A topics file has one sensitive word or phrase per line:

    # comments and blank lines are ignored
    taco
    project falcon
    q3 revenue

Single words also match simple plurals (taco -> tacos). Multi-word phrases match
when their words appear consecutively in the transcript.
"""

import re


def normalize(word):
    return re.sub(r"[^\w']", "", word.lower())


def load_topics(path):
    """Read a topics file -> list of phrases (each a tuple of normalized words)."""
    with open(path, encoding="utf-8") as f:
        return parse_topics(f.read().splitlines())


def parse_topics(lines):
    phrases = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = tuple(t for t in (normalize(w) for w in line.split()) if t)
        if tokens and tokens not in phrases:
            phrases.append(tokens)
    return phrases


def _word_matches(word, target):
    return word in (target, target + "s", target + "es")


def find_matches(words, phrases):
    """Find banned phrases in a transcript.

    words:   list of (text, start_sec, end_sec) from Whisper
    phrases: output of parse_topics / load_topics
    returns: list of (matched_text, start_sec, end_sec)
    """
    norm = [normalize(w[0]) for w in words]
    hits = []
    for i in range(len(words)):
        for phrase in phrases:
            n = len(phrase)
            if i + n > len(words):
                continue
            if all(_word_matches(norm[i + k], phrase[k]) for k in range(n)):
                text = " ".join(w[0].strip() for w in words[i : i + n])
                hits.append((text, words[i][1], words[i + n - 1][2]))
    return hits