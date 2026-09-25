"""Country-independent text normalization and pair similarity features."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


LEGAL = {"corporation": "corp", "incorporated": "inc", "limited": "ltd",
         "private": "pvt", "company": "co"}
FEATURE_NAMES = [
    "name_ratio", "name_token_ratio", "name_jaro", "name_jaccard",
    "address_ratio", "address_token_ratio", "address_jaccard",
    "number_jaccard", "number_agreement", "number_conflict",
    "name_length_ratio", "address_length_ratio", "name_exact",
    "address_exact", "country_agreement", "candidate_is_s3",
    "name_missing", "address_missing", "shared_postal_like",
]


def normalize(value: str, *, name: bool = False) -> str:
    text = unicodedata.normalize("NFKC", (value or "").casefold()).replace("&", " and ")
    text = "".join(char if char.isalnum() else " " for char in text)
    tokens = text.split()
    if name:
        tokens = [LEGAL.get(token, token) for token in tokens]
    return " ".join(tokens)


def tokens(text: str) -> set[str]:
    return set(text.split())


def numbers(text: str) -> set[str]:
    return set(re.findall(r"(?<!\w)\d+(?!\w)", text))


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def length_ratio(a: str, b: str) -> float:
    return min(len(a), len(b)) / max(len(a), len(b)) if a and b else 0.0


def keys(name: str, address: str) -> tuple[str, str, str]:
    """Three modest-recall lookup keys; empty keys never form blocks."""
    n = normalize(name, name=True)
    a = normalize(address)
    words = [word for word in n.split() if len(word) >= 3 and word not in LEGAL.values()]
    prefix = "".join(n.split())[:6]
    long_token = sorted(words, key=lambda word: (-len(word), word))[0][:10] if words else ""
    address_words = [word for word in a.split() if len(word) >= 4 and not word.isdigit()]
    address_token = sorted(address_words, key=lambda word: (-len(word), word))[0][:8] if address_words else ""
    numeric = sorted(numbers(a), key=lambda value: (-len(value), value))
    address_key = numeric[0] + ":" + address_token if numeric and address_token else ""
    return prefix, long_token, address_key


def features(left: tuple[str, str, str, str], right: tuple[str, str, str, str]) -> list[float]:
    """Input tuples contain entity_id, name, address, country."""
    name_a, name_b = normalize(left[1], name=True), normalize(right[1], name=True)
    addr_a, addr_b = normalize(left[2]), normalize(right[2])
    nums_a, nums_b = numbers(addr_a), numbers(addr_b)
    post_a = {n for n in nums_a if 5 <= len(n) <= 6}
    post_b = {n for n in nums_b if 5 <= len(n) <= 6}
    return [
        fuzz.ratio(name_a, name_b) / 100,
        fuzz.token_set_ratio(name_a, name_b) / 100,
        JaroWinkler.normalized_similarity(name_a, name_b),
        jaccard(tokens(name_a), tokens(name_b)),
        fuzz.ratio(addr_a, addr_b) / 100,
        fuzz.token_set_ratio(addr_a, addr_b) / 100,
        jaccard(tokens(addr_a), tokens(addr_b)),
        jaccard(nums_a, nums_b),
        float(bool(nums_a & nums_b)),
        float(bool(nums_a and nums_b and not nums_a & nums_b)),
        length_ratio(name_a, name_b),
        length_ratio(addr_a, addr_b),
        float(bool(name_a) and name_a == name_b),
        float(bool(addr_a) and addr_a == addr_b),
        float(bool(left[3]) and left[3] == right[3]),
        float(right[0].startswith("S3-")),
        float(not name_a or not name_b),
        float(not addr_a or not addr_b),
        float(bool(post_a & post_b)),
    ]
