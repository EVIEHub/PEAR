"""Standard QA scoring functions: accuracy, exact match, token F1.

The implementations follow the SQuAD-style normalisation used by HotpotQA,
but they are intentionally generic so the same code works for every
benchmark with the right :class:`Task`-level pre-processing.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable, List, Sequence


# Text normalisation
_PUNC_TABLE = str.maketrans({c: " " for c in string.punctuation})


def normalize_text(s: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace.

    This is the canonical SQuAD/HotpotQA normalisation. We use it for both
    EM and F1 so the metrics agree.
    """
    s = s.lower()
    s = s.translate(_PUNC_TABLE)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# Aggregate metrics
def accuracy(predictions: Sequence[bool]) -> float:
    """Fraction of ``True`` values in ``predictions``. Empty -> 0.0."""
    if not predictions:
        return 0.0
    return float(sum(1 for p in predictions if p)) / len(predictions)


def exact_match(prediction: str, gold: str) -> bool:
    """Normalised string equality."""
    return normalize_text(prediction) == normalize_text(gold)


def f1_score(prediction: str, gold: str) -> float:
    """Token-level F1 on normalised strings.

    Returns 1.0 if both strings are empty after normalisation, 0.0 if the
    intersection is empty, otherwise the harmonic mean of token precision
    and recall.
    """
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_macro(predictions: Iterable[str], golds: Iterable[str]) -> float:
    """Mean of per-example token F1."""
    pairs = list(zip(predictions, golds))
    if not pairs:
        return 0.0
    return sum(f1_score(p, g) for p, g in pairs) / len(pairs)


def em_macro(predictions: Iterable[str], golds: Iterable[str]) -> float:
    """Mean of per-example exact-match scores."""
    pairs = list(zip(predictions, golds))
    if not pairs:
        return 0.0
    return sum(1.0 for p, g in pairs if exact_match(p, g)) / len(pairs)
