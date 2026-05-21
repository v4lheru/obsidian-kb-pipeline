"""
src/embeddings.py -- Optional semantic-embedding helpers for paraphrase dedup.

Wraps `model2vec` with graceful no-op fallbacks when the optional dependency
is absent. The pipeline's exact-text dedup (`synthesizer._exact_and_prefix
_dedup` equivalent) remains the primary filter; this module adds a second
pass that catches paraphrases the first pass misses.

When `model2vec` is not installed, `is_available()` returns False and every
public helper returns its inputs unchanged after logging one INFO line per
process. This module never raises -- semantic dedup is enhancement, not gate.

Used by `synthesizer._deduplicate_extractions` after pass-1 exact dedup, and
by `synthesizer._synthesize_new` for intra-batch paraphrase dedup.
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

from .state_store import Extraction

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Truncate any single text before embedding to bound per-vector runtime.
# model2vec is mean-pooled static embeddings: 4000 chars stays sub-millisecond
# per text on a typical CPU.
_MAX_EMBED_CHARS = 4000

# HuggingFace model id loaded lazily on first embed call. Cached to
# `~/.cache/huggingface/` on first use; subsequent runs hit the cache.
_MODEL_ID = "minishlab/potion-base-8M"


@functools.lru_cache(maxsize=1)
def is_available() -> bool:
    """True iff `model2vec` is importable. Logs one INFO line per process when
    unavailable. lru_cache ensures the import-attempt and log fire exactly once.
    """
    try:
        import model2vec  # noqa: F401
        return True
    except ImportError:
        logger.info(
            "model2vec not installed; semantic dedup disabled. Install with "
            "`pip install obsidian-kb-pipeline[embeddings]` for paraphrase "
            "detection."
        )
        return False


@functools.lru_cache(maxsize=1)
def _load_model() -> Any | None:
    """Lazy-load StaticModel on first call. Returns None on any failure
    (missing dep, offline + no cache, HF Hub error). Cached so model loads at
    most once per process.
    """
    if not is_available():
        return None
    try:
        from model2vec import StaticModel
        return StaticModel.from_pretrained(_MODEL_ID)
    except Exception as exc:  # noqa: BLE001 -- model load failures are diverse
        logger.warning(
            "Failed to load model2vec model %s: %s. Semantic dedup disabled "
            "for the rest of this process.", _MODEL_ID, exc,
        )
        return None


def embed_texts(texts: list[str]) -> "np.ndarray | None":
    """Embed a list of strings into a 2D float32 array of shape (N, D).

    Returns None if model2vec is missing OR the model failed to load. Empty
    input returns None as well (no work to do).

    Texts are truncated to ~4000 chars before embedding to bound per-call
    runtime. model2vec handles batching internally.
    """
    if not texts:
        return None
    model = _load_model()
    if model is None:
        return None
    truncated = [t[:_MAX_EMBED_CHARS] for t in texts]
    try:
        vectors = model.encode(truncated)
    except Exception as exc:  # noqa: BLE001
        logger.warning("model2vec encode failed: %s. Skipping semantic pass.", exc)
        return None
    return vectors


def _cosine_max_row(candidate_vecs: "np.ndarray", existing_vecs: "np.ndarray") -> "np.ndarray":
    """For each row in candidate_vecs, compute the MAX cosine similarity against
    any row in existing_vecs. Returns a 1D array of length candidate_vecs.shape[0].
    """
    import numpy as np

    cand = candidate_vecs.astype("float32")
    exist = existing_vecs.astype("float32")

    cand_norm = np.linalg.norm(cand, axis=1, keepdims=True)
    exist_norm = np.linalg.norm(exist, axis=1, keepdims=True)

    cand_unit = np.divide(cand, cand_norm, out=np.zeros_like(cand), where=cand_norm > 0)
    exist_unit = np.divide(exist, exist_norm, out=np.zeros_like(exist), where=exist_norm > 0)

    sims = cand_unit @ exist_unit.T   # (N_cand, N_exist)
    return sims.max(axis=1)


def semantic_dedup_extractions(
    candidates: list[Extraction],
    existing_texts: list[str],
    threshold: float = 0.86,
) -> list[Extraction]:
    """Return candidates whose max-cosine to any existing text is BELOW threshold.

    No-op when model2vec is unavailable, `existing_texts` is empty, or
    `candidates` is empty. Comparison is strict `>` so threshold==similarity
    keeps the candidate (boundary equality is conservative).
    """
    if not candidates or not existing_texts:
        return candidates
    if not is_available():
        return candidates

    cand_texts = [e.content for e in candidates]
    cand_vecs = embed_texts(cand_texts)
    exist_vecs = embed_texts(existing_texts)
    if cand_vecs is None or exist_vecs is None:
        return candidates

    max_sims = _cosine_max_row(cand_vecs, exist_vecs)
    return [ext for ext, sim in zip(candidates, max_sims) if sim <= threshold]


def intra_batch_dedup(
    candidates: list[Extraction],
    threshold: float = 0.86,
) -> list[Extraction]:
    """Within a single batch, drop later candidates whose cosine to any
    earlier-kept candidate exceeds threshold. Keeps the first occurrence.

    No-op when model2vec is unavailable or the batch has fewer than 2 items.
    """
    if len(candidates) < 2 or not is_available():
        return candidates

    texts = [e.content for e in candidates]
    vecs = embed_texts(texts)
    if vecs is None:
        return candidates

    import numpy as np

    kept_idx: list[int] = []
    norms = np.linalg.norm(vecs, axis=1)
    for i in range(len(candidates)):
        is_dup = False
        for j in kept_idx:
            denom = norms[i] * norms[j]
            if denom <= 0:
                continue
            sim = float(np.dot(vecs[i], vecs[j]) / denom)
            if sim > threshold:
                is_dup = True
                break
        if not is_dup:
            kept_idx.append(i)
    return [candidates[i] for i in kept_idx]
