"""
tests/test_embeddings.py -- Semantic-dedup embeddings module unit tests.

Covers `is_available()` flip, no-op fallback when the dep is missing, stubbed-
vector dedup, intra-batch dedup keep-first, and threshold strictness (`>`
not `>=`).

Direct unit tests use a stubbed embed function so the real model isn't loaded
during the suite (which would be slow on a cold cache).
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from src import embeddings
from src.state_store import Extraction


def _ext(content: str, ext_id: str = "x") -> Extraction:
    return Extraction(
        id=ext_id, source_id="s1", source_type="pact_memory",
        extraction_type="lesson", topic="t",
        title="t", content=content,
    )


class TestIsAvailable(unittest.TestCase):
    def test_returns_false_when_model2vec_missing(self):
        embeddings.is_available.cache_clear()
        with mock.patch.dict("sys.modules", {"model2vec": None}):
            self.assertFalse(embeddings.is_available())
        embeddings.is_available.cache_clear()


class TestSemanticDedupFallback(unittest.TestCase):
    """When dep is missing, helpers return inputs unchanged."""

    def setUp(self):
        embeddings.is_available.cache_clear()
        self.patcher = mock.patch.object(embeddings, "is_available", return_value=False)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()
        embeddings.is_available.cache_clear()

    def test_semantic_dedup_no_op(self):
        cands = [_ext("a"), _ext("b")]
        result = embeddings.semantic_dedup_extractions(cands, ["x"], threshold=0.8)
        self.assertEqual(result, cands)

    def test_intra_batch_no_op(self):
        cands = [_ext("a"), _ext("b")]
        result = embeddings.intra_batch_dedup(cands, threshold=0.8)
        self.assertEqual(result, cands)


class TestSemanticDedupWithStubbedVectors(unittest.TestCase):
    """Patch embed_texts to return controlled vectors; assert filtering logic."""

    def setUp(self):
        self.available_patcher = mock.patch.object(embeddings, "is_available", return_value=True)
        self.available_patcher.start()

    def tearDown(self):
        self.available_patcher.stop()

    def _mocked_embed(self, mapping: dict[str, list[float]]):
        """Return a function that maps text -> fixture vector."""
        def fake(texts: list[str]) -> np.ndarray:
            vecs = [mapping[t] for t in texts]
            return np.array(vecs, dtype="float32")
        return fake

    def test_candidate_above_threshold_is_dropped(self):
        # cosine([1,0], [1,0.05]) ≈ 0.998 > 0.86 → drop
        mapping = {"new": [1.0, 0.0], "old": [1.0, 0.05]}
        with mock.patch.object(embeddings, "embed_texts",
                               side_effect=self._mocked_embed(mapping)):
            cands = [_ext("new")]
            result = embeddings.semantic_dedup_extractions(cands, ["old"], threshold=0.86)
        self.assertEqual(result, [])

    def test_candidate_below_threshold_kept(self):
        # cosine([1,0], [0,1]) = 0 < 0.86 → keep
        mapping = {"new": [1.0, 0.0], "old": [0.0, 1.0]}
        with mock.patch.object(embeddings, "embed_texts",
                               side_effect=self._mocked_embed(mapping)):
            cands = [_ext("new")]
            result = embeddings.semantic_dedup_extractions(cands, ["old"], threshold=0.86)
        self.assertEqual(len(result), 1)

    def test_threshold_strict_inequality(self):
        # cosine equal to threshold should KEEP the candidate (`<=` keeps it).
        # Build vectors with cosine exactly 0.86.
        import math
        angle = math.acos(0.86)
        v = [math.cos(angle), math.sin(angle)]
        mapping = {"new": v, "old": [1.0, 0.0]}
        with mock.patch.object(embeddings, "embed_texts",
                               side_effect=self._mocked_embed(mapping)):
            cands = [_ext("new")]
            result = embeddings.semantic_dedup_extractions(cands, ["old"], threshold=0.86)
        self.assertEqual(len(result), 1)

    def test_empty_existing_texts_passes_through(self):
        cands = [_ext("a")]
        result = embeddings.semantic_dedup_extractions(cands, [], threshold=0.8)
        self.assertEqual(result, cands)

    def test_empty_candidates_passes_through(self):
        result = embeddings.semantic_dedup_extractions([], ["x"], threshold=0.8)
        self.assertEqual(result, [])


class TestIntraBatchDedup(unittest.TestCase):
    """Intra-batch keeps the first occurrence."""

    def setUp(self):
        self.available_patcher = mock.patch.object(embeddings, "is_available", return_value=True)
        self.available_patcher.start()

    def tearDown(self):
        self.available_patcher.stop()

    def test_pair_above_threshold_keeps_first(self):
        mapping = {"x": [1.0, 0.0], "y": [1.0, 0.05]}  # cos~0.998
        def fake(texts):
            return np.array([mapping[t] for t in texts], dtype="float32")
        with mock.patch.object(embeddings, "embed_texts", side_effect=fake):
            cands = [_ext("x", "1"), _ext("y", "2")]
            result = embeddings.intra_batch_dedup(cands, threshold=0.86)
        self.assertEqual([c.id for c in result], ["1"])

    def test_short_batch_passes_through(self):
        cands = [_ext("a")]
        result = embeddings.intra_batch_dedup(cands, threshold=0.8)
        self.assertEqual(result, cands)


if __name__ == "__main__":
    unittest.main()
