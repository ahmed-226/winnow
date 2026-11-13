"""Unit tests for the RRF fusion math (plan §17.4, §12.5)."""

from __future__ import annotations

from app.services.search import rrf_score


def test_swap_symmetric_ranks_score_equally() -> None:
    # Plan §17.4: dense ranks [1, 2] and sparse ranks [2, 1] -> both chunks
    # score 1/61 + 1/62; neither retriever's order wins by itself.
    first = rrf_score(1, 2)
    second = rrf_score(2, 1)
    assert first == second
    assert first == 1 / 61 + 1 / 62


def test_top_of_both_lists_beats_single_list_winners() -> None:
    top_both = rrf_score(1, 1)
    assert top_both == 2 / 61
    assert top_both > rrf_score(1, None)
    assert top_both > rrf_score(1, 5)
    assert top_both > rrf_score(3, 1)


def test_missing_rank_contributes_nothing() -> None:
    assert rrf_score(None, 4) == 1 / 64
    assert rrf_score(4, None) == 1 / 64
    assert rrf_score(None, None) == 0.0


def test_weights_scale_contributions() -> None:
    # Weighted RRF formula (§12.5): w_d/(k+r_d) + w_s/(k+r_s).
    assert rrf_score(1, None, dense_weight=2.0) == 2 / 61
    assert rrf_score(1, 2, dense_weight=2.0, sparse_weight=0.5) == 2 / 61 + 0.5 / 62


def test_lower_k_sharpens_top_ranks() -> None:
    top = rrf_score(1, 1, k=1)
    mid = rrf_score(10, 10, k=1)
    # With k=1 the gap between rank 1 and rank 10 is much wider than with k=60.
    assert (top - mid) > (rrf_score(1, 1) - rrf_score(10, 10))
