import numpy as np
import pytest

from mlsysim.labs.experiments.same_bits_training_experiments import (
    COLLECTIVES,
    WORLD_SIZES,
    allreduce,
    collective_contrast,
    collective_snapshot,
    rank_gradients,
    ring_order,
    ulp_distance,
)


def test_every_collective_computes_the_sum():
    grads = rank_gradients(8, 512)
    exact = grads.astype(np.float64).sum(axis=0)
    for name in COLLECTIVES:
        out = allreduce(grads, name)
        assert out.dtype == np.float32
        assert np.allclose(out, exact, rtol=1e-5, atol=1e-6), name
    with pytest.raises(KeyError):
        allreduce(grads, "butterfly")


def test_ring_order_visits_every_rank_once_and_ends_on_the_chunk_owner():
    for ranks in WORLD_SIZES:
        for chunk in range(ranks):
            order = ring_order(chunk, ranks)
            assert sorted(order) == list(range(ranks))
            assert order[-1] == chunk


def test_collectives_agree_when_addition_is_exact():
    # Small integers are exact in float32, so every order gives the same bits.
    grads = np.random.default_rng(0).integers(-1000, 1000, (8, 256)).astype(np.float32)
    results = [allreduce(grads, name) for name in COLLECTIVES]
    assert all(np.array_equal(results[0], r) for r in results[1:])


def test_each_collective_repeats_but_the_three_disagree():
    data = collective_contrast()
    assert data["repeatable"]
    assert data["distinct_results"] == 3
    assert 0 < data["params_where_collectives_disagree"] < data["params"]
    star = next(r for r in data["rows"] if r["collective"] == "star")
    assert star["params_differing_from_star"] == 0 and star["max_ulps_from_star"] == 0
    for row in data["rows"]:
        assert row["max_error_vs_exact"] < 1e-5


def test_two_ranks_have_only_one_order():
    # With two contributions every schedule computes a + b (float addition is commutative).
    data = collective_contrast(ranks=2)
    assert data["distinct_results"] == 1
    assert data["params_where_collectives_disagree"] == 0


def test_more_ranks_means_more_disagreement():
    counts = [collective_contrast(ranks=r)["params_where_collectives_disagree"] for r in (4, 8, 16)]
    assert counts == sorted(counts) and counts[0] < counts[-1]


def test_snapshot_matches_contrast_and_rejects_unknown_sizes():
    snap = collective_snapshot("ring")
    assert snap["inputs"]["collective"] == "ring"
    assert snap["distinct_over_all_collectives"] == collective_contrast()["distinct_results"]
    with pytest.raises(KeyError):
        collective_contrast(ranks=3)


def test_ulp_distance():
    one = np.array([1.0], np.float32)
    assert ulp_distance(one, one) == 0
    assert ulp_distance(one, np.nextafter(one, np.float32(2))) == 1
