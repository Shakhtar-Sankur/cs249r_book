"""Same-Bits experiments, Part 2: why the same data-parallel job gives different weights.

Every number here is computed, not modelled.  numpy's float32 is IEEE 754 binary32,
rounded after every addition, so the bit patterns below are the ones a processor
that adds in the same order produces.  Each collective writes its reduction order
out explicitly instead of calling a library whose order is not specified.

Part A (this prototype): one set of per-rank gradients, reduced by three AllReduce
schedules.  Each schedule is deterministic on its own (run it twice and the bits
repeat), but the schedules add the same contributions in different orders:

- ring: a reduce-scatter in which chunk c of the gradient starts at rank c + 1
  and travels once around the ring, so each chunk is summed in its own rotation;
- tree: recursive halving, a pairwise tree over the ranks;
- star: one root (or a switch) adds every rank's contribution in rank order.

The orders are teaching fixtures with the shapes of these schedules, not the
measured order of any named library.  Gradients are seeded synthetic values whose
magnitudes span three decades, as layers of a real network do.
"""

from __future__ import annotations

import hashlib

import numpy as np

MODEL_ID = "same_bits_training_experiments"

COLLECTIVES = ("ring", "tree", "star")
WORLD_SIZES = (2, 4, 8, 16)


# Bit-level helpers -------------------------------------------------------------------------

def digest(a: np.ndarray) -> str:
    """First 12 hex digits of the SHA-256 of an array's bytes: equal digests mean equal bits."""
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]


def ulp_distance(a, b) -> int:
    """Largest number of representable float32 values between matching elements (0 = same bits)."""
    ia = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    ib = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    top = np.int64(1) << 31
    ia = np.where(ia < 0, -top - ia, ia)
    ib = np.where(ib < 0, -top - ib, ib)
    return int(np.max(np.abs(ia - ib)))


# Gradients ---------------------------------------------------------------------------------

def rank_gradients(ranks: int = 8, params: int = 4096, seed: int = 0) -> np.ndarray:
    """float32 gradients of shape (ranks, params): what each rank computed on its own shard."""
    rng = np.random.default_rng(seed)
    scale = 10.0 ** rng.uniform(-3.0, 0.0, size=params)
    return (rng.standard_normal((ranks, params)) * scale).astype(np.float32)


# The three collectives ---------------------------------------------------------------------

def _add_in_order(grads: np.ndarray, order) -> np.ndarray:
    """Add rows of grads in the given rank order, rounding to float32 after every addition."""
    acc = grads[order[0]].copy()
    for r in order[1:]:
        acc = (acc + grads[r]).astype(np.float32)
    return acc


def ring_order(chunk: int, ranks: int) -> list[int]:
    """Rank order in which a ring reduce-scatter accumulates chunk `chunk`."""
    return [(chunk + 1 + k) % ranks for k in range(ranks)]


def allreduce_ring(grads: np.ndarray) -> np.ndarray:
    ranks = grads.shape[0]
    out = np.empty(grads.shape[1], np.float32)
    for c, idx in enumerate(np.array_split(np.arange(grads.shape[1]), ranks)):
        out[idx] = _add_in_order(grads[:, idx], ring_order(c, ranks))
    return out


def allreduce_tree(grads: np.ndarray) -> np.ndarray:
    level = [g for g in grads]
    while len(level) > 1:
        nxt = [(level[i] + level[i + 1]).astype(np.float32) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return level[0].astype(np.float32)


def allreduce_star(grads: np.ndarray) -> np.ndarray:
    return _add_in_order(grads, list(range(grads.shape[0])))


def allreduce(grads: np.ndarray, collective: str) -> np.ndarray:
    """Sum of the per-rank gradients, as every rank holds it after the AllReduce."""
    if collective == "ring":
        return allreduce_ring(grads)
    if collective == "tree":
        return allreduce_tree(grads)
    if collective == "star":
        return allreduce_star(grads)
    raise KeyError(f"unknown collective {collective!r}; expected one of {COLLECTIVES}")


# Part A: one gradient, three collectives ---------------------------------------------------

def collective_contrast(ranks: int = 8, params: int = 4096, seed: int = 0) -> dict:
    """The same per-rank gradients reduced by every collective in COLLECTIVES."""
    if ranks not in WORLD_SIZES:
        raise KeyError(f"unsupported world size {ranks}; expected one of {WORLD_SIZES}")
    grads = rank_gradients(ranks, params, seed)
    exact = grads.astype(np.float64).sum(axis=0)
    results = {name: allreduce(grads, name) for name in COLLECTIVES}
    stacked = np.stack([results[name] for name in COLLECTIVES]).view(np.uint32)
    disagree = np.any(stacked != stacked[0], axis=0)
    ref = results["star"]
    rows = [
        {
            "collective": name,
            "bits": digest(v),
            "params_differing_from_star": int(np.sum(v.view(np.uint32) != ref.view(np.uint32))),
            "max_ulps_from_star": ulp_distance(v, ref),
            "max_error_vs_exact": float(np.max(np.abs(v.astype(np.float64) - exact))),
        }
        for name, v in results.items()
    ]
    return {
        "inputs": {"ranks": ranks, "params": params, "seed": seed},
        "rows": rows,
        "distinct_results": len({row["bits"] for row in rows}),
        "params": params,
        "params_where_collectives_disagree": int(disagree.sum()),
        "repeatable": all(np.array_equal(allreduce(grads, n), results[n]) for n in COLLECTIVES),
    }


def collective_snapshot(collective: str, *, ranks: int = 8, params: int = 4096, seed: int = 0) -> dict:
    """One collective's row of the contrast, for the lab's evidence capture."""
    data = collective_contrast(ranks, params, seed)
    row = next(r for r in data["rows"] if r["collective"] == collective)
    return {
        "inputs": {"collective": collective, **data["inputs"]},
        "bits": row["bits"],
        "params_differing_from_star": row["params_differing_from_star"],
        "max_ulps_from_star": row["max_ulps_from_star"],
        "distinct_over_all_collectives": data["distinct_results"],
    }
