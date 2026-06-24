"""Benchmark: vectorized multi-chain NUTS sampling for CODA models.

Compares wall-clock runtime of :meth:`Sccoda.run_nuts` (and optionally Tasccoda)
with a single chain versus several chains sampled on a single device via
``chain_method="vectorized"``. The total number of posterior draws is held
constant across configurations so the comparison is fair.

Run directly (not part of the test suite)::

    python benchmarks/coda_multichain_bench.py

The script prints the device/JAX info and a results table that can be pasted
into the PR description.
"""

from __future__ import annotations

import platform
import statistics
import time

import jax
import scanpy as sc

import pertpy as pt

# Total posterior draws kept constant across configs; per-chain draws = TOTAL_SAMPLES / num_chains.
TOTAL_SAMPLES = 4000
NUM_WARMUP = 500
REPEATS = 3  # timed runs per config (a warm-up run is always discarded first to exclude JIT compilation)
CHAIN_CONFIGS = [
    (1, "vectorized"),
    (2, "vectorized"),
    (4, "vectorized"),
    (4, "sequential"),  # isolates the vectorization gain from simply running more chains
]


def _load_sccoda():
    adata = pt.dt.haber_2017_regions()
    sccoda = pt.tl.Sccoda()
    mdata = sccoda.load(
        adata,
        type="cell_level",
        generate_sample_level=True,
        cell_type_identifier="cell_label",
        sample_identifier="batch",
        covariate_obs=["condition"],
    )
    mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
    return sccoda, mdata


def _time_run(model, mdata, *, num_chains: int, chain_method: str) -> float:
    per_chain = max(TOTAL_SAMPLES // num_chains, 1)
    # Work on a copy so each run starts from a clean prepared object.
    t0 = time.perf_counter()
    model.run_nuts(
        mdata,
        num_samples=per_chain,
        num_warmup=NUM_WARMUP,
        num_chains=num_chains,
        rng_key=0,
        copy=True,
        chain_method=chain_method,
    )
    return time.perf_counter() - t0


def main() -> None:
    """Run the benchmark matrix and print the environment info and results table."""
    print("=== Environment ===")
    print(f"platform: {platform.platform()}")
    print(f"jax: {jax.__version__}")
    print(f"jax devices: {jax.devices()} (count={jax.device_count()})")
    print(f"total_samples={TOTAL_SAMPLES}, num_warmup={NUM_WARMUP}, repeats={REPEATS}\n")

    sccoda, mdata = _load_sccoda()

    rows = []
    baseline = None
    for num_chains, chain_method in CHAIN_CONFIGS:
        # Discarded warm-up run to exclude one-time JIT compilation from the timing.
        _time_run(sccoda, mdata, num_chains=num_chains, chain_method=chain_method)
        times = [_time_run(sccoda, mdata, num_chains=num_chains, chain_method=chain_method) for _ in range(REPEATS)]
        median = statistics.median(times)
        if baseline is None:
            baseline = median
        rows.append((num_chains, chain_method, median, baseline / median))

    print("=== Results (scCODA NUTS) ===")
    print(f"{'chains':>6}  {'method':>10}  {'median_s':>9}  {'speedup_x':>9}")
    for num_chains, chain_method, median, speedup in rows:
        print(f"{num_chains:>6}  {chain_method:>10}  {median:>9.2f}  {speedup:>9.2f}")


if __name__ == "__main__":
    main()
