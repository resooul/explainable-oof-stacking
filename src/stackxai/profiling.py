"""Inference-cost profiling, executed in a fresh process so memory figures are not contaminated by training."""
from __future__ import annotations

import io
import json
import os
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import psutil


def rss_mb() -> float:
    """Resident set size of the current process in MB."""
    return psutil.Process(os.getpid()).memory_info().rss / 2 ** 20


class PeakRSS:
    """Context manager that samples the resident set size every ``interval`` seconds and records its peak."""
    def __init__(self, interval=0.01):
        self.interval, self.peak, self._stop = interval, 0.0, threading.Event()

    def _loop(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, rss_mb())
            time.sleep(self.interval)

    def __enter__(self):
        self.peak = rss_mb()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, rss_mb())


def latency(fn, rows: list, warmup: int) -> dict:
    """Latency statistics of ``fn`` over ``rows`` in milliseconds, after ``warmup`` untimed calls."""
    for r in rows[:warmup]:
        fn(r)
    t = []
    for r in rows:
        t0 = time.perf_counter()
        fn(r)
        t.append((time.perf_counter() - t0) * 1e3)
    t = np.asarray(t)
    return {"mean_ms": float(t.mean()), "p50_ms": float(np.percentile(t, 50)), "p95_ms": float(np.percentile(t, 95)),
            "p99_ms": float(np.percentile(t, 99)), "max_ms": float(t.max())}


def serialized_mb(obj) -> float:
    """Size of ``obj`` serialised with joblib, in MB."""
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.tell() / 2 ** 20


def system_info() -> dict:
    """Platform, processor, Python version, core counts and installed memory of the machine."""
    vm = psutil.virtual_memory()
    return {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(), "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True), "ram_gb": round(vm.total / 2 ** 30, 1)}


# State of a profiling worker process: its own pipeline and sample rows.
_WORKER: dict = {}


def _init_worker(run_dir: str):
    """Process-pool initializer: every worker loads its own pipeline and sample, single-threaded, and warms up."""
    pipe = joblib.load(Path(run_dir) / "models" / "pipeline.joblib")
    pipe.set_inference_threads(1)
    X = pd.read_parquet(Path(run_dir) / "models" / "profiling_sample.parquet")
    for i in range(min(10, len(X))):
        pipe.predict(X.iloc[[i]])
    _WORKER.update(pipe=pipe, X=X)


def _ready(_):
    """Short task used to confirm that every worker process has started before timing begins."""
    time.sleep(0.2)
    return os.getpid()


def _predict_row(i: int) -> int:
    """Predict sample row ``i`` as a single-record request in a worker process."""
    _WORKER["pipe"].predict(_WORKER["X"].iloc[[i]])
    return i


def process_throughput(run_dir: Path, n_rows: int, workers: int) -> float:
    """Single-record requests/s with ``workers`` processes; model loading and warm-up are excluded."""
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=workers, initializer=_init_worker, initargs=(str(run_dir),)) as pool:
        seen = set()
        for _ in range(20):
            seen |= set(pool.map(_ready, range(workers), chunksize=1))
            if len(seen) >= workers:
                break
        t0 = time.perf_counter()
        pool.map(_predict_row, range(n_rows), chunksize=1)
        return n_rows / (time.perf_counter() - t0)


def profile(run_dir: str | Path, cfg: dict) -> dict:
    """Profile the deployed pipeline of a finished run and write the results to ``tables/profiling*``.

    Meant to run in a fresh process (``python -m stackxai profile``). Measures single-record latency per stage and
    end to end with one thread, the latency of every combiner, single-record throughput with threads sharing one
    pipeline and with processes that each load their own copy, batch throughput, the memory held by the loaded
    pipeline and the peak memory during batch prediction.
    """
    run_dir = Path(run_dir)
    p = cfg["profiling"]
    rss0 = rss_mb()
    pipe = joblib.load(run_dir / "models" / "pipeline.joblib")
    X = pd.read_parquet(run_dir / "models" / "profiling_sample.parquet")
    rss_loaded = rss_mb()
    n = min(p["latency_rows"], len(X))
    rows = [X.iloc[[i]] for i in range(n)]
    base_rows = [pipe.base_matrix(r) for r in rows[: min(n, 200)]]

    pipe.set_inference_threads(1)
    stages = []
    shared = hasattr(pipe, "shares_encoding") and pipe.shares_encoding()
    if shared:
        # the deployed path encodes a request once and passes it to every base learner
        encoded = [pipe.encode(r) for r in rows]
        stages.append({"stage": "encode", "size_mb": 0.0, **latency(pipe.encode, rows, p["warmup"])})
        for b in pipe.bases:
            stages.append({"stage": b.name, "size_mb": serialized_mb(b),
                           **latency(b.predict_encoded, encoded, p["warmup"])})
    else:
        for b in pipe.bases:
            stages.append({"stage": b.name, "size_mb": serialized_mb(b), **latency(b.predict, rows, p["warmup"])})
    n_pre = len(stages)
    stages.append({"stage": f"meta:{pipe.name}", "size_mb": serialized_mb(pipe.meta),
                   **latency(pipe.meta.predict, base_rows, min(p["warmup"], len(base_rows)))})
    stages.append({"stage": "end_to_end", "size_mb": serialized_mb(pipe), **latency(pipe.predict, rows, p["warmup"])})

    meta_rows = []
    metas_path = run_dir / "models" / "meta_learners.joblib"
    if metas_path.exists():
        base_sum = sum(st["mean_ms"] for st in stages[:n_pre])
        for name, meta in joblib.load(metas_path).items():
            if hasattr(meta, "set_inference_threads"):
                meta.set_inference_threads(1)
            lat = latency(meta.predict, base_rows, min(p["warmup"], len(base_rows)))
            meta_rows.append({"combiner": name, "deployed": name == pipe.name, "size_mb": serialized_mb(meta), **lat,
                              "implied_end_to_end_mean_ms": base_sum + lat["mean_ms"]})

    throughput = []
    for w in p["workers"]:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=w) as ex:
            list(ex.map(pipe.predict, rows))
        throughput.append({"mode": "single-record requests, threads", "workers": w,
                           "requests_per_s": n / (time.perf_counter() - t0)})
    for w in p.get("process_workers") or []:
        throughput.append({"mode": "single-record requests, processes", "workers": w,
                           "requests_per_s": process_throughput(run_dir, n, w)})

    pipe.set_inference_threads(cfg["run"]["n_jobs"])
    batch = X.iloc[: min(p["batch_rows"], len(X))]
    pipe.predict(batch.iloc[:1000])
    with PeakRSS() as mon:
        t0 = time.perf_counter()
        pipe.predict(batch)
        elapsed = time.perf_counter() - t0
    throughput.append({"mode": f"batch ({len(batch):,} rows)", "workers": cfg["run"]["n_jobs"],
                       "requests_per_s": len(batch) / elapsed})
    memory = {"rss_before_load_mb": rss0, "rss_after_load_mb": rss_loaded,
              "model_resident_mb": rss_loaded - rss0, "peak_rss_batch_mb": mon.peak}
    out = {"system": system_info(), "stages": stages, "meta": meta_rows, "throughput": throughput, "memory": memory}
    with open(run_dir / "tables" / "profiling.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    pd.DataFrame(stages).to_csv(run_dir / "tables" / "profiling_latency.csv", index=False)
    pd.DataFrame(throughput).to_csv(run_dir / "tables" / "profiling_throughput.csv", index=False)
    pd.DataFrame([memory]).to_csv(run_dir / "tables" / "profiling_memory.csv", index=False)
    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(run_dir / "tables" / "profiling_meta.csv", index=False)
    return out
