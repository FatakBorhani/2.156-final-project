"""Simulate designs in parallel, one thread each, with a time limit per design.

Two fixes over scripts/evaluate.py, with the same worker function (evaluate._evaluate_one)
and ledger format, so everything downstream reads the results unchanged:

1. One thread per simulation (OMP_NUM_THREADS=1 etc.). By default the Boozer transform
   (OpenMP) starts a thread per core in every worker; with 8 workers that is ~80 threads
   spinning on 10 cores. In the surface batch this made 15% of simulations take 5-40
   minutes and cut throughput from ~370 to ~130 per hour. The same slow designs re-run
   with one thread took 58-76 s, with identical results.
2. A time limit: a simulation past `limit` seconds is killed and recorded as failed
   (ok false, timeout true); its time still counts toward the budget. Normal runs take
   ~60-100 s, so this only catches a genuinely stuck one.

Workers are long-lived; only a killed worker is replaced.

    python scripts/seeded/run_sims.py designs.jsonl data/seeded/runs/x.jsonl --workers 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import queue
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import _evaluate_one, default_workers, read_jsonl  # noqa: E402

LIMIT = 450.0
ONE_THREAD = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                               "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")}


def _worker(wid: int, inbox, out, fidelity: str) -> None:
    from constellaration import forward_model, problems  # noqa: F401  load before the clock starts
    while (design := inbox.get()) is not None:
        out.put(("start", wid, design["id"], time.time()))
        out.put(("done", wid, design["id"], _evaluate_one(design, fidelity)))


def run(designs: list[dict], ledger: str | Path, workers: int | None = None,
        limit: float = LIMIT, fidelity: str = "high_fidelity") -> list[dict]:
    """Like evaluate.evaluate_designs, plus the time limit. Returns all ledger rows for `designs`."""
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["id"], r["fidelity"]) for r in read_jsonl(ledger)}
    todo = [d for d in designs if (d["id"], fidelity) not in done][::-1]
    workers = max(1, min(workers or default_workers(), len(todo)))
    total = len(todo)
    print(f"{len(designs) - total} already in {ledger}; evaluating {total} with {workers} "
          f"workers at {fidelity}, {limit:.0f} s limit", flush=True)
    want = {d["id"] for d in designs}
    if not todo:
        return [r for r in read_jsonl(ledger) if r["id"] in want and r["fidelity"] == fidelity]

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    pool: dict[int, dict] = {}
    ids = itertools.count()
    count = [0]

    def spawn() -> None:
        wid, inbox = next(ids), ctx.Queue()
        p = ctx.Process(target=_worker, args=(wid, inbox, out, fidelity), daemon=True)
        saved = {k: os.environ.get(k) for k in ONE_THREAD}
        os.environ.update(ONE_THREAD)        # the child copies the environment at start
        try:
            p.start()
        finally:
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
        pool[wid] = {"proc": p, "inbox": inbox, "design": None, "t0": None}

    t_start = time.perf_counter()
    with ledger.open("a") as f:
        def record(row: dict) -> None:
            count[0] += 1
            f.write(json.dumps(row) + "\n")
            f.flush()
            status = ("FEASIBLE" if row["feasible"] else
                      f"max violation {row['max_violation']:.3f}" if row["ok"] else
                      "TIMEOUT" if row.get("timeout") else "FAILED")
            print(f"[{count[0]}/{total}] {row['id']}: {status} ({row['seconds']:.0f} s)", flush=True)

        for _ in range(workers):
            spawn()
        while todo or any(w["design"] for w in pool.values()):
            for w in pool.values():
                if w["design"] is None and todo:
                    w["design"], w["t0"] = todo.pop(), None
                    w["inbox"].put(w["design"])
            msgs = []
            try:
                msgs.append(out.get(timeout=1.0))
                while True:
                    msgs.append(out.get_nowait())
            except queue.Empty:
                pass
            for kind, wid, did, val in msgs:
                w = pool.get(wid)
                if w is None or w["design"] is None or w["design"]["id"] != did:
                    continue                                   # stale message from a killed worker
                if kind == "start":
                    w["t0"] = val
                else:
                    record(val)
                    w["design"] = None
            now = time.time()
            for wid, w in list(pool.items()):
                if w["design"] is None:
                    continue
                over = w["t0"] is not None and now - w["t0"] > limit
                if over or not w["proc"].is_alive():
                    w["proc"].kill()
                    w["proc"].join()
                    record({"id": w["design"]["id"], "fidelity": fidelity, "ok": False,
                            "feasible": False, "timeout": over,
                            "error": f"timeout after {limit:.0f} s" if over else "worker died",
                            "seconds": round(now - w["t0"], 2) if w["t0"] else 0.0})
                    del pool[wid]
                    if todo:
                        spawn()
        for w in pool.values():
            w["inbox"].put(None)
        for w in pool.values():
            w["proc"].join(timeout=10)
            if w["proc"].is_alive():
                w["proc"].kill()
    print(f"wall time {time.perf_counter() - t_start:.0f} s", flush=True)
    return [r for r in read_jsonl(ledger) if r["id"] in want and r["fidelity"] == fidelity]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("designs", type=Path)
    ap.add_argument("ledger", type=Path)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=float, default=LIMIT)
    a = ap.parse_args()
    run(read_jsonl(a.designs), a.ledger, a.workers, a.limit)


if __name__ == "__main__":
    main()
