"""Test 9: finish the job with multi-fidelity polishing.

Test 8's polishing ALM got the best design to 0.033 at high fidelity, but near the line
very low fidelity is off by ~0.02, as much as what's left. Here each outer iteration:

1. searches with INNER very-low-fidelity runs (cheap), scoring each candidate with its
   very-low-fidelity violations PLUS the measured (high - very low) offset at the current
   design, so the cheap search steers by high-fidelity numbers;
2. re-checks the TOP_HI best candidates at high fidelity (in parallel);
3. moves only if one of them is better at high fidelity (by augmented Lagrangian value),
   and re-measures the offset there; otherwise shrinks the box.

A start stops as soon as any high-fidelity check passes Benchmark 3.
Everything else as Test 8 (normalized constraints, small margin, L_gradB as tie-breaker).
Starts: Test 8's 3 best designs (different seeds), each with a reset box.

    python scripts/feasibility_tests/multifid_test.py            # ~7-8 h
    python scripts/feasibility_tests/multifid_test.py --dry      # plumbing check
    python scripts/feasibility_tests/multifid_test.py --report   # re-plot only
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent import futures
from pathlib import Path

import numpy as np

# registers SurfaceRZFourier as a JAX pytree; workers need it to unpickle the unravel fn
import constellaration.optimization.augmented_lagrangian_runner  # noqa: F401, E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import ROOT  # noqa: E402
from polish_test import evaluate  # noqa: E402  (raw metrics + violations for one design)

N_STARTS = 3
AR_BOUND = 12.0
MARGIN = 0.005  # small: the offset already corrects for the fidelity gap
W_OBJ = 0.05
P0 = 100.0
B0, B_MIN, SHRINK = 0.02, 0.003, 0.7
INNER, MAXIT, TOP_HI = 300, 30, 3
STOP_ON_FEASIBLE = True  # Test 10 sets False to keep improving L_gradB while feasible

OUT = ROOT / "results" / "feasibility_tests"
DATA = ROOT / "data" / "feasibility_tests"


def starts() -> list[tuple[str, str, float]]:
    """Test 8's best high-fidelity design for each of its N_STARTS best start designs."""
    rows = [json.loads(line) for line in open(OUT / "polish_checked_designs.jsonl")]
    best = {}
    for r in rows:
        if r["ok"] and (r["start"] not in best or r["worst_hi"] < best[r["start"]]["worst_hi"]):
            best[r["start"]] = r
    top = sorted(best.values(), key=lambda r: r["worst_hi"])[:N_STARTS]
    return [(r["start"], r["boundary"], r["worst_hi"]) for r in top]


def cons_obj(row: dict, offset: np.ndarray) -> tuple[float, list[float]]:
    """Objective and constraints; offset (6,) is added to [AR, 5 violations]."""
    if not row["ok"]:
        return W_OBJ, [1.0] * 6
    raw = np.array([row["aspect_ratio"], *row["violations"]]) + offset
    return (W_OBJ * (20.0 - row["L_gradB"]) / 20.0,
            [(raw[0] - AR_BOUND) / AR_BOUND, *(raw[1:] + MARGIN).tolist()])


def raw_vec(row: dict) -> np.ndarray:
    return np.array([row["aspect_ratio"], *row["violations"]])


def run_start(name: str, bjson: str, dry: bool, ledger: Path) -> None:
    import jax.numpy as jnp
    import nevergrad

    import constellaration.geometry.surface_rz_fourier as rz
    import constellaration.optimization.augmented_lagrangian as al
    from constellaration import forward_model
    from constellaration.mhd import vmec_settings
    from constellaration.utils import pytree

    from evaluate import default_workers

    workers = default_workers()
    inner, maxit = (24, 2) if dry else (INNER, MAXIT)
    vlo = forward_model.ConstellarationSettings(
        vmec_preset_settings=vmec_settings.VmecPresetSettings(fidelity="very_low_fidelity"))
    hi = forward_model.ConstellarationSettings.default_high_fidelity()
    alset = al.AugmentedLagrangianSettings(constraint_violation_tolerance_reduction_factor=0.8,
                                           penalty_parameters_increase_factor=2.0,
                                           penalty_parameters_max=1e6, bounds_min=B_MIN)
    boundary = rz.set_max_mode_numbers(rz.SurfaceRZFourier.model_validate_json(bjson),
                                       max_poloidal_mode=4, max_toroidal_mode=4)
    mask = rz.build_mask(boundary, max_poloidal_mode=4, max_toroidal_mode=4)
    x0, unravel = pytree.mask_and_ravel(pytree=boundary, mask=mask)
    scale = rz.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes=boundary.poloidal_modes.flatten(),
        toroidal_modes=boundary.toroidal_modes.flatten(), alpha=1.5,
    ).reshape(boundary.poloidal_modes.shape)
    scale = jnp.array(np.concatenate([scale[mask.r_cos], scale[mask.z_sin]]))
    x = np.asarray(jnp.array(x0) / scale)

    def bjson_of(xx):
        return unravel(jnp.asarray(xx) * scale).model_dump_json()

    ctx = multiprocessing.get_context("forkserver")
    n, n_hi, t_start = 0, 0, time.time()
    with ledger.open("a") as f, futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        def log(row, k, kind):
            row.update(start=name, k=k, n=n, n_hi=n_hi, kind=kind, wall=round(time.time() - t_start, 1))
            f.write(json.dumps(row) + "\n")
            f.flush()

        def check_hi(xs, k):
            """High-fidelity rows for candidate points, logged with boundaries."""
            nonlocal n_hi
            rows = [fu.result() for fu in [ex.submit(evaluate, xx, scale, unravel, hi) for xx in xs]]
            for xx, r in zip(xs, rows):
                n_hi += 1
                log({**{k2: v for k2, v in r.items() if k2 != "boundary"}, "boundary": bjson_of(xx)}, k, "hi")
            return rows

        cur_vlo = ex.submit(evaluate, x, scale, unravel, vlo).result()
        cur_hi = check_hi([x], 0)[0]
        log({k2: v for k2, v in cur_vlo.items() if k2 != "boundary"}, 0, "start_vlo")
        offset = raw_vec(cur_hi) - raw_vec(cur_vlo)
        obj, cons = cons_obj(cur_hi, np.zeros(6))
        state = al.AugmentedLagrangianState(
            x=jnp.array(x), multipliers=jnp.zeros(6), penalty_parameters=jnp.full(6, P0),
            objective=jnp.array(obj), constraints=jnp.array(cons), bounds=jnp.full(len(x), B0))
        bounds, rng = B0, np.random.RandomState(123)
        print(f"[{name}] start: high {cur_hi['worst']:.4f}, very low {cur_vlo['worst']:.4f}", flush=True)
        if cur_hi["feasible"] and STOP_ON_FEASIBLE:
            print(f"[{name}] start is already feasible", flush=True)
            return

        for k in range(1, maxit + 1):
            def lag(o, c):
                return float(al.augmented_lagrangian_function(jnp.array(o), jnp.array(c), state))

            par = nevergrad.p.Array(init=x, lower=x - bounds, upper=x + bounds)
            par.random_state = np.random.RandomState(rng.randint(2**31))
            oracle = nevergrad.optimizers.NGOpt(parametrization=par, budget=inner, num_workers=workers)
            cands = []  # (corrected AL, x, vlo row)
            rest, running = inner, []
            while rest or running:
                while len(running) < min(workers, rest):
                    cand = oracle.ask()
                    running.append((ex.submit(evaluate, np.array(cand.value), scale, unravel, vlo), cand))
                    rest -= 1
                done, _ = futures.wait([fu for fu, _ in running], return_when=futures.FIRST_COMPLETED)
                for fu, cand in running:
                    if fu in done:
                        n += 1
                        row = fu.result()
                        val = lag(*cons_obj(row, offset))
                        oracle.tell(cand, val)
                        cands.append((val, np.array(cand.value), row))
                        log({k2: v for k2, v in row.items() if k2 != "boundary"}, k, "oracle")
                running = [(fu, c) for fu, c in running if fu not in done]

            cands.sort(key=lambda t: t[0])
            picks, seen = [], set()
            for val, xx, row in cands:
                key = round(row.get("worst", 9), 4)
                if row["ok"] and key not in seen:
                    seen.add(key)
                    picks.append((xx, row))
                if len(picks) == TOP_HI:
                    break
            his = check_hi([xx for xx, _ in picks], k)
            cur_val = lag(*cons_obj(cur_hi, np.zeros(6)))
            scored = [(lag(*cons_obj(h, np.zeros(6))), xx, v, h)
                      for (xx, v), h in zip(picks, his) if h["ok"]]
            feasible = [s for s in scored if s[3]["feasible"]]
            best = min(feasible or scored, key=lambda s: s[0], default=None)
            # never trade a feasible current design for a worse-scoring one
            moved = best is not None and (best[0] < cur_val
                                          or (best[3]["feasible"] and not cur_hi["feasible"]))
            if moved:
                _, x, v, cur_hi = best
                offset = raw_vec(cur_hi) - raw_vec(v)
            else:
                bounds = max(B_MIN, bounds * SHRINK)
            obj, cons = cons_obj(cur_hi, np.zeros(6))
            state = al.update_augmented_lagrangian_state(
                x=jnp.array(x), objective=jnp.array(obj), constraints=jnp.array(cons),
                state=state, settings=alset)
            log({"worst_hi": cur_hi["worst"], "L_gradB_hi": cur_hi["L_gradB"],
                 "aspect_ratio_hi": cur_hi["aspect_ratio"], "violations_hi": cur_hi["violations"],
                 "checked_hi": [h.get("worst") for h in his], "moved": moved, "bounds": bounds,
                 "offset": offset.tolist(), "penalties": np.asarray(state.penalty_parameters).tolist(),
                 "multipliers": np.asarray(state.multipliers).tolist()}, k, "iteration")
            print(f"[{name}] it {k}: runs {n} + {n_hi} high, checked "
                  f"{[round(h['worst'], 4) if h['ok'] else None for h in his]}, current high "
                  f"{cur_hi['worst']:.4f} (L_gradB {cur_hi['L_gradB']:.2f}), box {bounds:.4f}, "
                  f"{(time.time() - t_start) / 60:.0f} min", flush=True)
            if feasible:
                print(f"[{name}] FEASIBLE at high fidelity in iteration {k} "
                      f"({len(feasible)} of {len(his)} checked)", flush=True)
                if STOP_ON_FEASIBLE:
                    return


def read(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def report(ledger: Path, dry: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = "_dry" if dry else ""
    rows = read(ledger)
    names = list(dict.fromkeys(r["start"] for r in rows))
    summary = {"settings": {"margin": MARGIN, "w_obj": W_OBJ, "p0": P0, "box": [B0, B_MIN, SHRINK],
                            "inner": INNER, "maxit": MAXIT, "top_hi": TOP_HI}, "starts": {}}
    fig, ax = plt.subplots(figsize=(7, 4))
    with open(OUT / f"multifid_checked_designs{tag}.jsonl", "w") as f:
        for s in names:
            rs = [r for r in rows if r["start"] == s]
            his = [r for r in rs if r["kind"] == "hi"]
            its = [r for r in rs if r["kind"] == "iteration"]
            ok = [h for h in his if h["ok"]]
            feas = [h for h in his if h.get("feasible")]
            best = min(ok, key=lambda h: h["worst"])
            ax.plot([0] + [r["k"] for r in its], [his[0]["worst"]] + [r["worst_hi"] for r in its],
                    marker=".", label=s)
            for h in his:
                f.write(json.dumps({"start": s, "k": h["k"], "n_hi": h["n_hi"], "ok": h["ok"],
                                    "feasible_hi": h.get("feasible"), "worst_hi": h.get("worst"),
                                    "L_gradB": h.get("L_gradB"), "aspect_ratio": h.get("aspect_ratio"),
                                    "violations": h.get("violations"), "boundary": h["boundary"]}) + "\n")
            summary["starts"][s] = {
                "start_worst_hi": his[0]["worst"], "iterations": len(its),
                "very_low_runs": sum(r["kind"] == "oracle" for r in rs), "high_runs": len(his),
                "best_worst_hi": best["worst"], "best_L_gradB": best["L_gradB"],
                "best_aspect_ratio": best["aspect_ratio"],
                "best_violations": best["violations"],
                "feasible_hi": [{"k": h["k"], "worst": h["worst"], "L_gradB": h["L_gradB"],
                                 "aspect_ratio": h["aspect_ratio"]} for h in feas],
                "hours": round(rs[-1]["wall"] / 3600, 2),
            }
    ax.axhline(0.01, color="k", ls=":", lw=1)
    ax.text(0, 0.0105, "feasible (≤ 0.01)", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("outer iteration (300 very-low + 3 high-fidelity runs each)")
    ax.set_ylabel("current design, worst violation (high fidelity)")
    ax.set_title("Multi-fidelity polishing, by start design")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"multifid{tag}.png", dpi=150)
    summary["feasible_designs_high_fidelity"] = sum(len(v["feasible_hi"]) for v in summary["starts"].values())
    (OUT / f"multifid{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    ledger = DATA / f"multifid_ledger{'_dry' if a.dry else ''}.jsonl"
    if not a.report:
        done = {r["start"] for r in read(ledger)}
        todo = starts()[:1] if a.dry else starts()
        print("starts:", [(s, round(w, 4)) for s, _, w in todo], flush=True)
        rows = read(ledger)
        for name, bjson, _ in todo:
            if name in done:
                # interrupted run: resume from its best high-fidelity design so far
                runs = [r for r in rows if r["start"].split("_r")[0] == name]
                its = sum(r["kind"] == "iteration" for r in runs)
                feas = any(r["kind"] == "hi" and r.get("feasible") for r in runs)
                if feas or its >= (2 if a.dry else MAXIT):
                    print(f"[{name}] already done; skipping")
                    continue
                best = min((r for r in runs if r["kind"] == "hi" and r["ok"]), key=lambda r: r["worst"])
                name2 = f"{name}_r{sum(1 for s in done if s.startswith(name + '_r')) + 1}"
                if name2 in done:
                    continue
                print(f"[{name}] interrupted after {its} iterations; resuming as {name2} "
                      f"from its best high-fidelity design ({best['worst']:.4f})", flush=True)
                name, bjson = name2, best["boundary"]
            run_start(name, bjson, a.dry, ledger)
    report(ledger, a.dry)


if __name__ == "__main__":
    main()
