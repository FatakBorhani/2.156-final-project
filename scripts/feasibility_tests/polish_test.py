"""Test 8: ALM-NGOpt retuned to polish our best designs toward feasible.

Test 7 ran the paper's ALM-NGOpt as-is from our best design and it wandered away from
feasibility. Reading the package code shows why: its search box never shrinks (the
bounds_reduction_factor setting is unused, so the box stays at +-0.33), its constraints
are in raw units (a mirror-ratio shortfall weighs ~10x less than the same QI shortfall),
it ignores penalty_parameters_initial, and it always accepts Nevergrad's recommendation
even when that is worse than the current point. Changes here, all else as the paper:

- constraints are the benchmark's own normalized violations, plus MARGIN (aim 0.02 past
  each limit, to survive the very-low -> high fidelity check), plus aspect ratio <= 12
- objective (20 - L_gradB) / 20 scaled by W_OBJ: a tie-breaker until feasible
- small box (B0) that shrinks by SHRINK whenever an outer iteration doesn't improve
- each outer iteration keeps the best evaluated point (by augmented Lagrangian value)
- more outer iterations with a smaller inner budget, so multipliers adapt more often

Starts: Test 6's best design for each of its 5 seeds, plus Test 7's two best designs.
Afterwards the closest designs per start are re-checked at high fidelity.

    python scripts/feasibility_tests/polish_test.py            # overnight
    python scripts/feasibility_tests/polish_test.py --dry      # plumbing check
    python scripts/feasibility_tests/polish_test.py --report   # re-plot / re-check only
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

AR_BOUND = 12.0
MARGIN = 0.02
W_OBJ = 0.05
P0, P_GROW, P_MAX = 100.0, 2.0, 1e6
B0, B_MIN, SHRINK = 0.05, 0.005, 0.7
INNER, MAXIT = 300, 20  # 6,000 very-low-fidelity runs per start
HI_CHECK = 6  # closest distinct designs per start re-checked at high fidelity
KEEP_BOUNDARY_BELOW = 0.12

OUT = ROOT / "results" / "feasibility_tests"
DATA = ROOT / "data" / "feasibility_tests"


def starts() -> list[tuple[str, str, float]]:
    """(name, boundary json, high-fidelity worst) for each start design."""
    push = [json.loads(line) for line in open(OUT / "push_checked_designs.jsonl")]
    out = []
    for s in sorted({r["start"] for r in push}):
        best = min((r for r in push if r["start"] == s and r.get("ok")), key=lambda r: r["worst_hi"])
        out.append((f"t6_start{s}", best["boundary"], best["worst_hi"]))
    alm = [json.loads(line) for line in open(OUT / "alm_checked_designs.jsonl")]
    for arm in ("ours", "dataset"):
        best = min((r for r in alm if r["id"].startswith(f"alm_{arm}_") and r["ok"]
                    and not r["id"].startswith("alm_ours_n0") and not r["id"].startswith("alm_ours_n7")),
                   key=lambda r: r["worst_hi"])
        out.append((f"t7_{arm}", best["boundary"], best["worst_hi"]))
    return sorted(out, key=lambda t: t[2])


def evaluate(x, scale, unravel, fm_settings) -> dict:
    """Normalized constraints (+ margin) and scaled objective for one design."""
    import jax.numpy as jnp

    from constellaration import forward_model, problems

    t0 = time.perf_counter()
    problem = problems.MHDStableQIStellarator()
    boundary = unravel(jnp.asarray(x) * scale)
    row = {"ok": False}
    try:
        m, _ = forward_model.forward_model(boundary=boundary, settings=fm_settings)
        v = np.asarray(problem._normalized_constraint_violations(m), float)
        row = {"ok": True, "worst": float(np.maximum(v, 0).max()), "feasible": bool(problem.is_feasible(m)),
               "violations": v.tolist(), "L_gradB": float(m.minimum_normalized_magnetic_gradient_scale_length),
               "aspect_ratio": float(m.aspect_ratio)}
        row["objective"] = W_OBJ * (20.0 - row["L_gradB"]) / 20.0
        row["constraints"] = [(row["aspect_ratio"] - AR_BOUND) / AR_BOUND, *(v + MARGIN).tolist()]
        if row["worst"] < KEEP_BOUNDARY_BELOW:
            row["boundary"] = boundary.model_dump_json()
    except Exception as e:  # VMEC++ failing on a bad shape is expected
        row["error"] = f"{type(e).__name__}: {e}"[:120]
    if not row["ok"]:
        row["objective"], row["constraints"] = W_OBJ, [1.0] * 6
    row["seconds"] = round(time.perf_counter() - t0, 2)
    return row


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
    fm = forward_model.ConstellarationSettings(
        vmec_preset_settings=vmec_settings.VmecPresetSettings(fidelity="very_low_fidelity"))
    alset = al.AugmentedLagrangianSettings(constraint_violation_tolerance_reduction_factor=0.8,
                                           penalty_parameters_increase_factor=P_GROW,
                                           penalty_parameters_max=P_MAX, bounds_min=B_MIN)
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
    args = (scale, unravel, fm)

    n, t_start = 0, time.time()
    with ledger.open("a") as f:
        def log(row, k, kind):
            row.update(start=name, k=k, n=n, kind=kind, wall=round(time.time() - t_start, 1))
            f.write(json.dumps(row) + "\n")
            f.flush()

        cur = evaluate(x, *args)
        log(dict(cur), 0, "start")
        state = al.AugmentedLagrangianState(
            x=jnp.array(x), multipliers=jnp.zeros(6), penalty_parameters=jnp.full(6, P0),
            objective=jnp.array(cur["objective"]), constraints=jnp.array(cur["constraints"]),
            bounds=jnp.full(len(x), B0))
        bounds = B0
        best_worst = cur["worst"] if cur["ok"] else np.inf
        print(f"[{name}] start worst {cur.get('worst')}", flush=True)
        ctx = multiprocessing.get_context("forkserver")
        rng = np.random.RandomState(123)
        for k in range(1, maxit + 1):
            def lag(row):
                return float(al.augmented_lagrangian_function(
                    jnp.array(row["objective"]), jnp.array(row["constraints"]), state))

            par = nevergrad.p.Array(init=x, lower=x - bounds, upper=x + bounds)
            par.random_state = np.random.RandomState(rng.randint(2**31))
            oracle = nevergrad.optimizers.NGOpt(parametrization=par, budget=inner, num_workers=workers)
            best = (lag(cur), x, cur)
            with futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                rest, running = inner, []
                while rest or running:
                    while len(running) < min(workers, rest):
                        cand = oracle.ask()
                        running.append((ex.submit(evaluate, np.array(cand.value), *args), cand))
                        rest -= 1
                    done, _ = futures.wait([fu for fu, _ in running], return_when=futures.FIRST_COMPLETED)
                    for fu, cand in running:
                        if fu in done:
                            n += 1
                            row = fu.result()
                            val = lag(row)
                            oracle.tell(cand, val)
                            if val < best[0]:
                                best = (val, np.array(cand.value), row)
                            log(dict(row), k, "oracle")
                    running = [(fu, c) for fu, c in running if fu not in done]
            _, x_new, row = best
            moved = x_new is not x
            x, cur = x_new, row
            state = al.update_augmented_lagrangian_state(
                x=jnp.array(x), objective=jnp.array(row["objective"]),
                constraints=jnp.array(row["constraints"]), state=state, settings=alset)
            improved = row["ok"] and row["worst"] < best_worst - 1e-4
            best_worst = min(best_worst, row["worst"]) if row["ok"] else best_worst
            if not improved:
                bounds = max(B_MIN, bounds * SHRINK)
            log({"worst": row.get("worst"), "L_gradB": row.get("L_gradB"), "moved": moved,
                 "bounds": bounds, "penalties": np.asarray(state.penalty_parameters).tolist(),
                 "multipliers": np.asarray(state.multipliers).tolist()}, k, "iteration")
            print(f"[{name}] it {k}: runs {n}, current worst {row.get('worst')}, "
                  f"L_gradB {row.get('L_gradB')}, box {bounds:.4f}, {(time.time() - t_start) / 60:.0f} min",
                  flush=True)


def read(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()] if p.exists() else []


def report(ledger: Path, dry: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from evaluate import evaluate_designs

    tag = "_dry" if dry else ""
    rows = read(ledger)
    names = list(dict.fromkeys(r["start"] for r in rows))
    summary = {"settings": {"margin": MARGIN, "w_obj": W_OBJ, "p0": P0, "box": [B0, B_MIN, SHRINK],
                            "inner": INNER, "maxit": MAXIT, "aspect_ratio_bound": AR_BOUND}, "starts": {}}
    to_check = []
    fig, ax = plt.subplots(figsize=(7, 4))
    for s in names:
        rs = [r for r in rows if r["start"] == s]
        ev = [r for r in rs if r["kind"] == "oracle"]
        w = np.array([r["worst"] if r["ok"] else np.inf for r in ev])
        ax.plot(np.arange(1, len(ev) + 1), np.minimum.accumulate(w), label=s)
        feas = [r for r in ev if r.get("feasible")]
        seen, picks = set(), []
        for r in sorted((r for r in ev if "boundary" in r), key=lambda r: (r["worst"], -r["L_gradB"])):
            key = round(r["worst"], 4)
            if key not in seen:
                seen.add(key)
                picks.append(r)
            if len(picks) == (1 if dry else HI_CHECK):
                break
        for j, r in enumerate(picks):
            to_check.append({"id": f"polish_{s}_n{r['n']}_{j}", "boundary": r["boundary"], "start": s,
                             "worst_vlo": r["worst"]})
        summary["starts"][s] = {
            "start_worst_very_low": rs[0].get("worst"), "runs": len(ev),
            "failed_frac": round(float(np.isinf(w).mean()), 3) if len(ev) else None,
            "best_worst_very_low": float(w.min()) if len(ev) else None,
            "first_feasible_run": feas[0]["n"] if feas else None, "n_feasible_very_low": len(feas),
            "best_feasible_L_gradB_very_low": max((r["L_gradB"] for r in feas), default=None),
            "hours": round(ev[-1]["wall"] / 3600, 2) if ev else None,
        }
    ax.axhline(0.01, color="k", ls=":", lw=1)
    ax.text(1, 0.011, "feasible (≤ 0.01)", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("VMEC++ runs (very low fidelity)")
    ax.set_ylabel("best worst violation so far")
    ax.set_title("Polishing ALM-NGOpt, by start design")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / f"polish{tag}.png", dpi=150)

    if to_check:
        hi = {r["id"]: r for r in evaluate_designs(to_check, DATA / f"polish_hi_ledger{tag}.jsonl")}
        with open(OUT / f"polish_checked_designs{tag}.jsonl", "w") as f:
            for d in to_check:
                r = hi[d["id"]]
                rec = {"id": d["id"], "start": d["start"], "worst_vlo": d["worst_vlo"], "ok": r["ok"],
                       "feasible_hi": r["feasible"], "worst_hi": r.get("max_violation"),
                       "L_gradB": r.get("L_gradB"), "aspect_ratio": r.get("aspect_ratio"),
                       "violations": r.get("violations")}
                summary["starts"][d["start"]].setdefault("high_fidelity_checks", []).append(rec)
                f.write(json.dumps({**rec, "boundary": d["boundary"]}) + "\n")
        feas_hi = [c for v in summary["starts"].values() for c in v.get("high_fidelity_checks", [])
                   if c["feasible_hi"]]
        summary["feasible_high_fidelity"] = len(feas_hi)
        summary["best_high_fidelity"] = min((c for v in summary["starts"].values()
                                             for c in v.get("high_fidelity_checks", []) if c["ok"]),
                                            key=lambda c: c["worst_hi"], default=None)
    (OUT / f"polish{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    ledger = DATA / f"polish_ledger{'_dry' if a.dry else ''}.jsonl"
    if not a.report:
        done = {r["start"] for r in read(ledger)}
        todo = starts()[:1] if a.dry else starts()
        print("starts:", [(s, round(w, 3)) for s, _, w in todo], flush=True)
        for name, bjson, _ in todo:
            if name in done:
                print(f"[{name}] already in ledger; skipping")
                continue
            run_start(name, bjson, a.dry, ledger)
    report(ledger, a.dry)


if __name__ == "__main__":
    main()
