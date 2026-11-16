"""Test 7: warm-start the paper's ALM-NGOpt optimizer from our best design.

Same optimizer and settings as the paper's Benchmark 3 baseline
(optimization_examples/launch_alm_mhd_stable_stellarator.py, constellaration v0.2.6),
started from three places, with the same evaluation budget each:

    nae      the paper's own start (a near-axis expansion shape, nfp 3)
    dataset  the dataset's closest-to-feasible nfp-3 design (worst violation 0.203)
    ours     Test 6's best design (worst violation 0.078 at high fidelity)

The question: how many VMEC++ runs until the first design that passes Benchmark 3
(at very low fidelity, the fidelity the optimizer itself uses)? Every evaluation is
logged; afterwards the first feasible design and the best few per arm are re-checked
at high fidelity, the official fidelity.

The ALM loop below is the package's augmented_lagrangian_runner.run, copied so we
can log each evaluation; the optimizer logic is unchanged.

    python scripts/feasibility_tests/alm_warmstart_test.py            # all arms
    python scripts/feasibility_tests/alm_warmstart_test.py --dry      # plumbing check
    python scripts/feasibility_tests/alm_warmstart_test.py --report   # re-plot from ledger
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

# registers SurfaceRZFourier as a JAX pytree; worker processes need this at import
# time to unpickle the boundary-unravel function
import constellaration.optimization.augmented_lagrangian_runner  # noqa: F401, E402

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
from common import ROOT  # noqa: E402

ARMS = ["ours", "dataset", "nae"]  # most informative first
ASPECT_RATIO_BOUND = 12.0  # one of the paper's four runs; above all three starts' AR
OUTER_ITERS = 3  # paper: up to 50; budget per outer iteration 1500, +300 each time
DRY_BUDGET = 24
HI_CHECK_TOP = 4  # best designs per arm re-checked at high fidelity
KEEP_BOUNDARY_BELOW = 0.3  # store boundaries only for designs this close (ledger size)

OUT = ROOT / "results" / "feasibility_tests"
DATA = ROOT / "data" / "feasibility_tests"


def settings(dry: bool):
    import constellaration.forward_model as forward_model
    import constellaration.optimization.augmented_lagrangian as al
    import constellaration.optimization.settings as opt
    from constellaration.mhd import vmec_settings

    from evaluate import default_workers

    return opt.OptimizationSettings(  # paper's settings, except num_workers
        max_poloidal_mode=4,
        max_toroidal_mode=4,
        infinity_norm_spectrum_scaling=1.5,
        optimizer_settings=opt.AugmentedLagrangianMethodSettings(
            maxit=1 if dry else OUTER_ITERS,
            oracle_settings=opt.NevergradSettings(
                num_workers=default_workers(),
                budget_initial=DRY_BUDGET if dry else 1500,
                budget_max=20000,
                budget_increment=300,
                max_time=None,
                batch_mode=False,
            ),
            penalty_parameters_initial=10,
            bounds_initial=0.33,
            augmented_lagrangian_settings=al.AugmentedLagrangianSettings(
                constraint_violation_tolerance_reduction_factor=0.8,
                penalty_parameters_increase_factor=5,
                bounds_reduction_factor=0.95,
                penalty_parameters_max=1e8,
                bounds_min=0.05,
            ),
        ),
        forward_model_settings=forward_model.ConstellarationSettings(
            vmec_preset_settings=vmec_settings.VmecPresetSettings(
                fidelity="very_low_fidelity"
            )
        ),
    )


def start_boundary(arm: str):
    import constellaration.initial_guess as init
    from constellaration.geometry import surface_rz_fourier as rz

    if arm == "nae":  # exactly the paper's start
        return init.generate_nae(aspect_ratio=20.0 * 0.25 * 3, max_elongation=4.0,
                                 rotational_transform=0.25 * 3, mirror_ratio=0.25,
                                 n_field_periods=3, max_poloidal_mode=1, max_toroidal_mode=1)
    if arm == "dataset":
        from common import load_designs
        _, _, viol, meta = load_designs(3)
        best = viol.clip(lower=0).max(axis=1).idxmin()
        return rz.SurfaceRZFourier.model_validate_json(meta.loc[best, "boundary_json"])
    if arm == "ours":
        rows = [json.loads(line) for line in open(OUT / "push_checked_designs.jsonl")]
        best = min((r for r in rows if r.get("ok")), key=lambda r: r["worst_hi"])
        return rz.SurfaceRZFourier.model_validate_json(best["boundary"])
    raise ValueError(arm)


def evaluate(x, scale, problem, unravel, fm_settings, ar_bound):
    """ALM objective/constraints (as the paper) plus the Benchmark 3 violations."""
    import jax.numpy as jnp

    import constellaration.optimization.augmented_lagrangian_runner as runner

    t0 = time.perf_counter()
    (obj, cons), metrics = runner.objective_constraints(
        jnp.asarray(x), scale, problem, unravel, fm_settings, ar_bound)
    row = {"objective": float(obj), "constraints": np.asarray(cons).tolist(),
           "ok": metrics is not None}
    if metrics is not None:
        v = np.asarray(problem._normalized_constraint_violations(metrics), float)
        row.update(worst=float(np.maximum(v, 0).max()), feasible=bool(problem.is_feasible(metrics)),
                   violations=v.tolist(), L_gradB=float(metrics.minimum_normalized_magnetic_gradient_scale_length),
                   aspect_ratio=float(metrics.aspect_ratio))
        if row["worst"] < KEEP_BOUNDARY_BELOW:
            row["boundary"] = unravel(jnp.asarray(x) * scale).model_dump_json()
    row["seconds"] = round(time.perf_counter() - t0, 2)
    return row


def run_arm(arm: str, dry: bool, ledger: Path) -> None:
    """augmented_lagrangian_runner.run, logging every evaluation to the ledger."""
    import jax.numpy as jnp
    import nevergrad

    import constellaration.geometry.surface_rz_fourier as rz
    import constellaration.optimization.augmented_lagrangian as al
    import constellaration.problems as problems
    import constellaration.utils.seed_util as seed_util
    from constellaration.utils import pytree

    s = settings(dry)
    o = s.optimizer_settings
    seed_util.seed_everything(123)
    problem = problems.MHDStableQIStellarator()
    boundary = rz.set_max_mode_numbers(start_boundary(arm), max_poloidal_mode=s.max_poloidal_mode,
                                       max_toroidal_mode=s.max_toroidal_mode)
    mask = rz.build_mask(boundary, max_poloidal_mode=s.max_poloidal_mode,
                         max_toroidal_mode=s.max_toroidal_mode)
    x0, unravel = pytree.mask_and_ravel(pytree=boundary, mask=mask)
    scale = rz.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes=boundary.poloidal_modes.flatten(),
        toroidal_modes=boundary.toroidal_modes.flatten(),
        alpha=s.infinity_norm_spectrum_scaling).reshape(boundary.poloidal_modes.shape)
    scale = jnp.array(np.concatenate([scale[mask.r_cos], scale[mask.z_sin]]))
    x0 = jnp.array(x0) / scale
    args = (scale, problem, unravel, s.forward_model_settings, ASPECT_RATIO_BOUND)

    n = 0
    t_start = time.time()
    with ledger.open("a") as f:
        def log(row, k):
            nonlocal n
            row.update(arm=arm, k=k, n=n, wall=round(time.time() - t_start, 1))
            f.write(json.dumps(row) + "\n")
            f.flush()

        first = evaluate(x0, *args)
        log({**first, "kind": "start"}, 0)
        print(f"[{arm}] start: worst {first.get('worst')}, AR {first.get('aspect_ratio')}", flush=True)
        state = al.AugmentedLagrangianState(
            x=jnp.copy(x0), multipliers=jnp.zeros(len(first["constraints"])),
            penalty_parameters=jnp.ones(len(first["constraints"])),
            objective=jnp.array(first["objective"]), constraints=jnp.array(first["constraints"]),
            bounds=jnp.ones_like(x0) * o.bounds_initial)

        budget = o.oracle_settings.budget_initial
        ctx = multiprocessing.get_context("forkserver")
        for k in range(o.maxit):
            par = nevergrad.p.Array(init=np.array(state.x), lower=np.array(state.x - state.bounds),
                                    upper=np.array(state.x + state.bounds))
            par.random_state.set_state(np.random.get_state())  # noqa: NPY002
            oracle = nevergrad.optimizers.NGOpt(parametrization=par, budget=o.oracle_settings.budget_initial,
                                                num_workers=o.oracle_settings.num_workers)
            oracle.suggest(np.array(state.x))
            with futures.ProcessPoolExecutor(max_workers=o.oracle_settings.num_workers,
                                             mp_context=ctx) as ex:
                rest, running = budget, []
                while rest or running:
                    while len(running) < min(o.oracle_settings.num_workers, rest):
                        cand = oracle.ask()
                        running.append((ex.submit(evaluate, np.array(cand.value), *args), cand))
                        rest -= 1
                    done, _ = futures.wait([fu for fu, _ in running], return_when=futures.FIRST_COMPLETED)
                    for fu, cand in running:
                        if fu in done:
                            n += 1
                            row = fu.result()
                            oracle.tell(cand, al.augmented_lagrangian_function(
                                jnp.array(row["objective"]), jnp.array(row["constraints"]), state).item())
                            log({**row, "kind": "oracle"}, k + 1)
                            if row.get("feasible"):
                                print(f"[{arm}] FEASIBLE at very low fidelity, run {n}", flush=True)
                    running = [(fu, c) for fu, c in running if fu not in done]
                    if n % 250 == 0 and done:
                        print(f"[{arm}] k={k + 1} runs {n}", flush=True)
                x = oracle.provide_recommendation().value

            row = evaluate(x, *args)
            log({**row, "kind": "recommendation"}, k + 1)
            state = al.update_augmented_lagrangian_state(
                x=jnp.copy(x), objective=jnp.array(row["objective"]),
                constraints=jnp.array(row["constraints"]), state=state,
                settings=o.augmented_lagrangian_settings)
            budget = int(min(o.oracle_settings.budget_max, budget + o.oracle_settings.budget_increment))
            print(f"[{arm}] outer iteration {k + 1}: runs {n}, recommendation worst "
                  f"{row.get('worst')}, AR {row.get('aspect_ratio')}, "
                  f"{(time.time() - t_start) / 60:.0f} min", flush=True)


def read(ledger: Path) -> list[dict]:
    return [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]


def report(ledger: Path, dry: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from evaluate import evaluate_designs

    rows = read(ledger)
    arms = [a for a in ARMS if any(r["arm"] == a for r in rows)]
    summary, to_check = {"aspect_ratio_bound": ASPECT_RATIO_BOUND, "arms": {}}, []
    fig, ax = plt.subplots(figsize=(7, 4))
    for a in arms:
        rs = [r for r in rows if r["arm"] == a]
        evals = [r for r in rs if r["kind"] == "oracle"]
        worst = np.array([r["worst"] if r["ok"] else np.nan for r in evals])
        best_so_far = np.fmin.accumulate(np.where(np.isnan(worst), np.inf, worst))
        ax.plot(np.arange(1, len(evals) + 1), best_so_far, label=a)
        feas = [r for r in evals if r.get("feasible")]
        start = rs[0]
        ok = sorted((r for r in rs if r["ok"] and "boundary" in r), key=lambda r: r["worst"])
        picks = ([feas[0]] if feas else []) + [r for r in ok if not feas or r is not feas[0]][:1 if dry else HI_CHECK_TOP]
        recs = [r for r in rs if r["kind"] == "recommendation"]
        for j, r in enumerate(picks):
            to_check.append({"id": f"alm_{a}_n{r['n']}_{j}", "boundary": r["boundary"]})
        summary["arms"][a] = {
            "start_worst_very_low": start.get("worst"), "start_aspect_ratio": start.get("aspect_ratio"),
            "runs": len(evals), "failed_frac": round(float(np.isnan(worst).mean()), 3),
            "best_worst_very_low": float(np.nanmin(worst)) if len(evals) else None,
            "first_feasible_run": feas[0]["n"] if feas else None,
            "n_feasible_very_low": len(feas),
            "recommendations": [{"k": r["k"], "worst": r.get("worst"), "aspect_ratio": r.get("aspect_ratio"),
                                 "L_gradB": r.get("L_gradB")} for r in recs],
            "hours": round(evals[-1]["wall"] / 3600, 2) if evals else None,
        }
    ax.axhline(0.01, color="k", ls=":", lw=1)
    ax.text(1, 0.012, "feasible (≤ 0.01)", fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("VMEC++ runs (very low fidelity)")
    ax.set_ylabel("best worst violation so far")
    ax.set_title(f"ALM-NGOpt (paper settings, AR ≤ {ASPECT_RATIO_BOUND:g}) by starting point")
    ax.legend()
    fig.tight_layout()
    tag = "_dry" if dry else ""
    fig.savefig(OUT / f"alm_warmstart{tag}.png", dpi=150)

    if to_check:
        hi = {r["id"]: r for r in evaluate_designs(to_check, DATA / f"alm_hi_ledger{tag}.jsonl")}
        for a in arms:
            chk = [hi[d["id"]] for d in to_check if d["id"].startswith(f"alm_{a}_")]
            summary["arms"][a]["high_fidelity_checks"] = [
                {"id": r["id"], "ok": r["ok"], "feasible": r["feasible"], "worst": r.get("max_violation"),
                 "aspect_ratio": r.get("aspect_ratio"), "L_gradB": r.get("L_gradB"),
                 "violations": r.get("violations")} for r in chk]
        with open(OUT / f"alm_checked_designs{tag}.jsonl", "w") as f:
            for d in to_check:
                r = hi[d["id"]]
                f.write(json.dumps({"id": d["id"], "ok": r["ok"], "feasible_hi": r["feasible"],
                                    "worst_hi": r.get("max_violation"), "boundary": d["boundary"]}) + "\n")
    (OUT / f"alm_warmstart{tag}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--report", action="store_true", help="only re-plot / re-check from the ledger")
    p.add_argument("--arms", nargs="+", default=ARMS)
    a = p.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    ledger = DATA / f"alm_ledger{'_dry' if a.dry else ''}.jsonl"
    if not a.report:
        done = {r["arm"] for r in read(ledger)} if ledger.exists() else set()
        for arm in a.arms:
            if arm in done:
                print(f"[{arm}] already in {ledger}; skipping (delete its rows to rerun)")
                continue
            run_arm(arm, a.dry, ledger)
    report(ledger, a.dry)


if __name__ == "__main__":
    main()
