"""Refine the leaders' best public designs with the paper's optimizer.

The full public Benchmark 3 front (HV 285.27: 569 designs, NFP 3 and 4, 8x8-9x9 modes) has
only ever been *blended* by us (fullres.py, Nov 16): blends of two front designs land
0.5-1.5 L_gradB below the front, because a blend sits between its parents. This instead
warm-starts the paper's ALM-NGOpt (constellaration 0.2.6, augmented_lagrangian_runner.run;
the optimizer logic is unchanged) at one front design and asks it to do better:

  objective    maximize L_gradB (the paper's Benchmark 3 objective, 20 - L_gradB)
  constraints  the five Benchmark 3 checks, plus aspect ratio <= the start's own A
  resolution   the start's own (8x8 or 9x9 modes), so the start is exactly the leader's design
  fidelity     high, the official scorer's, not the paper's very low fidelity. On 4 front
               designs very low fidelity was off by up to 4.2 in L_gradB and wrong on
               pass/fail for 2 of 4; low fidelity was off by up to 1.8
               (data/refine/fidelity_probe.jsonl). So every passing design a chain logs
               is already verified at the official fidelity.
  search box   small (--box, in the optimizer's scaled units) so steps stay local; `box`
               prints the typical first-step size (~RMS surface distance)

Each chain refines one design on one core, one simulation at a time, so 8 chains use 8
cores. A simulation past LIMIT seconds is killed and counted as failed. Rerunning a chain
warm-starts from the best passing design in its ledger (same A bound).

    python scripts/seeded/refine_leaders.py pick --n 8      # starts -> data/refine/starts.json
    python scripts/seeded/refine_leaders.py box <start_id>  # first-step sizes, no simulations
    python scripts/seeded/refine_leaders.py chain <start_id> [--budget 1000] [--hours 20]
    python scripts/seeded/refine_leaders.py report
"""

from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")  # one thread per simulation (see run_sims.py)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent import futures  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

# registers SurfaceRZFourier as a JAX pytree; worker processes need it to unpickle `unravel`
import constellaration.optimization.augmented_lagrangian_runner  # noqa: F401, E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).parent), str(ROOT / "scripts")]
from loop import hypervolume  # noqa: E402

PUBLIC = ROOT / "data" / "seeded" / "public_p3.jsonl"
OUT = ROOT / "data" / "refine"
STARTS = OUT / "starts.json"
CHAINS = OUT / "chains"
LIMIT = 450.0             # s per simulation, as run_sims.py
KEEP_BOUNDARY_BELOW = 0.05  # store boundaries for designs at most this far over the line
ALM = dict(penalty_parameters_initial=10, constraint_violation_tolerance_reduction_factor=0.8,
           penalty_parameters_increase_factor=5, bounds_reduction_factor=0.95,
           penalty_parameters_max=1e8, bounds_min=0.0)  # paper's, except bounds_min (box is already small)
SPECTRUM_ALPHA = 1.5      # paper's infinity_norm_spectrum_scaling


def public_front() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """Every passing public design (any NFP, any resolution), and the indices of the front."""
    rows = [r for r in map(json.loads, PUBLIC.open()) if r["feasible"]]
    A = np.array([r["aspect_ratio"] for r in rows]); L = np.array([r["L_gradB"] for r in rows])
    front, best = [], -np.inf
    for i in np.argsort(A, kind="stable"):
        if L[i] > best:
            front.append(int(i)); best = L[i]
    return rows, A, L, np.array(front)


def front_height(a: float, A: np.ndarray, L: np.ndarray) -> float:
    """Best L_gradB among public passing designs with aspect ratio <= a."""
    k = A <= a
    return float(L[k].max()) if k.any() else -1.0


def pick(n: int) -> None:
    """n front designs spread along A: split the front into n equal-count stretches and take
    the design with the most room to spare (smallest worst violation) in each."""
    rows, A, L, front = public_front()
    out = []
    for part in np.array_split(front, n):
        i = min(part, key=lambda j: rows[j]["max_violation"])
        r = rows[i]
        out.append({"id": r["id"], "nfp": r["nfp"], "shape": r["shape"], "A": r["aspect_ratio"],
                    "L": r["L_gradB"], "max_violation": r["max_violation"],
                    "user": r["sources"][0]["user"], "boundary": r["boundary"]})
    OUT.mkdir(parents=True, exist_ok=True)
    STARTS.write_text(json.dumps(out, indent=1))
    print(f"front: {len(front)} designs, HV {hypervolume(A, L):.3f}")
    for s in out:
        print(f"  {s['id']}  NFP {s['nfp']} {s['shape']}  A {s['A']:.3f}  L {s['L']:.3f}  "
              f"worst {s['max_violation']:.4f}  ({s['user']})")


def start_of(sid: str) -> dict:
    return next(s for s in json.loads(STARTS.read_text()) if s["id"] == sid)


def setup(boundary_json: str, shape, modes: int = 3):
    """The optimizer's variables, as runner.run, but only modes up to `modes` are free: the
    boundary keeps its own resolution and every higher mode stays at the leader's value."""
    import jax.numpy as jnp

    from constellaration.geometry import surface_rz_fourier as rz
    from constellaration.utils import pytree

    mp, mt = shape[0] - 1, (shape[1] - 1) // 2
    b = rz.SurfaceRZFourier.model_validate_json(boundary_json)
    b = rz.set_max_mode_numbers(b, max_poloidal_mode=mp, max_toroidal_mode=mt)
    mask = rz.build_mask(b, max_poloidal_mode=min(modes, mp), max_toroidal_mode=min(modes, mt))
    x0, unravel = pytree.mask_and_ravel(pytree=b, mask=mask)
    scale = rz.compute_infinity_norm_spectrum_scaling_fun(
        poloidal_modes=b.poloidal_modes.flatten(), toroidal_modes=b.toroidal_modes.flatten(),
        alpha=SPECTRUM_ALPHA).reshape(b.poloidal_modes.shape)
    scale = jnp.array(np.concatenate([scale[mask.r_cos], scale[mask.z_sin]]))
    return jnp.array(x0) / scale, scale, unravel


def evaluate(x, scale, unravel, ar_bound) -> dict:
    """The paper's objective/constraints at high fidelity, plus the Benchmark 3 verdict."""
    import jax.numpy as jnp

    import constellaration.forward_model as fm
    import constellaration.optimization.augmented_lagrangian_runner as runner
    import constellaration.problems as problems

    t0 = time.perf_counter()
    problem = problems.MHDStableQIStellarator()
    (obj, cons), m = runner.objective_constraints(
        jnp.asarray(x), scale, problem, unravel, fm.ConstellarationSettings.default_high_fidelity(), ar_bound)
    row = {"objective": float(obj), "constraints": np.asarray(cons).tolist(), "ok": m is not None}
    if m is not None:
        v = np.asarray(problem._normalized_constraint_violations(m), float)
        row.update(worst=float(np.maximum(v, 0).max()), feasible=bool(problem.is_feasible(m)),
                   L_gradB=float(m.minimum_normalized_magnetic_gradient_scale_length),
                   aspect_ratio=float(m.aspect_ratio), violations=v.tolist())
        if row["worst"] <= KEEP_BOUNDARY_BELOW:
            row["boundary"] = unravel(jnp.asarray(x) * scale).model_dump_json()
    row["seconds"] = round(time.perf_counter() - t0, 1)
    return row


class OneWorker:
    """One simulation at a time in a child process, killed after LIMIT seconds."""

    def __init__(self):
        self.ctx = multiprocessing.get_context("forkserver")
        self.ex = futures.ProcessPoolExecutor(max_workers=1, mp_context=self.ctx)

    def __call__(self, *args) -> dict:
        import constellaration.optimization.augmented_lagrangian_runner as runner

        fu = self.ex.submit(evaluate, *args)
        try:
            return fu.result(timeout=LIMIT)
        except Exception as e:  # timeout or a dead worker: count as failed, start a fresh one
            for p in list(getattr(self.ex, "_processes", {}).values()):
                p.kill()
            self.ex.shutdown(wait=False, cancel_futures=True)
            self.ex = futures.ProcessPoolExecutor(max_workers=1, mp_context=self.ctx)
            bad = float(runner.NAN_TO_HIGH_VALUE)
            return {"objective": bad, "constraints": [bad] * 6, "ok": False,
                    "error": type(e).__name__, "timeout": isinstance(e, futures.TimeoutError), "seconds": LIMIT}

    def close(self):
        self.ex.shutdown(wait=True)


def oracle_for(x, bounds, budget, seed):
    import nevergrad

    par = nevergrad.p.Array(init=np.array(x), lower=np.array(x - bounds), upper=np.array(x + bounds))
    par.random_state.seed(seed)
    opt = nevergrad.optimizers.NGOpt(parametrization=par, budget=budget, num_workers=1)
    opt.suggest(np.array(x))
    return opt


def box(sid: str, modes: int, per_round: int, sizes=(0.02, 0.05, 0.1)) -> None:
    """Typical step size for each box, without simulating (~RMS surface distance). Each ask
    gets a random dummy score back: with one worker NGOpt may run its inner method in a
    thread that waits for every result, so asking without telling deadlocks."""
    s = start_of(sid)
    x0, scale, _ = setup(s["boundary"], s["shape"], modes)
    rng = np.random.default_rng(0)
    print(f"{sid}: {len(x0)} variables")
    for b in sizes:
        opt = oracle_for(x0, np.ones_like(np.array(x0)) * b, per_round, 0)
        d = []
        for _ in range(per_round):
            c = opt.ask()
            d.append(float(np.linalg.norm((np.array(c.value) - np.array(x0)) * np.array(scale)) / np.sqrt(2)))
            opt.tell(c, float(rng.random()))
        d = [x for x in d if x > 0]
        inner = getattr(getattr(opt, "optim", None), "name", type(getattr(opt, "optim", opt)).__name__)
        print(f"  box {b:<6} step (~RMS surface distance): median {np.median(d):.4f}, max {max(d):.4f}; NGOpt picks {inner}")


def chain(sid: str, budget: int, hours: float, box_size: float, per_round: int, modes: int, dry: bool) -> None:
    import jax.numpy as jnp

    import constellaration.optimization.augmented_lagrangian as al

    s = start_of(sid)
    CHAINS.mkdir(parents=True, exist_ok=True)
    ledger = CHAINS / f"{sid}{'_dry' if dry else ''}.jsonl"
    prior = [json.loads(x) for x in ledger.open()] if ledger.exists() else []
    a_bound = s["A"]
    start_json = s["boundary"]
    best = max((r for r in prior if r.get("feasible") and r["aspect_ratio"] <= a_bound and "boundary" in r),
               key=lambda r: r["L_gradB"], default=None)
    if best is not None:
        start_json = best["boundary"]
        print(f"[{sid}] resuming from its best passing design so far (L {best['L_gradB']:.4f})", flush=True)
    x0, scale, unravel = setup(start_json, s["shape"], modes)
    seed = int(hashlib.sha1(sid.encode()).hexdigest()[:6], 16)
    np.random.seed(seed)
    run = OneWorker()
    n = len(prior)
    t_start = time.time()
    stop = t_start + hours * 3600
    with ledger.open("a") as f:
        def log(row, k, kind):
            nonlocal n
            row.update(start=sid, k=k, n=n, kind=kind, wall=round(time.time() - t_start, 1))
            f.write(json.dumps(row) + "\n"); f.flush()
            n += 1

        first = run(x0, scale, unravel, a_bound)
        log(dict(first), 0, "start")
        print(f"[{sid}] start: A {first.get('aspect_ratio')} L {first.get('L_gradB')} worst {first.get('worst')} "
              f"({first['seconds']} s); public says A {s['A']:.4f} L {s['L']:.4f}", flush=True)
        state = al.AugmentedLagrangianState(
            x=jnp.copy(x0), multipliers=jnp.zeros(len(first["constraints"])),
            penalty_parameters=jnp.ones(len(first["constraints"])),
            objective=jnp.array(first["objective"]), constraints=jnp.array(first["constraints"]),
            bounds=jnp.ones_like(x0) * box_size)
        alm = al.AugmentedLagrangianSettings(**{k: v for k, v in ALM.items() if k != "penalty_parameters_initial"})
        done = 1
        cap = 3 if dry else budget
        k = 0
        while done < cap and time.time() < stop:
            k += 1
            m = min(2 if dry else per_round, cap - done)
            oracle = oracle_for(state.x, state.bounds, m, seed + k)
            for _ in range(m):
                if time.time() >= stop:
                    break
                cand = oracle.ask()
                row = run(jnp.array(cand.value), scale, unravel, a_bound)
                oracle.tell(cand, al.augmented_lagrangian_function(
                    jnp.array(row["objective"]), jnp.array(row["constraints"]), state).item())
                log(dict(row), k, "oracle"); done += 1
                if row.get("feasible") and row["L_gradB"] > s["L"] + 1e-3 and row["aspect_ratio"] <= a_bound:
                    print(f"[{sid}] BEYOND START: A {row['aspect_ratio']:.4f} L {row['L_gradB']:.4f} "
                          f"(+{row['L_gradB'] - s['L']:.4f}), eval {n}", flush=True)
            x = oracle.provide_recommendation().value
            row = run(jnp.array(x), scale, unravel, a_bound); done += 1
            log(dict(row), k, "recommendation")
            state = al.update_augmented_lagrangian_state(
                x=jnp.copy(jnp.array(x)), objective=jnp.array(row["objective"]),
                constraints=jnp.array(row["constraints"]), state=state, settings=alm)
            print(f"[{sid}] round {k}: {done} sims, recommendation L {row.get('L_gradB')} worst {row.get('worst')}, "
                  f"{(time.time() - t_start) / 60:.0f} min", flush=True)
    run.close()


def report() -> None:
    rows, A, L, front = public_front()
    base = hypervolume(A, L)
    starts = json.loads(STARTS.read_text())
    allA, allL = [], []
    print(f"public front HV {base:.3f}")
    for s in starts:
        led = CHAINS / f"{s['id']}.jsonl"
        if not led.exists():
            print(f"  {s['id']}: not started"); continue
        rs = [json.loads(x) for x in led.open()]
        ok = [r for r in rs if r.get("ok")]
        feas = [r for r in ok if r.get("feasible")]
        allA += [r["aspect_ratio"] for r in feas]; allL += [r["L_gradB"] for r in feas]
        beyond = [r for r in feas if r["L_gradB"] > front_height(r["aspect_ratio"], A, L)]
        best = max((r for r in feas if r["aspect_ratio"] <= s["A"]), key=lambda r: r["L_gradB"], default=None)
        gain = f"best L {best['L_gradB']:.4f} ({best['L_gradB'] - s['L']:+.4f})" if best else "no passing design"
        print(f"  {s['id']} A {s['A']:.3f} L {s['L']:.3f}: {len(rs)} sims, {len(rs) - len(ok)} failed, "
              f"{len(feas)} pass, {len(beyond)} above the public front; {gain}; "
              f"{np.median([r['seconds'] for r in rs]):.0f} s median")
    if allA:
        hv = hypervolume(np.r_[A, allA], np.r_[L, allL])
        print(f"public + refined HV {hv:.3f} ({hv - base:+.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["pick", "box", "chain", "report"])
    ap.add_argument("start", nargs="?")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--budget", type=int, default=1000, help="simulations per chain")
    ap.add_argument("--hours", type=float, default=20.0, help="stop a chain after this long")
    ap.add_argument("--box", type=float, default=0.01, help="search box half-width, scaled units")
    ap.add_argument("--per-round", type=int, default=150, help="optimizer evaluations per ALM round")
    ap.add_argument("--modes", type=int, default=3, help="free modes: max(|m|, |n|) <= this")
    ap.add_argument("--dry", action="store_true", help="3 simulations, separate ledger")
    a = ap.parse_args()
    if a.cmd == "pick":
        pick(a.n)
    elif a.cmd == "box":
        box(a.start, a.modes, a.per_round)
    elif a.cmd == "chain":
        chain(a.start, a.budget, a.hours, a.box, a.per_round, a.modes, a.dry)
    else:
        report()


if __name__ == "__main__":
    main()
