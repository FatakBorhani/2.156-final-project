"""Golden check: do Track B's scripts still reproduce the reported runs exactly?

Run it before and after any change to scripts/seeded/. Each check redoes a reported
computation and compares it, field by field, with what was recorded at the time:

  imports    every module in scripts/seeded/ imports; the names the notebooks use exist
  hv         the public NFP-3 4x4 front scores 205.518..., the full public front 285.270...
  front      ties on the full public front: refine_leaders.pick(8) rewrites starts.json
             exactly, fullres's controls lie on front_mask's front, front_height at the starts
  table      training_data.build(), rebuilt in memory, equals data/seeded/train.npz
  proposers  perturb.py and surface.py, run as scripts, rewrite pilot.jsonl and surface.jsonl;
             the gmm generator's pool holds its arm's round-0 picks; modes5_control.designs()
  loop       round 0 of four arms (b60 random and screened, w60, s60), picked again
  logs       every logged round of every loop.py and track_a_loop.py arm, recomputed
  track_a    rounds 0 and 1 of ta_random_s50 and round 0 of ta_screen_s50, picked again
  report     report_numbers.py, pinned to the runs it reported, vs results/seeded/
  fullres    only when named (~2 min): fullres's two proposal calls vs pilot_designs.jsonl

A change in the number or order of random draws shows up as a different design id.
Floats must agree to 1e-9 (relative above 1, absolute below); thread count alone moves
the screens' predictions by ~1e-15. A replayed round sees what the original did: the
shared data plus the arm's earlier rounds, never its later ones.

It writes nothing under the repo. Every open() for writing there is refused, so a
refactor that makes a check write fails that check instead; build() runs with saving
off; scripts that write output are pointed at a temporary directory; and at the end
every data and results file is checked for change. Don't run it alongside a loop.py
run (the refinement pilot is fine: it writes only data/refine/chains and logs).

It reads data/ (gitignored), so it runs only where the project's data is. Data, not
code, can also fail it: the table check if simulations were added since train.npz was
built (run training_data.py); loop and logs if non-loop simulations land in
data/seeded/runs/; track_a if a new *_feasible_designs.jsonl lands in
results/feasibility_tests/.

    python scripts/seeded/check_golden.py              # all but fullres, ~2.5 min on one core
    python scripts/seeded/check_golden.py loop hv      # some checks
    python scripts/seeded/check_golden.py fullres      # the slow one
"""

from __future__ import annotations

import os
import sys

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"            # light on a busy Mac; other thread counts agree to ~1e-15
os.environ.setdefault("MPLBACKEND", "Agg")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import builtins  # noqa: E402
import contextlib  # noqa: E402
import importlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
ROOT = HERE.parents[1]
SEEDED, TRACK_A = ROOT / "data" / "seeded", ROOT / "data" / "track_a"
# checked for change at the end; data/refine/chains and logs are left out (the pilot writes them)
WATCHED = [SEEDED, TRACK_A, ROOT / "data" / "fullres", ROOT / "data" / "gen", ROOT / "results",
           ROOT / "data" / "refine" / "starts.json"]

# Track B names the committed notebooks import; a refactor must keep them importable.
NOTEBOOK_NAMES = {
    "families": ["NOVELTY", "STRICT", "load", "nearest_public"],
    "generators": ["FINALISTS", "front_height"],
    "loop": ["REF_A", "REF_L", "arm_view", "hv_gain", "hypervolume"],
    "perturb": ["front_mask"],
    "screen": ["TOL", "load_table"],
    "shapes": ["to_vector"],
    "training_data": ["build"],
}
HV_PUBLIC_4X4 = 205.51844659839378      # results/seeded/*.json, README
HV_PUBLIC_ALL = 285.2708758861965       # refine_leaders.py report
# (arm, loop.pick keyword arguments), as run by data/scratch/*.sh
LOOP_ARMS = [
    ("b60_random_s10", dict(screen=False, seed=10)),
    ("b60_screen_s10", dict(screen=True, seed=10)),
    ("w60_random_s30", dict(screen=False, seed=30, layers=3, window=2.0)),
    ("s60_random_s40", dict(screen=False, seed=40, layers=3, window=2.0, min_novelty=0.01)),
]
# (arm, screened, round); track_a_loop.py's defaults otherwise
TRACK_A_REPLAYS = [("ta_random_s50", False, 0), ("ta_random_s50", False, 1), ("ta_screen_s50", True, 0)]
# logged rounds whose summary predates shared simulations every later view includes
LATE = {("blend_random_s0", 0): SEEDED / "runs" / "surface_retry.jsonl"}   # finished 18:24 Nov 15
REPLAY = "__golden__"       # a loop.py arm name with no simulations of its own
FLOAT_TOL = 1e-9            # a changed computation moves values far more than this
OPTIONAL = {"fullres"}      # run only when named

results: list[tuple[bool, str, float]] = []
_MISSING = object()


# ---------------------------------------------------------------- helpers

def report(ok: bool, msg: str, t0: float) -> None:
    results.append((ok, msg, time.time() - t0))
    print(f"{'PASS' if ok else 'FAIL'}  {msg}  ({time.time() - t0:.0f} s)", flush=True)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh]


def quiet(fn, *args, **kw):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows: crashed designs
        return fn(*args, **kw)


@contextlib.contextmanager
def patched(obj, **attrs):
    """Set module attributes for the duration; fails loudly if one no longer exists."""
    saved = {k: getattr(obj, k) for k in attrs}
    for k, v in attrs.items():
        setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in saved.items():
            setattr(obj, k, v)


@contextlib.contextmanager
def refuse_writes():
    """Every open() for writing under the repo raises PermissionError instead."""
    root, real = os.path.realpath(ROOT), io.open

    def guarded(file, mode="r", *args, **kw):
        if isinstance(file, (str, bytes, os.PathLike)) and set(mode) & set("wax+"):
            p = os.path.realpath(os.fsdecode(file))
            if p == root or p.startswith(root + os.sep):
                raise PermissionError(f"check_golden refused a write to {os.path.relpath(p, root)}")
        return real(file, mode, *args, **kw)

    targets = [(builtins, "open"), (io, "open")]
    accessor = getattr(pathlib, "_NormalAccessor", None)      # Python 3.10: Path.open goes here
    if accessor is not None and "open" in vars(accessor):
        targets.append((accessor, "open"))
    saved = [(obj, name, vars(obj)[name]) for obj, name in targets]
    try:
        for obj, name in targets:
            setattr(obj, name, staticmethod(guarded) if isinstance(obj, type) else guarded)
        yield
    finally:
        for obj, name, orig in saved:
            setattr(obj, name, orig)


def snapshot() -> dict:
    files = {}
    for w in WATCHED:
        for p in [w] if w.is_file() else (w.rglob("*") if w.exists() else []):
            if p.is_file():
                s = p.stat()
                files[p] = (s.st_mtime_ns, s.st_size)
    return files


def change(x, y) -> float:
    """0 if equal; for two finite floats the relative change (absolute below 1); else inf."""
    if isinstance(x, float) and isinstance(y, float):
        if math.isnan(x) and math.isnan(y):
            return 0.0
        if math.isfinite(x) and math.isfinite(y):
            return abs(x - y) / max(1.0, abs(x), abs(y))
    return 0.0 if x is not _MISSING and y is not _MISSING and x == y else math.inf


def leaf_changes(a, b, path: str = ""):
    """(path, change) for every leaf of two JSON values."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a.keys() | b.keys():
            yield from leaf_changes(a.get(k, _MISSING), b.get(k, _MISSING), f"{path}.{k}" if path else k)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from leaf_changes(x, y, f"{path}[{i}]")
    else:
        yield path, change(a, b)


def diff_designs(replayed: list[dict], recorded: list[dict], ignore=("arm",)) -> tuple[bool, str]:
    """Every field of every design, after the same JSON round trip the recorded file took."""
    a = [{k: v for k, v in json.loads(json.dumps(d)).items() if k not in ignore} for d in replayed]
    b = [{k: v for k, v in d.items() if k not in ignore} for d in recorded]
    if a == b:
        return True, f"{len(b)} designs identical"
    ia, ib = [d["id"] for d in a], [d["id"] for d in b]
    if ia != ib:
        first = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), min(len(ia), len(ib)))
        return False, (f"{len(ia)} designs replayed vs {len(ib)} recorded; the first {first} "
                       f"match in order, {len(set(ia) & set(ib))} in common")
    moved: dict[str, float] = {}
    for x, y in zip(a, b):
        for path, c in leaf_changes(x, y):
            if c:
                key = path.split(".")[0].split("[")[0]
                moved[key] = max(moved.get(key, 0.0), c)
    worst = max(moved.values(), default=0.0)
    return worst <= FLOAT_TOL, (f"{len(b)} designs identical in id and order; largest change "
                                + (", ".join(f"{k} {v:.1e}" for k, v in sorted(moved.items()))
                                   or "none (NaN on both sides)"))


def compare(label: str, replayed: list[dict], recorded: list[dict], t0: float, ignore=("arm",)) -> None:
    ok, msg = diff_designs(replayed, recorded, ignore)
    report(ok, f"{label}: {msg}", t0)


def compare_file(label: str, written: Path, recorded: Path, t0: float) -> None:
    """Byte-identical passes; same content in another format passes with a note."""
    if written.read_bytes() == recorded.read_bytes():
        return report(True, f"{label}: byte-identical", t0)
    ok, msg = diff_designs(read_jsonl(written), read_jsonl(recorded), ignore=())
    report(ok, f"{label}: {msg}; bytes differ (the writer's format changed)", t0)


_DISK = None


def disk_table() -> dict:
    global _DISK
    if _DISK is None:
        with np.load(SEEDED / "train.npz", allow_pickle=False) as z:
            _DISK = {k: z[k] for k in z.files}
    return _DISK


def run_script(module, argv: list[str]) -> None:
    with patched(sys, argv=argv):
        quiet(module.main)


@contextlib.contextmanager
def only_rounds(ta, upto: int):
    """Track A as it stood after round `upto`: the arm's later picks (and sims) hidden."""
    real, calls = ta.own_results, []

    def own_results(name):
        calls.append(name)
        picks, rows = real(name)
        return [d for d in picks if d["round"] <= upto], rows

    with patched(ta, own_results=own_results):
        yield
    if not calls:
        raise RuntimeError("track_a_loop.state() no longer calls own_results(); "
                           "update check_golden.only_rounds")


# ---------------------------------------------------------------- checks

def check_imports() -> None:
    t0 = time.time()
    names = sorted(p.stem for p in HERE.glob("*.py") if p.stem != "check_golden")
    broken = {}
    for n in names:
        try:
            quiet(importlib.import_module, n)
        except Exception as e:                    # noqa: BLE001 - report every broken module
            broken[n] = f"{type(e).__name__}: {e}"
    missing = [f"{m}.{a}" for m, attrs in NOTEBOOK_NAMES.items()
               if m not in broken for a in attrs if not hasattr(sys.modules[m], a)]
    report(not broken and not missing,
           f"imports: {len(names) - len(broken)}/{len(names)} modules import"
           + (f"; broken: {broken}" if broken else "")
           + (f"; notebook names missing: {missing}" if missing else
              f"; all {sum(map(len, NOTEBOOK_NAMES.values()))} notebook names present"), t0)


def check_hv() -> None:
    import families
    import loop
    import refine_leaders
    t0 = time.time()
    pub = families.load()
    for label, got, want in [
        ("public NFP-3 4x4 front", loop.hypervolume(pub["A"], pub["L"]), HV_PUBLIC_4X4),
        ("full public front", loop.hypervolume(*refine_leaders.public_front()[1:3]), HV_PUBLIC_ALL),
    ]:
        report(change(got, want) <= FLOAT_TOL, f"hv: {label} {got!r} (expected {want!r})", t0)


def check_front() -> None:
    """HV can't see ties, and the full public front has 2,642 tied A values: which tied
    design is kept decides refine_leaders' starts (stable sort) and fullres's controls
    (np.argsort's default). Any merged front helper must keep both."""
    import fullres
    import generators
    import refine_leaders
    t0 = time.time()
    recorded = json.loads((ROOT / "data" / "refine" / "starts.json").read_text())
    with tempfile.TemporaryDirectory() as tmp, \
            patched(refine_leaders, OUT=Path(tmp), STARTS=Path(tmp) / "starts.json"):
        quiet(refine_leaders.pick, len(recorded))
        got = json.loads((Path(tmp) / "starts.json").read_text())
    compare(f"front: refine_leaders.pick({len(recorded)}) vs data/refine/starts.json", got, recorded,
            t0, ignore=())

    t0 = time.time()
    rows, A, L, _ = refine_leaders.public_front()
    on_front = {rows[i]["id"] for i in np.flatnonzero(fullres.front_mask(A, L))}
    controls = [d["id"] for d in read_jsonl(ROOT / "data" / "fullres" / "pilot_designs.jsonl")
                if d["method"] == "control"]
    off = [c for c in controls if c not in on_front]
    report(not off, f"front: {len(controls) - len(off)}/{len(controls)} of fullres's controls lie on "
                    f"front_mask's front of all public designs" + (f"; off it: {off}" if off else ""), t0)

    t0 = time.time()
    sA, sL = np.array([s["A"] for s in recorded]), np.array([s["L"] for s in recorded])
    report(np.array_equal(generators.front_height(sA, A, L), sL),
           f"front: generators.front_height at the {len(recorded)} starts equals their L_gradB", t0)


def check_table() -> None:
    import training_data
    t0 = time.time()
    disk = disk_table()
    with patched(np, savez_compressed=lambda *a, **k: None):     # build() would rewrite train.npz
        fresh = quiet(training_data.build)

    def same(k):
        f, d = fresh[k], disk[k]
        if f.dtype.kind != d.dtype.kind or (f.dtype.kind != "U" and f.dtype != d.dtype):
            return False                                          # id width may change; nothing else
        return np.array_equal(f, d, equal_nan=f.dtype.kind == "f")

    differ = [k for k in sorted(fresh.keys() | disk.keys()) if k not in fresh or k not in disk or not same(k)]
    report(not differ, f"table: rebuilt training table ({len(fresh.get('X', ()))} rows) "
           + ("equals train.npz, dtypes included" if not differ else
              f"differs from train.npz ({len(disk['X'])} rows) in {differ}; if simulations were "
              "added since the last build, run training_data.py and re-check"), t0)


def check_proposers() -> None:
    import loop
    import modes5_control
    import perturb
    import screen
    import surface
    for module, name, n, seed in [(perturb, "pilot.jsonl", 2400, 0), (surface, "surface.jsonl", 960, 1)]:
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / name
            run_script(module, [f"{module.__name__}.py", str(out), "--n", str(n), "--seed", str(seed)])
            compare_file(f"proposers: {module.__name__}.py --n {n} --seed {seed} vs {name}", out,
                         SEEDED / name, t0)

    t0 = time.time()                    # the generators' path: loop.propose -> generators.propose
    arm = "g60_gmm_k16_n16_s30"
    recorded = read_jsonl(SEEDED / f"loop_{arm}_r0.jsonl")
    t = loop.arm_view(screen.load_table(), REPLAY)              # what loop.pick hands the proposer
    mine = loop.ours_status(t)["novel_pass"]
    ours = {k: t[k][mine] for k in ("X", "A", "L", "id")}
    pool = {d["id"]: d for d in quiet(loop.propose, "gen:gmm_k16_n16", 8000, 30 * 1000 + 0, ours)}
    absent = [d["id"] for d in recorded if d["id"] not in pool]
    label = f"proposers: gen:gmm_k16_n16 pool (8000, seed 30) vs {arm} round 0"
    if absent:
        report(False, f"{label}: {len(absent)}/{len(recorded)} recorded picks not in the pool", t0)
    else:
        compare(label, [pool[d["id"]] for d in recorded], recorded, t0,
                ignore=("A_geom", "arm", "round", "pred_pass", "pred_L", "pred_crash"))  # added by pick

    t0 = time.time()
    compare("proposers: modes5_control.designs() vs data/track_a/modes5_control_designs.jsonl",
            quiet(modes5_control.designs), read_jsonl(TRACK_A / "modes5_control_designs.jsonl"), t0,
            ignore=())


def check_loop() -> None:
    import families
    import loop
    import training_data
    pub = families.load()
    with patched(training_data, build=lambda *a, **k: dict(disk_table())):   # read train.npz as is
        for arm, kw in LOOP_ARMS:
            t0 = time.time()
            picks = quiet(loop.pick, REPLAY, "blend", batch=60, pool=20000, r=0, pub=pub, **kw)
            compare(f"loop: {arm} round 0", picks, read_jsonl(SEEDED / f"loop_{arm}_r0.jsonl"), t0)


def check_logs() -> None:
    import families
    import loop
    import screen
    import track_a_loop as ta
    t0 = time.time()
    table, pub = screen.load_table(), families.load()
    logs, bad, n = sorted((SEEDED / "logs").glob("loop_*.jsonl")), [], 0
    for log in logs:
        arm = log.stem[len("loop_"):]
        view = loop.arm_view(table, arm)
        picks = {int(m.group(1)): {d["id"] for d in read_jsonl(f)}
                 for f in SEEDED.glob(f"loop_{arm}_r*.jsonl")
                 if (m := re.fullmatch(rf"loop_{re.escape(arm)}_r(\d+)", f.stem))}
        for e in read_jsonl(log):
            r = e["round"]
            hide = set().union(*(ids for k, ids in picks.items() if k > r))
            if (arm, r) in LATE:
                hide |= {d["id"] for d in read_jsonl(LATE[arm, r])}
            keep = ~np.isin(view["id"].astype(str), list(hide))
            t = {k: v[keep] for k, v in view.items()}
            mine = loop.ours_status(t)["novel_pass"]
            got = {"novel_passing_total": int(mine.sum()),
                   "hv_ours_alone": loop.hypervolume(t["A"][mine], t["L"][mine]),
                   "hv_public_plus_ours": loop.hypervolume(np.r_[pub["A"], t["A"][mine]],
                                                           np.r_[pub["L"], t["L"][mine]])}
            n += 1
            if max(change(got[k], e[k]) for k in got) > FLOAT_TOL:
                bad.append(f"{arm} r{r}")
    report(not bad, f"logs: {n - len(bad)}/{n} logged rounds of {len(logs)} loop.py arms recomputed"
           + (f"; differ: {bad}" if bad else ""), t0)

    t0 = time.time()
    leif = ta.leif_designs()
    logs, bad, n = sorted((TRACK_A / "logs").glob("ta_*.jsonl")), [], 0
    for log in logs:
        for e in read_jsonl(log):
            with only_rounds(ta, e["round"]):
                st = ta.state(log.stem, leif)
            mine = st["novel_pass"]
            got = {"novel_passing_total": int(mine.sum()),
                   "hv_ours_alone": loop.hypervolume(st["A"][mine], st["L"][mine]),
                   "hv_leif_plus_ours": loop.hypervolume(np.r_[leif["A"], st["A"][mine]],
                                                         np.r_[leif["L"], st["L"][mine]]),
                   "min_A_pass": float(np.min(st["A"][mine])) if mine.any() else None}
            n += 1
            if max(change(got[k], e[k]) for k in got) > FLOAT_TOL:
                bad.append(f"{log.stem} r{e['round']}")
    report(not bad, f"logs: {n - len(bad)}/{n} logged rounds of {len(logs)} track_a_loop.py arms "
                    "recomputed" + (f"; differ: {bad}" if bad else ""), t0)


def check_track_a() -> None:
    import track_a_loop as ta
    leif = ta.leif_designs()
    for arm, screen, r in TRACK_A_REPLAYS:
        t0 = time.time()
        with only_rounds(ta, r - 1):
            picks = quiet(ta.pick, arm, screen, batch=40, pool=20000, seed=50, r=r, layers=3,
                          window=4.0, leif=leif)
        compare(f"track_a: {arm} round {r}", picks, read_jsonl(TRACK_A / "picks" / f"{arm}_r{r}.jsonl"),
                t0, ignore=())


def check_report() -> None:
    """report_numbers.py reads every ledger, so later runs change its totals. Pin it to the
    ledger groups the committed totals list (PR 4 moves this pin into the script)."""
    import report_numbers as rn
    t0 = time.time()
    gold = ROOT / "results" / "seeded"
    want_json = json.loads((gold / "track_b_totals.json").read_text())
    want_csv = (gold / "track_b_feasible_designs.csv").read_text()
    groups = set(want_json["simulations"]["by_ledger"])

    def group(f: Path) -> str:                        # report_numbers.simulations()'s grouping
        return "gen_pilot" if f.parent.parent.name == "gen" else re.sub(r"_s\d+$", "", f.stem)

    ledgers = [f for f in rn.LEDGERS if group(f) in groups]
    ids = list({d["id"] for f in ledgers for d in read_jsonl(f)})
    load_table = rn.load_table

    def pinned() -> dict:
        t = load_table()
        keep = (t["source"] != "ours") | np.isin(t["id"].astype(str), ids)
        return {k: v[keep] for k, v in t.items()}

    with tempfile.TemporaryDirectory() as tmp, patched(rn, LEDGERS=ledgers, load_table=pinned, OUT=Path(tmp)):
        quiet(rn.main)
        got_json = json.loads((Path(tmp) / "track_b_totals.json").read_text())
        got_csv = (Path(tmp) / "track_b_feasible_designs.csv").read_text()
    moved = sorted(p for p, c in leaf_changes(got_json, want_json) if c > FLOAT_TOL)
    report(not moved and got_csv == want_csv,
           f"report: report_numbers.py on the {len(ledgers)} ledgers it reported"
           + ("" if not moved else f"; track_b_totals.json differs at {moved[:8]}")
           + (" reproduces both files" if not moved and got_csv == want_csv else
              "" if got_csv == want_csv else "; track_b_feasible_designs.csv differs"), t0)


def check_fullres() -> None:
    import fullres
    t0 = time.time()
    recorded = ROOT / "data" / "fullres" / "pilot_designs.jsonl"
    with tempfile.TemporaryDirectory() as tmp, \
            patched(fullres, OUT=Path(tmp), DESIGNS=Path(tmp) / recorded.name):
        quiet(fullres.propose, 60, 8, 40, 3, 2.0, 20000)                 # seed 40: blends + controls
        quiet(fullres.propose, 33, 8, 41, 5, 2.0, 20000, append=True)    # seed 41: more blends
        compare_file("fullres: propose seeds 40 and 41 vs data/fullres/pilot_designs.jsonl",
                     Path(tmp) / recorded.name, recorded, t0)


CHECKS = {"imports": check_imports, "hv": check_hv, "front": check_front, "table": check_table,
          "proposers": check_proposers, "loop": check_loop, "logs": check_logs,
          "track_a": check_track_a, "report": check_report, "fullres": check_fullres}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checks", nargs="*",
                    help=f"any of {', '.join(CHECKS)} (default: all but {', '.join(sorted(OPTIONAL))})")
    chosen = ap.parse_args().checks or [c for c in CHECKS if c not in OPTIONAL]
    if set(chosen) - set(CHECKS):
        ap.error(f"unknown check(s) {sorted(set(chosen) - set(CHECKS))}; choose from {', '.join(CHECKS)}")
    t0, before = time.time(), snapshot()
    with refuse_writes():
        for name in chosen:
            t_check = time.time()
            try:
                CHECKS[name]()
            except Exception as e:                # noqa: BLE001 - one broken check mustn't hide the rest
                traceback.print_exc()
                report(False, f"{name}: crashed: {type(e).__name__}: {e}", t_check)
    after = snapshot()
    changed = sorted(str(p.relative_to(ROOT)) for p in before.keys() | after.keys()
                     if before.get(p) != after.get(p))
    report(not changed, "no data or results file changed" if not changed else
           f"files changed during the check: {changed[:10]}", t0)
    bad = sum(not ok for ok, _, _ in results)
    print(f"\n{len(results) - bad}/{len(results)} passed in {time.time() - t0:.0f} s")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
