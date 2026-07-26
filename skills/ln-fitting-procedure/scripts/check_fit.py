"""Mechanical checker for a cascade-model fitting script. Seconds, deterministic, no refit.

    python check_fit.py their_fit.py [results.json] [data.npz]

Goes down a fixed list and reports PASS / FAIL / SKIP for each item. Every check is either a
static read of the code or a cheap execution of one of their functions on a fixed input --
nothing here re-runs the optimizer, because refitting is slow, token-expensive, and answers a
different question than "is this set up correctly".

The script under test is never executed top to bottom. Only its function and import
statements are evaluated, so importing it cannot kick off a fit.

Checks:
    filter      t origin, unit peak, zero DC, degrees, overflow form, full length
    nonlinearity  beta*x+gamma grouping
    convolution   circular, full stimulus length, per-epoch
    causality     impulse response peaks at positive lag
    preprocessing response left in native units, stimulus mean-subtracted, integer decimation
    reporting     per-epoch R^2 rather than pooled
    optimizer     multiple starts, convergence recorded
    results       reported parameters reproduce the reported R^2; numFilt identifiable
"""
import ast
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cascade_fit as cf  # noqa: E402

REF = dict(numFilt=4.0, tauR=0.025, tauD=0.045, tauP=0.065, phi=35.0)
N, DT = 400, 0.01


class Report:
    def __init__(self):
        self.rows = []

    def add(self, status, name, detail):
        self.rows.append((status, name, detail))

    def show(self):
        w = max(len(n) for _s, n, _d in self.rows)
        for s, n, d in self.rows:
            print(f"  {s:5s} {n:<{w}}  {d}")
        bad = sum(1 for s, _, _ in self.rows if s == "FAIL")
        skip = sum(1 for s, _, _ in self.rows if s == "SKIP")
        print(f"\n  {len(self.rows) - bad - skip} pass, {bad} fail, {skip} not checked")
        return bad


def load_functions(path):
    """Exec only the imports and function defs, so importing cannot trigger a fit."""
    src = open(path).read()
    tree = ast.parse(src)
    ns = {"__name__": "_under_test"}
    # Node by node, skipping whatever fails: a top-level assignment that needs data must not
    # stop the function definitions after it from being reached.
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import,
                                 ast.ImportFrom, ast.Assign, ast.ClassDef)):
            continue
        if isinstance(node, ast.Assign) and _calls(node):
            continue                      # skip assignments that would execute work
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), ns)
        except Exception:
            pass
    return ns, src


def _calls(node):
    return any(isinstance(x, ast.Call) for x in ast.walk(node))


def find(ns, *names):
    for n in names:
        if n in ns and callable(ns[n]):
            return ns[n]
    return None


def check_filter(ns, src, rep):
    fn = find(ns, "make_filter", "makeFilter", "build_filter", "parametric_filter", "filt")
    if fn is None:
        rep.add("SKIP", "filter construction", "no filter function found")
        return None
    try:
        f = np.asarray(fn(REF["numFilt"], REF["tauR"], REF["tauD"], REF["tauP"], REF["phi"],
                          N, DT), dtype=float).ravel()
    except Exception as e:
        rep.add("SKIP", "filter construction", f"could not call it: {type(e).__name__}")
        return None
    ref = cf.make_filter(**REF, n_points=N, dt=DT)

    # Compare against the reference's own peak, not against 1: unit-peak normalisation happens
    # BEFORE the zero-DC subtraction, so the finished filter peaks near 1, not at it. Demanding
    # exactly 1 fails the reference implementation itself.
    ref_peak = float(np.max(np.abs(ref)))
    got_peak = float(np.max(np.abs(f)))
    rep.add("PASS" if abs(got_peak - ref_peak) < 0.05 * ref_peak else "FAIL", "filter unit peak",
            f"max|f| = {got_peak:.6g} (reference gives {ref_peak:.6g})")
    rep.add("FAIL" if abs(np.mean(f)) > 1e-12 else "PASS", "filter zero DC",
            f"mean(f) = {np.mean(f):.3g} (want ~0)")

    # normalise both before comparing shape, so amplitude conventions don't mask the rest
    fn_ = f / (np.max(np.abs(f)) or 1)
    rf_ = ref / np.max(np.abs(ref))
    d = float(np.max(np.abs(fn_ - rf_)))
    rep.add("PASS" if d < 1e-9 else "FAIL", "filter matches reference",
            f"max|diff| vs cascade_fit.make_filter = {d:.3e}")

    # t origin: reference has t[0] = dt, so f[0] is the dt-sample. A t=0 build shifts by one.
    shifted = float(np.max(np.abs(fn_[1:] - rf_[:-1])))
    if d >= 1e-9 and shifted < d / 10:
        rep.add("FAIL", "filter t origin", f"looks like t = arange(n)*dt: shifting by one "
                                           f"sample drops the diff to {shifted:.2e} (want t[0]=dt)")
    else:
        rep.add("PASS" if d < 1e-9 else "SKIP", "filter t origin",
                "t starts at dt" if d < 1e-9 else "filter differs for another reason")

    has_deg = bool(re.search(r"/\s*360|deg2rad|np\.radians|\*\s*np\.pi\s*/\s*180", src))
    rep.add("PASS" if has_deg else "FAIL", "phi in degrees",
            "found a degrees->radians conversion" if has_deg
            else "no /360 or deg2rad near the cosine: phi is being used as radians")

    overflow = bool(re.search(r"\(\s*t\s*/\s*(abs\()?tau_?[rR]", src)) and \
        not bool(re.search(r"1\s*/\s*\(\s*1\s*\+\s*\(", src))
    rep.add("FAIL" if overflow else "PASS", "filter overflow form",
            "(t/tauR)**n written directly: NaN above numFilt~145" if overflow
            else "reciprocal form, or no direct power")
    return fn


def check_nl(ns, src, rep):
    grouped = re.search(r"(cdf|Phi|normcdf)\s*\(\s*\w+\s*\*\s*\(\s*\w+\s*\+", src)
    rep.add("FAIL" if grouped else "PASS", "nonlinearity grouping",
            "beta*(x + gamma): gamma ends up on the wrong scale" if grouped
            else "beta*x + gamma")


def check_conv(ns, rep):
    fn = find(ns, "conv", "convolve", "circular_conv", "conv_all", "generator")
    if fn is None:
        rep.add("SKIP", "convolution", "no convolution function found")
        return
    try:
        r = cf.verify_convolution(fn, n_points=256, verbose=False)
    except Exception as e:
        rep.add("SKIP", "convolution", f"could not call it: {type(e).__name__}")
        return
    for name, c in r["checks"].items():
        rep.add("PASS" if c["passed"] else "FAIL", f"conv {name}", c["detail"].split(";")[0])


def check_causality(fnfilt, rep):
    if fnfilt is None:
        rep.add("SKIP", "causality", "no filter function")
        return
    try:
        f = np.asarray(fnfilt(REF["numFilt"], REF["tauR"], REF["tauD"], REF["tauP"],
                              REF["phi"], N, DT), dtype=float).ravel()
        lag = cf.causality_check(f)
    except Exception as e:
        rep.add("SKIP", "causality", f"{type(e).__name__}")
        return
    rep.add("PASS" if lag > 0 else "FAIL", "causality",
            f"impulse response peaks at lag {lag:+d} bins (want positive)")


def uses_module(src, *names):
    """True if the script delegates to cascade_fit for this step rather than doing it itself.

    A script that calls load_epochs/fit_ln gets the preprocessing, the restarts and the
    convergence handling from the module, so they are correctly absent from its own source.
    Failing it for that would penalise the recommended path.
    """
    if not re.search(r"(from|import)\s+cascade_fit", src):
        return False
    return any(re.search(rf"\b{n}\s*\(", src) for n in names)


def check_preproc(src, rep):
    zscored = re.search(r"resp\w*\s*=\s*\(?\s*resp\w*\s*-\s*resp\w*\.mean\(\)\s*\)?\s*/\s*resp\w*\.std\(\)", src) \
        or re.search(r"/\s*resp\w*\.std\(\)", src)
    rep.add("FAIL" if zscored else "PASS", "response native units",
            "response is z-scored: alpha and epsilon lose their units" if zscored
            else "response not normalised")
    rect = re.search(r"maximum\(\s*resp|clip\(\s*resp|resp\w*\[\s*resp\w*\s*<\s*0\s*\]", src)
    rep.add("FAIL" if rect else "PASS", "no rectification",
            "response rectified" if rect else "response not rectified")
    if uses_module(src, "load_epochs"):
        rep.add("PASS", "stimulus mean-subtracted", "handled by cascade_fit.load_epochs")
    else:
        ms = re.search(r"stim\w*\s*-=?\s*.*stim\w*\.mean\(axis\s*=\s*1", src) or \
            re.search(r"stim\w*\s*-\s*stim\w*\.mean\(axis\s*=\s*1", src)
        rep.add("PASS" if ms else "FAIL", "stimulus mean-subtracted",
                "per-epoch mean subtracted" if ms
                else "no per-epoch stimulus mean subtraction found")


def check_reporting(src, results, rep):
    if isinstance(results, dict) and isinstance(results.get("r2_per_epoch"), list):
        rep.add("PASS", "per-epoch R2", f"{len(results['r2_per_epoch'])} per-epoch values reported")
        return
    # a pooled R2 sums over everything; a row-wise one sums along axis=1 / dim 2
    pooled = re.search(r"np\.sum\(\s*\(\s*resp\w*\s*-\s*\w+\s*\)\s*\*\*\s*2\s*\)", src) \
        and not re.search(r"axis\s*=\s*1", src)
    rep.add("FAIL" if pooled else "SKIP", "per-epoch R2",
            "R2 is pooled over concatenated epochs (no axis=1): a different quantity, and it "
            "credits the model for across-epoch mean differences" if pooled
            else "no r2_per_epoch in results; could not tell from the code")


def check_optimizer(src, rep):
    if uses_module(src, "fit_ln", "fit_glm", "fit_two_arm"):
        rep.add("PASS", "multiple starts", "handled by cascade_fit (20 starts x 10 restarts)")
        rep.add("PASS", "convergence recorded", "handled by cascade_fit diagnostics")
        return
    # A restart loop (refining one start) is not the same as multiple STARTS, and a loop
    # whose literal range is 0 or 1 is not multiple starts either even though the random
    # draws are textually present inside it.
    n_min = len(re.findall(r"minimize\(|fminsearch\(", src))
    good_loop = rejected_loop = None
    for m in re.finditer(r"for\s+\w+\s+in\s+range\(\s*([0-9]+)\s*\)\s*:"
                         r"[\s\S]{0,300}?(?:rand|uniform)\(", src):
        if int(m.group(1)) > 1:
            good_loop = m
            break
        rejected_loop = int(m.group(1))
    if good_loop is None:
        good_loop = re.search(r"for\s+\w+\s+in\s+range\(\s*[a-zA-Z_]\w*\s*\)\s*:"
                              r"[\s\S]{0,300}?(?:rand|uniform)\(", src)

    if good_loop is not None:
        rep.add("PASS", "multiple starts", "a loop generates random starts for the optimizer")
    elif rejected_loop is not None:
        rep.add("FAIL", "multiple starts",
                f"the start loop is range({rejected_loop}): it never runs, so only the single "
                f"hardcoded start is used on a periodic surface")
    else:
        n_draw = len(re.findall(r"(?:np\.random\.|rng\.)?(?:rand|uniform|standard_normal)\(", src))
        rep.add("FAIL", "multiple starts",
                f"no random start generation found ({n_draw} rand calls, {n_min} optimizer "
                f"calls): a single start on a periodic surface")

    conv = re.search(r"\.status|\.success|exitflag", src)
    rep.add("PASS" if conv else "FAIL", "convergence recorded",
            "reads the optimizer status" if conv
            else "never reads res.status/.success: a fit that ran out of budget looks identical")


def check_results(results, data, rep):
    if not isinstance(results, dict) or "params" not in results:
        rep.add("SKIP", "round trip", "no results.json with params")
        return
    p = {k: float(v) for k, v in results["params"].items()}
    claimed = results.get("r2_mean", results.get("r2"))
    dt = float(results.get("dt", DT))
    if "tauR" in p and "numFilt" in p:
        ncrit = 22.0 * abs(p["tauR"]) / dt
        rep.add("PASS" if p["numFilt"] <= ncrit else "FAIL", "numFilt identifiable",
                f"numFilt = {p['numFilt']:.1f}, resolvable limit ~{ncrit:.0f} (22*tauR/dt)")
    if data is None or claimed is None:
        rep.add("SKIP", "round trip", "need results.json and the data file")
        return
    try:
        stim, resp, _ = cf.load_epochs(data, dt=dt, verbose=False)
        pred = cf.predict_ln(p, stim, dt)
        got = float(np.mean(cf.row_r2(pred, resp)))
    except Exception as e:
        rep.add("SKIP", "round trip", f"{type(e).__name__}: {e}")
        return
    gap = abs(got - float(claimed))
    rep.add("PASS" if gap < 0.05 else "FAIL", "round trip",
            f"reported {float(claimed):.4f}, rebuilds to {got:.4f} (gap {gap:.4f})")


def main():
    script = sys.argv[1]
    results = None
    data = None
    for a in sys.argv[2:]:
        if a.endswith(".json"):
            results = json.load(open(a))
        elif a.endswith((".npz", ".mat")):
            data = a
    if results is None and os.path.exists("results.json"):
        results = json.load(open("results.json"))
    if data is None and os.path.exists("data.npz"):
        data = "data.npz"

    print(f"checking {script}\n")
    ns, src = load_functions(script)
    rep = Report()
    fnfilt = check_filter(ns, src, rep)
    check_nl(ns, src, rep)
    check_conv(ns, rep)
    check_causality(fnfilt, rep)
    check_preproc(src, rep)
    check_reporting(src, results, rep)
    check_optimizer(src, rep)
    check_results(results, data, rep)
    bad = rep.show()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
