#!/usr/bin/env python3
"""Stop hook: if a fit was produced this turn, check it and draw the standard figures.

Auto-delegation to a reviewer agent turned out not to be forceable from a skill -- the model
reasons it already has the skill loaded and reviews inline. A hook is not a suggestion, so the
check runs whether or not anyone remembered to ask.

Getting it *heard* takes care, because the Stop event discards both streams on exit 0:

    Exit code 0 - stdout/stderr not shown
    Exit code 2 - show stderr to model and continue conversation
    Other exit codes - show stderr to user only

So a real problem leaves as exit 2, which hands the model one more turn to address it, and
anything merely informational leaves as a systemMessage on stdout. The seen-cache is what
makes exit 2 safe: the second Stop for the same results file is a no-op, so the turn cannot
bounce. stop_hook_active is checked as well, belt and braces.

Finding the fit is its own problem. Fits do not land in the working directory as often as you
would hope -- out/, results/, analysis/<cell>/ are all normal -- so this walks a bounded depth
and reports on EVERY results file that is new since it last looked, which is what makes a batch
fit of six cells produce six checks instead of one. Whatever it cannot do it says out loud;
silence is reserved for "nothing here changed".
"""
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.environ.get("CLAUDE_PLUGIN_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "skills", "ln-fitting-procedure", "scripts")
STATE = os.path.join(os.path.expanduser("~"), ".claude", ".ln_fit_hook_seen")
KEEP = 500                      # state is a dedup cache, not a log; keep it bounded

RESULTS_NAMES = ("results.json", "fit_results.json", "ln_fit.json")
DATA_NAMES = ("data.npz", "cell.npz", "data.mat", "cell.mat")

# Copies of the bundled scripts sitting next to a fit are not the fit that just ran.
NOT_THE_FITTER = {"check_fit.py", "make_figures.py", "cascade_fit.py", "parity_check.py"}

# Directories that never contain a fit and can contain thousands of files.
SKIP_DIRS = {"__pycache__", "node_modules", "site-packages", "venv", "env",
             "build", "dist", "target", ".git", ".venv", ".loop_tx",
             ".ipynb_checkpoints", ".mypy_cache", ".pytest_cache"}

MAX_DEPTH = 3          # out/results.json and analysis/<cell>/results.json both reachable
MAX_FITS = 8           # a cap that is announced, never silent
TIME_BUDGET_S = 150    # the hook's own timeout is 200s; leave room to report


def fit_key(path):
    """Identify a results file by content, so rewriting it unchanged does not re-report."""
    try:
        h = hashlib.sha1(open(path, "rb").read()).hexdigest()[:16]
    except OSError:
        return None
    return f"{os.path.abspath(path)}:{h}"


def already_seen(key):
    if not os.path.exists(STATE):
        return False
    try:
        return key in open(STATE).read().split("\n")
    except OSError:
        return False


def remember(keys):
    """Record only once the work is done: a run that died to a timeout has to be retried."""
    if not keys:
        return
    old = []
    if os.path.exists(STATE):
        try:
            old = [l for l in open(STATE).read().split("\n") if l]
        except OSError:
            pass
    try:
        with open(STATE, "w") as fh:
            fh.write("\n".join((old + list(keys))[-KEEP:]) + "\n")
    except OSError:
        pass


def run(args, timeout):
    """(returncode, stdout, stderr), with returncode None if it never got to exit.

    A hook that raises reports nothing at all, which is the one outcome worth avoiding.
    """
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return None, "", f"timed out after {timeout}s"
    except Exception as e:                                  # noqa: BLE001 - never crash the turn
        return None, "", f"{type(e).__name__}: {e}"


def last_line(text, limit=200):
    lines = [l for l in (text or "").strip().splitlines() if l.strip()]
    return lines[-1].strip()[:limit] if lines else "no output"


def find_results(root="."):
    """Every candidate results file at or below root, newest first."""
    found = []
    root = os.path.abspath(root)
    for dirpath, dirnames, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth >= MAX_DEPTH:
            dirnames[:] = []
        else:
            dirnames[:] = [d for d in dirnames
                           if d not in SKIP_DIRS and not d.startswith(".")]
        for name in RESULTS_NAMES:
            if name in filenames:
                found.append(os.path.join(dirpath, name))
    try:
        found.sort(key=os.path.getmtime, reverse=True)
    except OSError:
        pass
    return found


def is_ours(path):
    """A results file with a params block. Anything else belongs to some other tool."""
    try:
        d = json.load(open(path))
    except Exception:                                       # noqa: BLE001
        return False
    return isinstance(d, dict) and "params" in d


def siblings(results):
    """Data file, fitting script and any MATLAB fitter for one fit.

    Looked for beside the results first, then at the working directory: writing results into
    out/ while the fitter sits at the top level is normal, so a strict same-directory rule
    would find the fit and then refuse to check it.
    """
    here = os.path.dirname(results) or "."
    dirs = [here]
    if os.path.abspath(here) != os.path.abspath("."):
        dirs.append(".")

    data = next((os.path.join(d, n) for d in dirs for n in DATA_NAMES
                 if os.path.exists(os.path.join(d, n))), None)

    scripts = []
    for d in dirs:
        for p in glob.glob(os.path.join(d, "*.py")):
            b = os.path.basename(p)
            if b not in NOT_THE_FITTER and "fit" in b.lower():
                scripts.append(p)
    # Newest first: with fit_ln.py and fit_glm.py both present, the one just run is the one
    # to check. Filesystem order picked arbitrarily between them.
    try:
        scripts = sorted(set(scripts), key=os.path.getmtime, reverse=True)
    except OSError:
        scripts = sorted(set(scripts))

    matlab = any("fit" in os.path.basename(p).lower()
                 for d in dirs for p in glob.glob(os.path.join(d, "*.m")))

    return data, (scripts[0] if scripts else None), matlab


def check_one(results, label):
    """(problems, notes) for a single fit. Never raises."""
    problems, notes = [], []
    here = os.path.dirname(results) or "."
    data, script, matlab = siblings(results)

    if script:
        rc, out, err = run([sys.executable, os.path.join(SCRIPTS, "check_fit.py"),
                            script, results] + ([data] if data else []), 120)
        fails = [l.strip() for l in out.splitlines() if l.strip().startswith("FAIL")]
        tally = next((l.strip() for l in out.splitlines() if " pass," in l), "")
        if fails:
            # Report the denominator too. "1 problem" out of nineteen checks reads very
            # differently from "1 problem" with no idea how much was looked at.
            problems.append(f"{label}check_fit.py on {os.path.basename(script)} — "
                            f"{tally or f'{len(fails)} failed'}:")
            problems += ["  " + f for f in fails]
        elif rc != 0:
            notes.append(f"{label}check_fit.py could not run on "
                         f"{os.path.basename(script)}: {last_line(err)}")
    elif matlab:
        notes.append(f"{label}fit looks MATLAB-side — check_fit.py reads Python source, so "
                     f"the script was not checked (run check_fit.py yourself on a Python port)")

    if data:
        stem = os.path.splitext(os.path.basename(results))[0]
        fig = os.path.join(here, "fit_figures.png" if stem == "results"
                           else f"{stem}_figures.png")
        rc, out, err = run([sys.executable, os.path.join(SCRIPTS, "make_figures.py"),
                            results, data, fig], 180)
        if rc == 0:
            notes.append(f"{label}standard figures written to {os.path.relpath(fig)} "
                         f"(input, output+prediction, filter vs cross-correlation, "
                         f"nonlinearity, residuals, per-epoch R²)")
        else:
            # Silence here is the worst option: the usual cause is a missing meta.json beside
            # the data, and the underlying error says exactly that.
            notes.append(f"{label}could not draw {os.path.basename(fig)}: {last_line(err)}")
    else:
        notes.append(f"{label}no data file beside the results "
                     f"({', '.join(DATA_NAMES)}) — skipped the standard figures")

    return problems, notes


def main():
    payload = {}
    if not sys.stdin.isatty():
        try:
            payload = json.load(sys.stdin) or {}
        except Exception:                                   # noqa: BLE001
            payload = {}
    if payload.get("stop_hook_active"):        # we already gave the model its extra turn
        return 0

    fresh = []
    for p in find_results():
        if not is_ours(p):
            continue
        k = fit_key(p)
        if k and not already_seen(k):
            fresh.append((p, k))
    if not fresh:
        return 0

    dropped = max(0, len(fresh) - MAX_FITS)
    fresh = fresh[:MAX_FITS]

    problems, notes, done = [], [], []
    started = time.monotonic()
    for i, (results, key) in enumerate(fresh):
        if time.monotonic() - started > TIME_BUDGET_S:
            notes.append(f"stopped after {i} of {len(fresh)} fits — out of time budget; "
                         f"the rest stay unreported and will be picked up next turn")
            break
        rel = os.path.relpath(results)
        label = "" if (len(fresh) == 1 and os.path.dirname(rel) == "") else f"[{rel}] "
        p, n = check_one(results, label)
        problems += p
        notes += n
        done.append(key)

    if dropped:
        notes.append(f"{dropped} further new fit(s) not reported this turn "
                     f"(cap is {MAX_FITS}); they stay unreported rather than silently dropped")

    remember(done)

    if problems:
        print("[ln-fitting-procedure] " + "\n".join(problems + notes), file=sys.stderr)
        return 2
    if notes:
        print(json.dumps({"systemMessage": "[ln-fitting-procedure] " + "\n".join(notes)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
