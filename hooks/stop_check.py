#!/usr/bin/env python3
"""Stop hook: if a fit was produced this turn, check it and draw the standard figures.

Auto-delegation to a reviewer agent turned out not to be forceable from a skill -- the model
reasons it already has the skill loaded and reviews inline. A hook is not a suggestion, so the
check runs whether or not anyone remembered to ask for it.

Deliberately cheap and quiet: it only fires when a results.json in the working directory was
written since the last time it looked, and it says nothing at all when the fit is clean.
"""
import glob
import hashlib
import json
import os
import subprocess
import sys

ROOT = os.environ.get("CLAUDE_PLUGIN_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "skills", "ln-fitting-procedure", "scripts")
STATE = os.path.join(os.path.expanduser("~"), ".claude", ".ln_fit_hook_seen")


def seen(path):
    """Have we already reported on this exact results file?"""
    try:
        h = hashlib.sha1(open(path, "rb").read()).hexdigest()[:16]
    except OSError:
        return True
    key = f"{os.path.abspath(path)}:{h}"
    old = set()
    if os.path.exists(STATE):
        old = set(open(STATE).read().split("\n"))
    if key in old:
        return True
    with open(STATE, "a") as fh:
        fh.write(key + "\n")
    return False


def main():
    results = None
    for cand in ("results.json", "fit_results.json", "ln_fit.json"):
        if os.path.exists(cand):
            results = cand
            break
    if results is None:
        return 0
    try:
        d = json.load(open(results))
    except Exception:
        return 0
    if "params" not in d:                       # not one of ours
        return 0
    if seen(results):
        return 0

    data = next((p for p in ("data.npz", "cell.npz") if os.path.exists(p)), None)
    scripts = [p for p in glob.glob("*.py")
               if p not in ("make_figures.py", "check_fit.py") and "fit" in p.lower()]

    out = []
    if scripts:
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "check_fit.py"),
                            scripts[0], results] + ([data] if data else []),
                           capture_output=True, text=True, timeout=120)
        fails = [l.strip() for l in r.stdout.splitlines() if l.strip().startswith("FAIL")]
        if fails:
            out.append(f"check_fit.py on {scripts[0]} — {len(fails)} problem(s):")
            out += ["  " + f for f in fails]

    if data:
        fig = "fit_figures.png"
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "make_figures.py"),
                            results, data, fig], capture_output=True, text=True, timeout=180)
        if r.returncode == 0:
            out.append(f"standard figures written to {fig} "
                       f"(input, output+prediction, filter vs cross-correlation, "
                       f"nonlinearity, residuals, per-epoch R²)")

    if out:
        print("[ln-fitting-procedure] " + "\n".join(out), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
