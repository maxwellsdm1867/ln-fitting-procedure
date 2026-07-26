"""Score parameter recovery on the blind cells, modulo the exact degeneracies.

A correct fit may legitimately report the other sign branch, or an aliased tauP. Comparing
raw numbers would fail such a fit, so both truth and report are canonicalised the same way
before any comparison: de-alias tauP into the resolvable branch, then put alpha > 0.

Usage:  python tools/score_recovery.py <cell> <path/to/params.json-or-dict>
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import cascade_fit as cf  # noqa: E402

KEYS = list(cf.LN_KEYS)


def canon(p, dt):
    q = cf.canonical_sign(cf.dealias_tauP(cf.canonical_tauP_sign(cf.canonical_time_constants(dict(p))), dt))
    q["phi"] = (float(q["phi"]) + 180.0) % 360.0 - 180.0     # wrap to [-180, 180)
    return q


def truth_params(cell):
    t = json.load(open(os.path.join(ROOT, "truth", cell + ".json")))
    p = {"numFilt": t["filter"]["num_filt"], "tauR": t["filter"]["tau_r"],
         "tauD": t["filter"]["tau_d"], "tauP": t["filter"]["tau_p"], "phi": t["filter"]["phi"],
         "alpha": t["alpha"], "beta": t["beta"], "gamma": t["gamma"], "epsilon": t["epsilon"]}
    return p, t


def compare(cell, reported, dt=0.01):
    tp, t = truth_params(cell)
    a, b = canon(tp, t["dt"]), canon(reported, dt)
    rows = {}
    for k in KEYS:
        if k == "phi":                       # circular, in degrees
            d = abs((a[k] - b[k] + 180.0) % 360.0 - 180.0)
            rows[k] = {"truth": a[k], "fit": b[k], "abs_err_deg": round(float(d), 2),
                       "ok": bool(d <= 15.0)}
        else:
            rel = abs(b[k] - a[k]) / max(abs(a[k]), 1e-12)
            rows[k] = {"truth": round(float(a[k]), 6), "fit": round(float(b[k]), 6),
                       "rel_err": round(float(rel), 4), "ok": bool(rel <= 0.15)}
    return {"cell": cell, "ceiling_r2_mean": t["ceiling_r2_mean"],
            "per_parameter": rows,
            "n_recovered": int(sum(1 for v in rows.values() if v["ok"])),
            "n_params": len(KEYS)}


if __name__ == "__main__":
    cell = sys.argv[1]
    src = sys.argv[2]
    d = json.load(open(src))
    rep = d.get("params", d)
    print(json.dumps(compare(cell, rep, float(d.get("dt", 0.01))), indent=2))
