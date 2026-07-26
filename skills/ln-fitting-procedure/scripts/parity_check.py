"""Python <-> MATLAB cross-reference check for the CascadeGraph cascade model.

Run `parity_dump.m` in MATLAB first to write the reference .mat, then run this. It rebuilds
every quantity with cascade_fit.py and reports the maximum absolute difference against the
MATLAB values.

    matlab>  parity_dump('/path/to/cascadegraph', '/tmp/cg_reference.mat')
    shell$   python parity_check.py /tmp/cg_reference.mat

This exists because "the Python matches the MATLAB" is a claim with a short shelf life: it
stops being true the first time either side is edited, and nothing about a fitted R^2 reveals
that it has stopped being true. Run it after touching either implementation, and after any
MATLAB upgrade.

Exit code 0 = parity holds at the stated tolerance, 1 = it does not.
"""
import os
import sys

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cascade_fit as cf  # noqa: E402

TOL = 1e-10


def _cells(x):
    """MATLAB cell array -> list of 2-D arrays."""
    return [np.atleast_2d(c) for c in np.squeeze(x)]


def main(path):
    m = loadmat(path)
    dt = float(m["dt"].ravel()[0])
    stim = np.asarray(m["stim"], dtype=float)
    P = np.asarray(m["P"], dtype=float)
    nl_params = np.asarray(m["nl_params"], dtype=float)
    nl_x = np.asarray(m["nl_x"], dtype=float).ravel()
    filters, convs, preds, r2s, nls = (_cells(m[k]) for k in
                                       ("filters", "convs", "preds", "r2s", "nls"))
    n_pts = stim.shape[1]

    print(f"reference: {path}")
    print(f"  MATLAB {str(m['matlab_version'][0]).strip()}, dt={dt}, "
          f"stim {stim.shape}, {P.shape[0]} parameter sets\n")

    results = []
    expected = []          # divergences that are known, bounded and not failures
    for i, prow in enumerate(P):
        keys = dict(zip(cf.FILTER_KEYS, prow))
        f = cf.make_filter(*[keys[k] for k in cf.FILTER_KEYS], n_pts, dt)
        m_filt = np.asarray(filters[i]).ravel()

        # KNOWN, DELIBERATE divergence. CascadeGraph's ParamFilterNode forms (t/tauR)^numFilt
        # explicitly, which overflows to Inf (and Inf/(1+Inf)=NaN) above numFilt ~145. The
        # Python uses the algebraically identical reciprocal form and stays finite. We do NOT
        # patch the node: that region is unidentifiable anyway -- above ~22*tauR/dt the
        # discrete filter is bit-identical -- so the wall costs no R^2 (verified: the MATLAB
        # GLM fit reaches 0.8911 with the wall, the same as the Python without it). It only
        # produces a meaningless numFilt, which the fitters' diagnostics now flag.
        if not np.all(np.isfinite(m_filt)):
            n_crit = 22.0 * abs(keys["tauR"]) / dt
            expected.append(
                f"set {i+1}: MATLAB filter is NaN at numFilt={keys['numFilt']:g} "
                f"(overflow in the node); Python is finite. numFilt is unidentifiable above "
                f"~{n_crit:.0f} here, so this region carries no model information.")
            continue

        d_f = np.max(np.abs(f - m_filt))

        x = cf.circular_conv(stim, f)
        d_c = np.max(np.abs(x - np.asarray(convs[i])))

        a, b, g, e = nl_params[0]
        d_p = np.max(np.abs(cf.nl(x, a, b, g, e) - np.asarray(preds[i])))

        d_r = np.max(np.abs(cf.row_r2(np.asarray(preds[i]), stim)
                            - np.asarray(r2s[i]).ravel()))
        results += [(f"filter      set {i+1} (numFilt={prow[0]:g})", d_f),
                    (f"convolution set {i+1}", d_c),
                    (f"LN predict  set {i+1}", d_p),
                    (f"row-wise R2 set {i+1}", d_r)]

    # GLM free-running loop
    gp = dict(zip(cf.FILTER_KEYS, np.asarray(m["glmP"], dtype=float).ravel()))
    gnl = np.asarray(m["glmNL"], dtype=float).ravel()
    gfb = np.asarray(m["glmFb"], dtype=float).ravel()
    nfb = int(np.asarray(m["nFb"]).ravel()[0])
    gf = cf.make_filter(*[gp[k] for k in cf.FILTER_KEYS], n_pts, dt)
    gpred = cf.free_run(cf.circular_conv(stim, gf), gnl[0], gnl[1], gnl[2], gnl[3],
                        gfb[0], gfb[1], nfb, dt)
    results.append(("GLM free-running prediction",
                    float(np.max(np.abs(gpred - np.asarray(m["glmPred"]))))))

    # two-arm cascade
    fields = [str(x[0]) if isinstance(x, np.ndarray) else str(x)
              for x in np.asarray(m["twoArmFields"]).ravel()]
    vals = np.asarray(m["twoArmVals"], dtype=float).ravel()
    tap = dict(zip(fields, vals))
    tpred = cf.predict_two_arm(tap, stim, dt)
    results.append(("two-arm cascade prediction",
                    float(np.max(np.abs(tpred - np.asarray(m["twoArmPred"]))))))

    for j, q in enumerate(nl_params):
        d = np.max(np.abs(cf.nl(nl_x, *q) - np.asarray(nls[j]).ravel()))
        results.append((f"nonlinearity params {j+1}", d))

    # NaN must not be swallowed: max(0.0, nan) returns 0.0 in Python, so a NaN difference
    # would silently leave `worst` small and the verdict would read PARITY HOLDS.
    worst = 0.0
    n_bad = 0
    for name, d in results:
        ok = np.isfinite(d) and d <= TOL
        if not ok:
            n_bad += 1
        if np.isfinite(d):
            worst = max(worst, d)
        print(f"  {'OK  ' if ok else 'FAIL'}  {name:38s} max|diff| = {d:.3e}")

    for e in expected:
        print(f"  SKIP  {e}")
    print(f"\nworst FINITE difference: {worst:.3e}   (tolerance {TOL:.0e}); "
          f"{n_bad} of {len(results)} checks failed or were non-finite"
          + (f"; {len(expected)} known divergence(s) skipped" if expected else ""))
    if n_bad == 0:
        print("PARITY HOLDS — the Python and the MATLAB compute the same model.")
        return 0
    print("PARITY BROKEN — a convention has drifted between the two implementations.")
    print("Check, in order: t origin (dt vs 0), unit-peak and zero-DC normalization,")
    print("phi in degrees, beta*x+gamma grouping, circular vs padded convolution.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/cg_reference.mat"))
