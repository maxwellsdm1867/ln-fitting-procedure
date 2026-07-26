"""Deterministic scorer for ln-fitting-procedure eval runs.

Reads a run's results.json, rebuilds the model from the *reported* parameters, and
recomputes per-epoch R^2 against the real (untouched) response. This catches inflated
self-reported numbers and tells you whether the reported parameters are expressed in
CascadeGraph's conventions.

Usage:
    python score.py <dataset-name> <path/to/results.json>

    dataset-name in {off_parasol_ln, glm_feedback, on_parasol_broken}
"""
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import norm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DT_RAW = 1e-4


# ------------------------------------------------------------------ model pieces
def build_filter(p, n_points, dt, normalize=True, degrees=True, t_starts_at_dt=True):
    # CascadeGraph uses t = (1:n)*dt. A fit written with t = arange(n)*dt is a different
    # (shifted) parameterization, so both are tried before declaring a mismatch.
    t = np.arange(1, n_points + 1) * dt if t_starts_at_dt else np.arange(n_points) * dt
    t = np.where(t == 0, 1e-300, t)
    phase = 2 * np.pi * p["phi"] / 360.0 if degrees else p["phi"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # rise/(1+rise) with rise=(t/tauR)^n, written as 1/(1+(tauR/t)^n) so that a large
        # fitted numFilt (which several fits do produce) overflows to 0 rather than to NaN.
        rise_frac = 1.0 / (1.0 + (abs(p["tauR"]) / t) ** p["numFilt"])
        f = rise_frac * np.exp(-t / abs(p["tauD"])) * np.cos(2 * np.pi * t / p["tauP"] + phase)
    if not np.all(np.isfinite(f)):
        return None
    if normalize:
        m = np.max(np.abs(f))
        if m == 0:
            return None
        f = f / m
        f = f - np.mean(f)
    return f


def conv(stim, f):
    return np.real(np.fft.ifft(np.fft.fft(stim, axis=1) * np.fft.fft(f)[None, :], axis=1))


def nl(x, p, grouped=False):
    z = p["beta"] * (x + p["gamma"]) if grouped else p["beta"] * x + p["gamma"]
    return p["alpha"] * norm.cdf(z) + p["epsilon"]


def free_run(filtered, p, n_fb, grouped=False):
    dt = p["_dt"]
    h = p["a_fb"] * np.exp(-np.arange(1, n_fb + 1) * dt / p["tau_fb"])
    n_ep, T = filtered.shape
    out = np.zeros((n_ep, T))
    for e in range(n_ep):
        hist = np.zeros(n_fb)
        for t in range(T):
            drive = filtered[e, t] + float(np.dot(h, hist))
            z = p["beta"] * (drive + p["gamma"]) if grouped else p["beta"] * drive + p["gamma"]
            y = p["alpha"] * norm.cdf(z) + p["epsilon"]
            hist[1:] = hist[:-1]
            hist[0] = y
            out[e, t] = y
    return out


def row_r2(pred, meas):
    sse = np.sum((meas - pred) ** 2, axis=1)
    sst = np.sum((meas - meas.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - sse / sst


# ------------------------------------------------------------------ data
def load(dataset, dt):
    d = np.load(os.path.join(ROOT, "data", dataset, "data.npz"))
    stim, resp = d["stim"].astype(float), d["resp"].astype(float)
    n_ep, n_raw = stim.shape
    factor = int(round(dt / DT_RAW))
    n_bins = n_raw // factor
    s = stim[:, : n_bins * factor].reshape(n_ep, n_bins, factor).mean(axis=2)
    r = resp[:, : n_bins * factor].reshape(n_ep, n_bins, factor).mean(axis=2)
    return s - s.mean(axis=1, keepdims=True), r


def truth(dataset):
    with open(os.path.join(ROOT, "truth", dataset + ".json")) as fh:
        return json.load(fh)


# ------------------------------------------------------------------ scoring
FILTER_KEYS = ["numFilt", "tauR", "tauD", "tauP", "phi"]
NL_KEYS = ["alpha", "beta", "gamma", "epsilon"]


def normalize_params(raw):
    """Accept common key aliases so a run isn't penalised for spelling."""
    alias = {
        "numfilt": "numFilt", "n": "numFilt", "num_filt": "numFilt", "filter_order": "numFilt",
        "taur": "tauR", "tau_r": "tauR", "taud": "tauD", "tau_d": "tauD",
        "taup": "tauP", "tau_p": "tauP", "phi_deg": "phi", "phase": "phi",
        "a": "alpha", "b": "beta", "c": "gamma", "d": "epsilon",
        "afb": "a_fb", "a_feedback": "a_fb", "taufb": "tau_fb", "tau_feedback": "tau_fb",
        "nfb": "n_fb_bins", "n_fb": "n_fb_bins",
    }
    out = {}
    for k, v in raw.items():
        kk = alias.get(k.lower(), k)
        out[kk] = v
    return out


def score_static(dataset, params, dt, glm=False):
    """Try the convention variants and report the best reconstruction."""
    stim, resp = load(dataset, dt)
    n_bins = stim.shape[1]
    tr = truth(dataset)
    best = None
    for normalize, degrees, grouped, t0 in itertools.product(
            [True, False], [True, False], [False, True], [True, False]):
        f = build_filter(params, n_bins, dt, normalize=normalize, degrees=degrees, t_starts_at_dt=t0)
        if f is None:
            continue
        x = conv(stim, f)
        if glm:
            p = dict(params)
            p["_dt"] = dt
            n_fb = int(params.get("n_fb_bins", 30))
            pred = free_run(x, p, n_fb, grouped=grouped)
        else:
            pred = nl(x, params, grouped=grouped)
        if not np.all(np.isfinite(pred)):
            continue
        r2 = row_r2(pred, resp)
        cand = {
            "r2_per_epoch": [round(float(v), 4) for v in r2],
            "r2_mean": round(float(r2.mean()), 4),
            "convention": {"filter_normalized": normalize, "phi_degrees": degrees,
                           "nl_beta_x_plus_gamma": not grouped, "t_starts_at_dt": t0},
        }
        if best is None or cand["r2_mean"] > best["r2_mean"]:
            best = cand
            best["_t0"] = t0
            best["_degrees"] = degrees

    # filter shape recovery, on the ground-truth time base, using the convention that won
    tf = np.array(tr["filter_10ms"]) if "filter_10ms" in tr else None
    filt_corr = None
    if tf is not None:
        t0 = best.get("_t0", True) if best else True
        deg = best.get("_degrees", True) if best else True
        fa = build_filter(params, len(tf), tr["dt"], normalize=True, degrees=deg, t_starts_at_dt=t0)
        if fa is not None and np.all(np.isfinite(fa)) and fa.std() > 0:
            filt_corr = round(float(abs(np.corrcoef(fa, tf)[0, 1])), 4)
    if best:
        best.pop("_t0", None)
        best.pop("_degrees", None)
        # When no variant reconstructs the fit, the winning "convention" is just the
        # least-bad wrong one and must not be read as evidence about the run's code.
        if best["r2_mean"] < 0.5:
            best["convention_reliable"] = False
            best["convention_note"] = (
                "reconstruction failed under every variant; convention flags are "
                "indeterminate — read the run's source instead")
        else:
            best["convention_reliable"] = True
    return best, filt_corr, tr


def main():
    dataset, results_path = sys.argv[1], sys.argv[2]
    with open(results_path) as fh:
        res = json.load(fh)

    dt = float(res.get("dt", 0.01))
    out = {"dataset": dataset, "dt_reported": dt}

    if dataset == "glm_feedback":
        blocks = {}
        for key, is_glm in [("ln", False), ("glm", True)]:
            blk = res.get(key, {})
            params = normalize_params(blk.get("params", {}))
            missing = [k for k in FILTER_KEYS + NL_KEYS if k not in params]
            if is_glm:
                missing += [k for k in ["a_fb", "tau_fb"] if k not in params]
            if missing:
                blocks[key] = {"error": f"missing params: {missing}"}
                continue
            best, fc, tr = score_static(dataset, params, dt, glm=is_glm)
            blocks[key] = {
                "self_reported_r2_mean": blk.get("r2_mean"),
                "independent": best,
                "filter_corr_with_truth": fc,
                "a_fb_reported": params.get("a_fb"),
                "tau_fb_reported": params.get("tau_fb"),
            }
        out["blocks"] = blocks
        tr = truth(dataset)
        out["ceiling_r2_mean"] = tr["ceiling_r2_mean"]
        out["best_static_nl_r2_vs_clean"] = tr["best_static_nl_r2_vs_clean"]
        out["truth_a_fb"] = tr["a_fb"]
        out["truth_tau_fb"] = tr["tau_fb"]
        # A 180-degree phase shift flips the filter, which flips alpha AND a_fb together, so
        # the sign of a_fb alone is not convention-invariant. The product a_fb*alpha is.
        gp = normalize_params(res.get("glm", {}).get("params", {}))
        if "a_fb" in gp and "alpha" in gp:
            prod = float(gp["a_fb"]) * float(gp["alpha"])
            truth_prod = tr["a_fb"] * tr["alpha"]
            out["a_fb_times_alpha"] = round(prod, 5)
            out["truth_a_fb_times_alpha"] = round(truth_prod, 5)
            out["feedback_sign_matches_truth"] = bool(np.sign(prod) == np.sign(truth_prod))
        if "error" not in blocks.get("ln", {}) and "error" not in blocks.get("glm", {}):
            g = blocks["glm"]["independent"]["r2_mean"]
            l = blocks["ln"]["independent"]["r2_mean"]
            out["glm_minus_ln_independent"] = round(g - l, 4)
    else:
        params = normalize_params(res.get("params", {}))
        missing = [k for k in FILTER_KEYS + NL_KEYS if k not in params]
        if missing:
            out["error"] = f"missing params: {missing}"
        else:
            best, fc, tr = score_static(dataset, params, dt, glm=False)
            out["self_reported_r2_mean"] = res.get("r2_mean")
            out["self_reported_r2_per_epoch"] = res.get("r2_per_epoch")
            out["independent"] = best
            out["filter_corr_with_truth"] = fc
            out["ceiling_r2_mean"] = tr["ceiling_r2_mean"]
            out["truth_params"] = {**tr["filter"], "alpha": tr["alpha"], "beta": tr["beta"],
                                   "gamma": tr["gamma"], "epsilon": tr["epsilon"]}
            sr = res.get("r2_mean")
            if isinstance(sr, (int, float)):
                out["self_report_gap"] = round(abs(float(sr) - best["r2_mean"]), 4)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
