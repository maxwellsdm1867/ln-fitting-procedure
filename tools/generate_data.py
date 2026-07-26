"""Generate synthetic ground-truth datasets for the ln-fitting-procedure skill evals.

Ground truth follows CascadeGraph exactly:
  ParamFilterNode.getFilterWithParams:
      f(t) = ((t/|tauR|)^n / (1 + (t/|tauR|)^n)) * exp(-t/tauD) * cos(2*pi*t/tauP + 2*pi*phi/360)
      f = f / max(|f|);  f = f - mean(f)
  ParamFilterNode.processTempParams:
      pred = real(ifft(fft(stim) .* fft(filter)))        % circular, per epoch
  SigmoidNlNode.processTempParams:
      out = alpha * normcdf(beta*x + gamma) + epsilon
  computeVarianceExplained:
      R^2 per row (epoch) = 1 - SSE/SST

The stimulus updates every 10 ms and is held constant within each 10 ms block, and
the response is computed at 10 ms and held the same way, so that *any* reasonable
decimation (block mean or stride subsample) recovers the same 10 ms series. That
keeps the benchmark from grading an arbitrary decimation choice.
"""
import json
import os

import numpy as np
from scipy.stats import norm

DT_RAW = 1e-4      # 0.1 ms
DECIM = 100        # -> 10 ms bins
DT = DT_RAW * DECIM

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
TRUTH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "truth")


def cg_filter(num_filt, tau_r, tau_d, tau_p, phi_deg, n_points, dt):
    t = (np.arange(1, n_points + 1) * dt)
    rise = (t / abs(tau_r)) ** num_filt
    f = (rise / (1.0 + rise)) * np.exp(-t / tau_d) * np.cos(2 * np.pi * t / tau_p + 2 * np.pi * phi_deg / 360.0)
    f = f / np.max(np.abs(f))
    f = f - np.mean(f)
    return f


def circ_conv(stim2d, filt):
    return np.real(np.fft.ifft(np.fft.fft(stim2d, axis=1) * np.fft.fft(filt)[None, :], axis=1))


def nl(x, alpha, beta, gamma, epsilon):
    return alpha * norm.cdf(beta * x + gamma) + epsilon


def free_run(filtered, alpha, beta, gamma, epsilon, a_fb, tau_fb, n_fb_bins, dt):
    """pred[t] = NL(filtered[t] + sum_k h[k] * pred[t-k]),  h[k] = a*exp(-k*dt/tau)."""
    h = a_fb * np.exp(-np.arange(1, n_fb_bins + 1) * dt / tau_fb)
    n_ep, T = filtered.shape
    pred = np.zeros((n_ep, T))
    for e in range(n_ep):
        hist = np.zeros(n_fb_bins)  # hist[0] = pred[t-1]
        for t in range(T):
            drive = filtered[e, t] + float(np.dot(h, hist))
            y = nl(drive, alpha, beta, gamma, epsilon)
            hist[1:] = hist[:-1]
            hist[0] = y
            pred[e, t] = y
    return pred


def row_r2(pred, meas):
    sse = np.sum((meas - pred) ** 2, axis=1)
    sst = np.sum((meas - meas.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - sse / sst


def upsample(x2d, factor):
    return np.repeat(x2d, factor, axis=1)


def make_stim(rng, n_ep, n_bins):
    s = rng.standard_normal((n_ep, n_bins))
    return s - s.mean(axis=1, keepdims=True)


def add_noise_for_target_r2(rng, clean, target_r2):
    """Additive Gaussian noise sized so the true model's per-epoch R^2 ~= target."""
    var_sig = np.var(clean, axis=1, keepdims=True)
    noise_sd = np.sqrt(var_sig * (1.0 / target_r2 - 1.0))
    return clean + rng.standard_normal(clean.shape) * noise_sd


def best_static_map_r2(x, y, n_bins=40):
    """R^2 of the best *static* nonlinearity of x predicting y (binned conditional mean).

    This upper-bounds what any LN model using this filter can achieve, so it measures
    how much of y is genuinely due to history/feedback rather than an instantaneous
    function of the filter output.
    """
    xf, yf = x.ravel(), y.ravel()
    edges = np.quantile(xf, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(xf, edges) - 1, 0, n_bins - 1)
    pred = np.zeros_like(yf)
    for b in range(n_bins):
        m = idx == b
        if m.any():
            pred[m] = yf[m].mean()
    return float(1.0 - np.sum((yf - pred) ** 2) / np.sum((yf - yf.mean()) ** 2))


def save(name, stim_raw, resp_raw, meta, truth):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TRUTH, exist_ok=True)
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(
        os.path.join(d, "data.npz"),
        stim=stim_raw.astype(np.float32),
        resp=resp_raw.astype(np.float32),
    )
    with open(os.path.join(d, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    with open(os.path.join(TRUTH, name + ".json"), "w") as fh:
        json.dump(truth, fh, indent=2)
    print(f"  wrote {d}/data.npz  stim{stim_raw.shape} resp{resp_raw.shape}")


# ---------------------------------------------------------------- dataset 1: OFF parasol LN
def dataset_off_parasol_ln():
    print("[1] off_parasol_ln")
    rng = np.random.default_rng(20260726)
    n_ep, n_bins = 3, 1000
    p = dict(num_filt=4, tau_r=0.025, tau_d=0.045, tau_p=0.065, phi=35.0)
    stim = make_stim(rng, n_ep, n_bins)
    f = cg_filter(p["num_filt"], p["tau_r"], p["tau_d"], p["tau_p"], p["phi"], n_bins, DT)
    x = circ_conv(stim, f)
    sx = x.std()
    alpha, beta, gamma, epsilon = -55.0, 1.8 / sx, -0.4, 18.0
    clean = nl(x, alpha, beta, gamma, epsilon)
    resp = add_noise_for_target_r2(rng, clean, 0.88)
    r2 = row_r2(clean, resp)
    print(f"    ceiling per-epoch R^2 = {np.round(r2, 3)}  (mean {r2.mean():.3f})")
    save(
        "off_parasol_ln",
        upsample(stim, DECIM), upsample(resp, DECIM),
        {
            "cell_type": "Off parasol",
            "protocol": "Variable Mean Noise, ConeResponseFull",
            "sample_interval_s": DT_RAW,
            "response_units": "mV",
            "stimulus_units": "R*/cone/s (mean-subtracted at analysis time)",
            "n_epochs": n_ep,
            "note": "stim and resp are (epochs x time) at 0.1 ms.",
        },
        {
            "filter": p, "alpha": alpha, "beta": beta, "gamma": gamma, "epsilon": epsilon,
            "dt": DT, "ceiling_r2_per_epoch": r2.tolist(), "ceiling_r2_mean": float(r2.mean()),
            "filter_10ms": f.tolist(),
        },
    )


# ---------------------------------------------------------------- dataset 2: GLM w/ feedback
def dataset_glm_feedback():
    print("[2] glm_feedback")
    rng = np.random.default_rng(981)
    n_ep, n_bins = 2, 500
    p = dict(num_filt=3, tau_r=0.03, tau_d=0.05, tau_p=0.08, phi=-20.0)
    stim = make_stim(rng, n_ep, n_bins)
    f = cg_filter(p["num_filt"], p["tau_r"], p["tau_d"], p["tau_p"], p["phi"], n_bins, DT)
    x = circ_conv(stim, f)
    sx = x.std()
    alpha, beta, gamma, epsilon = -70.0, 1.6 / sx, -0.3, 25.0
    n_fb, tau_fb = 30, 0.06
    # Pick the feedback amplitude so the loop stays stable but a *static* nonlinearity of
    # the filter output can no longer explain the response: that is the headroom the
    # feedback kernel has to earn. Loop gain ~ |a| * sum_k h_k * max|dNL/ddrive|.
    decay_sum = np.exp(-np.arange(1, n_fb + 1) * DT / tau_fb).sum()
    max_slope = abs(alpha) * beta * norm.pdf(0.0)
    a_unit = 1.0 / (decay_sum * max_slope)          # loop gain == 1 at this amplitude
    best = None
    for k in [1.0, 1.5, 1.8, 2.0, 2.2, 2.5]:
        a_try = -k * a_unit
        c = free_run(x, alpha, beta, gamma, epsilon, a_try, tau_fb, n_fb, DT)
        if not np.all(np.isfinite(c)):
            continue
        static_r2 = best_static_map_r2(x, c)
        print(f"    loop gain {k:.2f}: a_fb={a_try:.5f}  best-static-NL R^2 vs clean = {static_r2:.3f}")
        if best is None or abs(static_r2 - 0.70) < abs(best[2] - 0.70):
            best = (a_try, c, static_r2)
    a_fb, clean, static_r2 = best
    print(f"    chosen a_fb={a_fb:.5f}  (static-NL ceiling {static_r2:.3f} -> feedback carries the rest)")
    resp = add_noise_for_target_r2(rng, clean, 0.90)
    r2 = row_r2(clean, resp)
    print(f"    ceiling per-epoch R^2 = {np.round(r2, 3)}  (mean {r2.mean():.3f})")
    save(
        "glm_feedback",
        upsample(stim, DECIM), upsample(resp, DECIM),
        {
            "cell_type": "Off parasol",
            "protocol": "Variable Mean Noise, ConeResponseFull",
            "sample_interval_s": DT_RAW,
            "response_units": "mV",
            "n_epochs": n_ep,
            "note": "stim and resp are (epochs x time) at 0.1 ms.",
        },
        {
            "filter": p, "alpha": alpha, "beta": beta, "gamma": gamma, "epsilon": epsilon,
            "a_fb": float(a_fb), "tau_fb": tau_fb, "n_fb_bins": n_fb, "dt": DT,
            "filter_10ms": f.tolist(),
            "ceiling_r2_per_epoch": r2.tolist(), "ceiling_r2_mean": float(r2.mean()),
            "best_static_nl_r2_vs_clean": static_r2,
        },
    )


# ---------------------------------------------------------------- dataset 3: ON cell, broken script
def dataset_on_parasol_broken():
    print("[3] on_parasol_broken")
    rng = np.random.default_rng(4242)
    n_ep, n_bins = 3, 900
    p = dict(num_filt=5, tau_r=0.018, tau_d=0.035, tau_p=0.05, phi=120.0)
    stim = make_stim(rng, n_ep, n_bins)
    f = cg_filter(p["num_filt"], p["tau_r"], p["tau_d"], p["tau_p"], p["phi"], n_bins, DT)
    x = circ_conv(stim, f)
    sx = x.std()
    # INCREASING nonlinearity (ON cell): alpha > 0, and a steep/offset sigmoid
    alpha, beta, gamma, epsilon = 45.0, 2.6 / sx, 0.9, -62.0
    clean = nl(x, alpha, beta, gamma, epsilon)
    resp = add_noise_for_target_r2(rng, clean, 0.86)
    r2 = row_r2(clean, resp)
    print(f"    ceiling per-epoch R^2 = {np.round(r2, 3)}  (mean {r2.mean():.3f})")
    save(
        "on_parasol_broken",
        upsample(stim, DECIM), upsample(resp, DECIM),
        {
            "cell_type": "On parasol",
            "protocol": "Variable Mean Noise, ConeResponseFull",
            "sample_interval_s": DT_RAW,
            "response_units": "mV",
            "n_epochs": n_ep,
            "note": "stim and resp are (epochs x time) at 0.1 ms.",
        },
        {
            "filter": p, "alpha": alpha, "beta": beta, "gamma": gamma, "epsilon": epsilon,
            "dt": DT, "ceiling_r2_per_epoch": r2.tolist(), "ceiling_r2_mean": float(r2.mean()),
            "filter_10ms": f.tolist(),
            "planted_bugs": [
                "default init only, no random restarts",
                "phi used as radians instead of degrees",
                "response z-scored (normalized) before fitting",
                "rectify (clip at 0) applied although response_units are mV",
                "filter not normalized (missing f/max|f| and f-mean(f))",
                "beta*(x+gamma) instead of beta*x+gamma",
                "alpha init hardcoded negative (resp_min - resp_max) for an ON cell",
            ],
        },
    )


if __name__ == "__main__":
    dataset_off_parasol_ln()
    dataset_glm_feedback()
    dataset_on_parasol_broken()
    print("done.")
