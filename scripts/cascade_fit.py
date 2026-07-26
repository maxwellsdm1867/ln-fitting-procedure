"""Reference implementation of the parametric cascade models and their staged fit.

This is the parameterization CascadeGraph uses. Import it rather than rewriting the filter
and nonlinearity: every reimplementation is a chance to drift on one of the conventions
(t origin, filter normalization, degrees vs radians, beta*x+gamma, circular convolution),
and parameters fitted under a drifted convention are not comparable to anything.

Typical use:

    from cascade_fit import load_epochs, fit_ln, fit_glm, row_r2, roundtrip

    stim, resp, info = load_epochs("data.npz")      # dt/units read from meta.json
    res = fit_ln(stim, resp, dt=0.01)
    print(res["params"], res["r2_per_epoch"], res["r2_mean"])
    assert roundtrip(res["params"], stim, resp, dt=0.01)["ok"]

Everything is plain numpy/scipy. numba is used for the GLM inner loop when available and
falls back to numpy otherwise.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

PENALTY = 1e12
FILTER_KEYS = ("numFilt", "tauR", "tauD", "tauP", "phi")
NL_KEYS = ("alpha", "beta", "gamma", "epsilon")
LN_KEYS = FILTER_KEYS + NL_KEYS
GLM_KEYS = LN_KEYS + ("a_fb", "tau_fb")

RANDOM_RANGES = {
    "numFilt": (1.0, 10.0),
    "tauR": (0.005, 0.1),
    "tauD": (0.005, 0.2),
    "tauP": (0.01, 0.1),
    "phi": (-180.0, 180.0),
    "scFact": (-500.0, 500.0),
}
DEFAULT_START = dict(numFilt=4.0, tauR=0.02, tauD=0.01, tauP=0.02, phi=1.0, scFact=-100.0)


# --------------------------------------------------------------------------- model pieces
def make_filter(numFilt, tauR, tauD, tauP, phi, n_points, dt):
    """CascadeGraph ParamFilterNode.getFilterWithParams. phi is in DEGREES."""
    t = np.arange(1, n_points + 1) * dt          # starts at dt, not 0
    if not np.isfinite([numFilt, tauR, tauD, tauP, phi]).all() or tauP == 0:
        return None
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        # 1/(1+(tauR/t)^n) == (t/tauR)^n / (1+(t/tauR)^n), but saturates instead of
        # overflowing when a fit drives numFilt into the hundreds.
        rise = 1.0 / (1.0 + (abs(tauR) / t) ** numFilt)
        f = rise * np.exp(-t / abs(tauD)) * np.cos(2 * np.pi * t / tauP + 2 * np.pi * phi / 360.0)
    if not np.all(np.isfinite(f)):
        return None
    peak = np.max(np.abs(f))
    if peak == 0:
        return None
    f = f / peak                                  # unit peak
    return f - np.mean(f)                         # zero DC


def circular_conv(stim, filt):
    """FFT circular convolution, per epoch, filter at full stimulus length."""
    return np.real(np.fft.ifft(np.fft.fft(stim, axis=1) * np.fft.fft(filt)[None, :], axis=1))


def nl(x, alpha, beta, gamma, epsilon):
    """alpha * Phi(beta*x + gamma) + epsilon  -- note beta*x + gamma, not beta*(x+gamma)."""
    return alpha * norm.cdf(beta * x + gamma) + epsilon


def row_r2(pred, meas):
    """CascadeGraph computeVarianceExplained: one R^2 per epoch, against that epoch's mean."""
    sse = np.sum((meas - pred) ** 2, axis=1)
    sst = np.sum((meas - meas.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - sse / sst


def predict_ln(params, stim, dt):
    f = make_filter(*[params[k] for k in FILTER_KEYS], stim.shape[1], dt)
    if f is None:
        return None
    return nl(circular_conv(stim, f), *[params[k] for k in NL_KEYS])


# --------------------------------------------------------------------------- data
def load_epochs(npz_path, dt=0.01, raw_dt=None, stim_key="stim", resp_key="resp",
                meta_path=None, verbose=True):
    """Load, decimate and preprocess — inferring the setup rather than making you restate it.

    `raw_dt` is read from the recording's metadata (`sample_interval_s`) when not given, so the
    sampling interval and decimation factor come from the file instead of from memory. The
    setup actually used is returned in `info` and printed once, because a decimation or
    orientation mistake is invisible downstream and expensive.

    Returns (stim, resp, info). Old two-value call sites still work via tuple unpacking of the
    first two elements only if they index; prefer the three-value form.
    """
    npz_path = str(npz_path)
    d = np.load(npz_path)
    if stim_key not in d or resp_key not in d:
        raise KeyError(f"{npz_path} has {list(d.keys())}, expected '{stim_key}' and '{resp_key}'")
    stim, resp = d[stim_key].astype(float), d[resp_key].astype(float)

    info = {"source": npz_path}
    meta = {}
    mp = meta_path or os.path.join(os.path.dirname(npz_path), "meta.json")
    if os.path.exists(mp):
        try:
            with open(mp) as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}
    info["meta"] = mp if meta else None

    if raw_dt is None:
        raw_dt = meta.get("sample_interval_s")
        if raw_dt is None:
            raise ValueError(
                f"raw sampling interval unknown: no 'sample_interval_s' in {mp}. "
                "Pass raw_dt= explicitly rather than guessing.")
        info["raw_dt_from_metadata"] = True
    else:
        info["raw_dt_from_metadata"] = False
    raw_dt = float(raw_dt)

    if stim.ndim != 2 or resp.ndim != 2:
        raise ValueError(f"expected (epochs x time) 2-D arrays, got {stim.shape} and {resp.shape}")
    if stim.shape != resp.shape:
        raise ValueError(f"stim {stim.shape} and resp {resp.shape} differ")
    if stim.shape[0] > stim.shape[1]:
        raise ValueError(
            f"array is {stim.shape}: more epochs than time points, which almost always means "
            "it is transposed. Expected (epochs x time).")

    factor_f = dt / raw_dt
    factor = int(round(factor_f))
    if abs(factor_f - factor) > 1e-6 or factor < 1:
        raise ValueError(f"dt={dt} is not an integer multiple of raw_dt={raw_dt} "
                         f"(factor {factor_f:.4f}); pick a dt the sampling supports.")
    n_ep, n_raw = stim.shape
    n_bins = n_raw // factor
    if n_bins < 50:
        raise ValueError(f"decimating to dt={dt} leaves only {n_bins} bins per epoch")
    dropped = n_raw - n_bins * factor
    if factor > 1:
        stim = stim[:, : n_bins * factor].reshape(n_ep, n_bins, factor).mean(axis=2)
        resp = resp[:, : n_bins * factor].reshape(n_ep, n_bins, factor).mean(axis=2)

    # Anything KNOWN and checked stays quiet. Anything UNKNOWN gets surfaced -- a silent
    # default is a guess wearing a fact's clothes, and it is invisible from here on.
    unresolved = []
    units = meta.get("response_units")
    KNOWN_RATE = {"spikes/s", "spikes/sec", "Hz", "sp/s"}
    KNOWN_ANALOG = {"mV", "pA", "nA", "uA", "V", "A"}
    if units in KNOWN_RATE:
        rectify = True
    elif units in KNOWN_ANALOG:
        rectify = False
    else:
        rectify = False
        unresolved.append(
            f"response_units is {units!r}, which is not a recognised rate ({sorted(KNOWN_RATE)}) "
            f"or analog ({sorted(KNOWN_ANALOG)}) unit. Assuming ANALOG, so rectify=False. If this "
            "is a firing rate, pass the units explicitly -- rectifying an analog trace, or "
            "failing to rectify a rate, changes which model you are fitting.")
    if not meta:
        unresolved.append(
            f"no metadata file at {mp}: cell type, protocol and units are unknown, so nothing "
            "here has been checked against the recording's own description.")

    info.update(raw_dt=raw_dt, dt=dt, decimation_factor=factor, n_epochs=n_ep,
                n_bins=n_bins, dropped_raw_samples=dropped, response_units=units,
                rectify=rectify, stimulus_mean_subtracted=True, response_untouched=True,
                unresolved=unresolved)

    if verbose:
        src = "metadata" if info["raw_dt_from_metadata"] else "caller"
        print(f"[cascade_fit] {n_ep} epochs x {n_bins} bins @ dt={dt*1e3:.0f} ms "
              f"(raw {raw_dt*1e6:.0f} us from {src}, decimation {factor}x"
              + (f", dropped {dropped} trailing samples" if dropped else "") + ")")
        print(f"[cascade_fit] response_units={units!r} -> rectify={rectify}; "
              "stimulus mean-subtracted per epoch, response left in native units")
        for u in unresolved:
            print(f"[cascade_fit] UNRESOLVED: {u}")

    return stim - stim.mean(axis=1, keepdims=True), resp, info


# --------------------------------------------------------------------------- optimizer
def _simplex(p0, rel=0.05, floor=1e-3):
    """Initial simplex with an absolute floor, so a coordinate starting at 0 still moves.

    SciPy's default perturbs by 5% relative and falls back to 2.5e-4 absolute at zero, which
    is orders of magnitude too fine for a parameter whose natural scale is O(1).
    """
    p0 = np.asarray(p0, dtype=float)
    n = len(p0)
    step = np.maximum(np.abs(p0) * rel, floor)
    sx = np.vstack([p0] + [p0 + np.eye(n)[i] * step[i] for i in range(n)])
    return sx


def nm_restarts(loss, p0, n_restarts=10, use_simplex=True, want_status=False):
    """A restart = call minimize again from the previous best; the simplex is rebuilt.

    With want_status=True also returns whether the LAST call converged rather than exhausting
    maxfev. scipy status 1 means "maximum function evaluations exceeded" -- the optimizer
    disclaiming its own answer -- and it is otherwise invisible.
    """
    p = np.asarray(p0, dtype=float)
    best_p, best_f = p.copy(), loss(p)
    status, success = None, None
    for _ in range(n_restarts):
        opts = {"xatol": 1e-4, "fatol": 1e-4, "maxfev": 200 * len(p)}
        if use_simplex:
            opts["initial_simplex"] = _simplex(best_p)
        res = minimize(loss, best_p, method="Nelder-Mead", options=opts)
        status, success = int(res.status), bool(res.success)
        if np.isfinite(res.fun) and res.fun < best_f:
            best_p, best_f = res.x.copy(), float(res.fun)
    if want_status:
        return best_p, best_f, {"status": status, "success": success}
    return best_p, best_f


def _sse(pred, resp):
    if pred is None or not np.all(np.isfinite(pred)):
        return PENALTY
    e = float(np.sum((resp - pred) ** 2))
    return e if np.isfinite(e) else PENALTY


# --------------------------------------------------------------------------- stages
def stage1_filter(stim, resp, dt, n_random_inits=20, n_restarts=10, rng=None):
    """Filter only, 6 params: prediction = scFact * conv(stim, filter)."""
    rng = np.random.default_rng(rng)
    n_points = stim.shape[1]

    def loss(p):
        f = make_filter(p[0], p[1], p[2], p[3], p[4], n_points, dt)
        if f is None:
            return PENALTY
        return _sse(p[5] * circular_conv(stim, f), resp)

    starts = [np.array([DEFAULT_START[k] for k in list(FILTER_KEYS) + ["scFact"]])]
    for _ in range(max(0, n_random_inits - 1)):
        starts.append(np.array([rng.uniform(*RANDOM_RANGES[k]) for k in list(FILTER_KEYS) + ["scFact"]]))

    best_p, best_f, all_losses, last = None, np.inf, [], None
    for s in starts:
        p, fv, st = nm_restarts(loss, s, n_restarts, want_status=True)
        all_losses.append(fv)
        if fv < best_f:
            best_p, best_f, last = p, fv, st
    return dict(zip(FILTER_KEYS, best_p[:5])), best_f, {"losses": all_losses, "status": last}


def binned_nl(x, y, n_bins=30):
    """Equal-N binning of the (filter output, response) relationship -- CascadeGraph sampleNl."""
    x, y = np.ravel(x), np.ravel(y)
    order = np.argsort(x)
    xb = np.array([c.mean() for c in np.array_split(x[order], n_bins)])
    yb = np.array([c.mean() for c in np.array_split(y[order], n_bins)])
    return xb, yb


def stage2_nl(x, resp, n_restarts=10, n_bins=30):
    """Nonlinearity only, 4 params, filter fixed. Prefit to the binned NL, then refine."""
    xf, yf = np.ravel(x), np.ravel(resp)
    s = np.sign(np.corrcoef(xf, yf)[0, 1]) or 1.0
    seed = np.array([
        s * (yf.max() - yf.min()),                       # alpha, sign follows the cell
        2.0 / (xf.std() or 1.0),                         # beta, spans ~+/-2 SD of the drive
        0.0,                                             # gamma, replaced below
        yf.max() if s < 0 else yf.min(),                 # epsilon
    ])
    seed[2] = -seed[1] * np.median(xf)                   # median: mean(x) is exactly 0

    xb, yb = binned_nl(xf, yf, n_bins)
    p_pre, _ = nm_restarts(lambda p: _sse(nl(xb, *p), yb), seed, n_restarts=5)
    p_full, f_full = nm_restarts(lambda p: _sse(nl(xf, *p), yf), p_pre, n_restarts)

    if abs(p_full[1]) < 1e-2 * abs(p_pre[1]):
        # beta collapsed: the sigmoid fell into its linear regime. Retry from the prefit.
        p_retry, f_retry = nm_restarts(lambda p: _sse(nl(xf, *p), yf), p_pre, n_restarts)
        if f_retry < f_full:
            p_full, f_full = p_retry, f_retry
    return dict(zip(NL_KEYS, p_full)), f_full


def stage3_joint(p_filt, p_nl, stim, resp, dt, n_restarts=10):
    n_points = stim.shape[1]

    def loss(p):
        f = make_filter(p[0], p[1], p[2], p[3], p[4], n_points, dt)
        if f is None:
            return PENALTY
        return _sse(nl(circular_conv(stim, f), p[5], p[6], p[7], p[8]), resp)

    p0 = np.array([p_filt[k] for k in FILTER_KEYS] + [p_nl[k] for k in NL_KEYS])
    p, fv = nm_restarts(loss, p0, n_restarts)
    return dict(zip(LN_KEYS, p)), fv


def fit_ln(stim, resp, dt, n_random_inits=20, n_restarts=10, rng=None, dealias=True,
           diagnose=True):
    """Full staged LN fit. Returns params, per-epoch R^2, mean R^2."""
    p_filt, _, s1 = stage1_filter(stim, resp, dt, n_random_inits, n_restarts, rng)
    f = make_filter(*[p_filt[k] for k in FILTER_KEYS], stim.shape[1], dt)
    x = circular_conv(stim, f)
    p_nl, _ = stage2_nl(x, resp, n_restarts)
    params, _ = stage3_joint(p_filt, p_nl, stim, resp, dt, n_restarts)
    if dealias:
        params = dealias_tauP(canonical_tauP_sign(canonical_time_constants(params)), dt)
    params = canonical_sign(params)
    pred = predict_ln(params, stim, dt)
    r2 = row_r2(pred, resp)
    out = {"params": params, "r2_per_epoch": [float(v) for v in r2], "r2_mean": float(r2.mean()),
           "dt": dt}
    if diagnose:
        n = stim.shape[1]

        def _loss(v):
            q = dict(zip(LN_KEYS, v))
            return _sse(predict_ln(q, stim, dt), resp)

        out["diagnostics"] = _diagnose(params, stim, resp, dt, s1, predict_ln, _loss, r2)
    return out


# --------------------------------------------------------------------------- GLM
def _free_run_numpy(filtered, alpha, beta, gamma, epsilon, a_fb, tau_fb, n_fb, dt):
    """Free-running loop without numba, using the exact O(1) recursion.

    The feedback sum is evaluated recursively rather than as a dot product, which is exact
    because the kernel is exponential. With rho = exp(-dt/tau) and S(t) = sum_k rho^k pred(t-k),

        S(t+1) = rho*(pred(t) + S(t)) - rho^(n_fb+1) * pred(t-n_fb)

    so each step is O(1) and nothing is reallocated -- no np.roll, no per-sample dot product.
    That is the whole cost of GLM fitting, since the loop runs once per objective evaluation.
    """
    tau_fb = max(tau_fb, dt)
    rho = np.exp(-dt / tau_fb)
    rho_k1 = rho ** (n_fb + 1)
    n_ep, T = filtered.shape
    out = np.empty((n_ep, T))
    c = 1.0 / np.sqrt(2.0)
    from math import erfc
    for e in range(n_ep):
        buf = np.zeros(n_fb)
        idx = 0
        S = 0.0
        fe = filtered[e]
        for t in range(T):
            y = alpha * (0.5 * erfc(-(beta * (fe[t] + a_fb * S) + gamma) * c)) + epsilon
            oldest = buf[idx]
            S = rho * (y + S) - rho_k1 * oldest
            buf[idx] = y
            idx += 1
            if idx >= n_fb:
                idx = 0
            out[e, t] = y
    return out


try:                                                       # numba is ~80x faster here
    from numba import njit

    @njit(cache=True)
    def _free_run_njit(filtered, alpha, beta, gamma, epsilon, a_fb, tau_fb, n_fb, dt):
        # Exact O(1) recursion for the exponential feedback sum (see _free_run_numpy), and
        # math.erfc rather than an Abramowitz & Stegun polynomial: numba compiles erfc fine,
        # and the approximation costs bit-parity with the MATLAB for nothing. Using A&S here
        # put the GLM prediction 1.4e-5 away from the reference implementation.
        rho = np.exp(-dt / tau_fb)
        rho_k1 = rho ** (n_fb + 1)
        n_ep, T = filtered.shape
        out = np.empty((n_ep, T))
        c = 1.0 / np.sqrt(2.0)
        for e in range(n_ep):
            buf = np.zeros(n_fb)
            idx = 0
            S = 0.0
            for t in range(T):
                y = alpha * (0.5 * math.erfc(-(beta * (filtered[e, t] + a_fb * S) + gamma) * c)) \
                    + epsilon
                oldest = buf[idx]
                S = rho * (y + S) - rho_k1 * oldest
                buf[idx] = y
                idx += 1
                if idx >= n_fb:
                    idx = 0
                out[e, t] = y
        return out

    _HAVE_NUMBA = True
except Exception:                                          # pragma: no cover
    _HAVE_NUMBA = False


def free_run(filtered, alpha, beta, gamma, epsilon, a_fb, tau_fb, n_fb, dt):
    """pred[t] = NL(filtered[t] + h . pred[t-1..t-n_fb]). Never uses the observed response."""
    tau_fb = max(tau_fb, dt)
    if _HAVE_NUMBA:
        return _free_run_njit(np.ascontiguousarray(filtered), alpha, beta, gamma, epsilon,
                              a_fb, tau_fb, n_fb, dt)
    return _free_run_numpy(filtered, alpha, beta, gamma, epsilon, a_fb, tau_fb, n_fb, dt)


def fit_glm(stim, resp, dt, n_fb_bins=30, n_random_inits=20, n_restarts=10, rng=None,
            diagnose=True):
    """Staged GLM fit: filter (LN stage 1) -> NL+feedback free-running -> joint."""
    p_filt, _, s1 = stage1_filter(stim, resp, dt, n_random_inits, n_restarts, rng)
    f = make_filter(*[p_filt[k] for k in FILTER_KEYS], stim.shape[1], dt)
    x = circular_conv(stim, f)
    p_nl, _ = stage2_nl(x, resp, n_restarts)

    r_scale = float(resp.max() - resp.min())
    fb_inits = [(0.0, 5 * dt), (-0.01 * r_scale, dt), (-0.05 * r_scale, 5 * dt),
                (0.05 * r_scale, 5 * dt), (-0.01 * r_scale, 20 * dt)]
    # rescale amplitudes if the drive is not on the response scale (see references/glm-feedback.md)
    scale = float(x.std()) / r_scale if r_scale > 0 else 1.0
    fb_inits = [(a * scale, tau) for a, tau in fb_inits]

    def loss2(p):
        return _sse(free_run(x, p[0], p[1], p[2], p[3], p[4], max(p[5], dt), n_fb_bins, dt), resp)

    best_p2, best_f2 = None, np.inf
    for a0, tau0 in fb_inits:
        p0 = np.array([p_nl[k] for k in NL_KEYS] + [a0, tau0])
        p, fv = nm_restarts(loss2, p0, n_restarts)
        if fv < best_f2:
            best_p2, best_f2 = p, fv

    n_points = stim.shape[1]

    def loss3(p):
        fl = make_filter(p[0], p[1], p[2], p[3], p[4], n_points, dt)
        if fl is None:
            return PENALTY
        xx = circular_conv(stim, fl)
        return _sse(free_run(xx, p[5], p[6], p[7], p[8], p[9], max(p[10], dt), n_fb_bins, dt), resp)

    p0 = np.array([p_filt[k] for k in FILTER_KEYS] + list(best_p2))
    p, _ = nm_restarts(loss3, p0, n_restarts)
    params = dict(zip(GLM_KEYS, p))
    params["n_fb_bins"] = n_fb_bins
    params = canonical_sign(dealias_tauP(canonical_tauP_sign(canonical_time_constants(params)), dt))
    pred = predict_glm(params, stim, dt)
    r2 = row_r2(pred, resp)
    out = {"params": params, "r2_per_epoch": [float(v) for v in r2], "r2_mean": float(r2.mean()),
           "dt": dt}
    if diagnose:
        def _loss(v):
            q = dict(zip(GLM_KEYS, v))
            q["n_fb_bins"] = n_fb_bins
            return _sse(predict_glm(q, stim, dt), resp)

        out["diagnostics"] = _diagnose(params, stim, resp, dt, s1, predict_glm, _loss, r2, glm=True)
    return out


def predict_glm(params, stim, dt):
    f = make_filter(*[params[k] for k in FILTER_KEYS], stim.shape[1], dt)
    if f is None:
        return None
    x = circular_conv(stim, f)
    return free_run(x, *[params[k] for k in NL_KEYS], params["a_fb"], params["tau_fb"],
                    int(params.get("n_fb_bins", 30)), dt)



# --------------------------------------------------------------------------- diagnostics
def _diagnose(params, stim, resp, dt, s1, predict, loss, r2, glm=False, keys=None):
    """Run every mechanical check and return a verdict, so no one has to remember them.

    The point is not to produce more numbers to read. It is that the conventions, the
    degeneracies and the optimizer's own status are handled or reported automatically, so
    attention goes to the fit and the data instead of to bookkeeping.
    """
    warn = []
    checks = {}

    st = (s1 or {}).get("status") or {}
    checks["converged"] = {"success": st.get("success"), "status": st.get("status")}
    if st.get("success") is False:
        warn.append("stage-1 optimizer did not converge (scipy status %s); it exhausted its "
                    "evaluation budget rather than settling" % st.get("status"))

    losses = np.array((s1 or {}).get("losses") or [])
    if losses.size:
        best = float(losses.min())
        n_close = int(np.sum(losses <= best * 1.01))
        checks["start_agreement"] = {"n_within_1pct": n_close, "n_starts": int(losses.size)}
        if n_close < 2:
            warn.append("only 1 of %d starts reached the best loss: the answer depends on the "
                        "seed, so report the dispersion or add starts" % losses.size)

    if keys is None:
        keys = GLM_KEYS if glm else LN_KEYS
    v = np.array([params[k] for k in keys])
    ok_loc, worst, _ = local_optimality(loss, v)
    checks["local_optimum"] = {"ok": ok_loc, "worst_relative_change": round(worst, 8)}
    if not ok_loc:
        warn.append("not at a local minimum: perturbing a parameter lowers the loss by %.1e "
                    "relative, so the search stopped early" % abs(worst))

    pred_rt = predict(params, stim, dt)
    if pred_rt is None or not np.all(np.isfinite(pred_rt)):
        rt = {"ok": False, "reason": "parameters do not produce a finite prediction"}
    else:
        gap = abs(float(np.mean(row_r2(pred_rt, resp))) - float(np.mean(r2)))
        rt = {"ok": gap < 0.01, "gap": round(gap, 8)}
    checks["roundtrip"] = rt
    if not rt.get("ok"):
        warn.append("reported parameters do not reproduce the fit -- do not report them yet")

    f = make_filter(*[params[k] for k in (("numFilt1","tauR1","tauD1","tauP1","phi1") if
                    "numFilt1" in params else FILTER_KEYS)], stim.shape[1], dt)
    if f is not None:
        lag = causality_check(f)
        checks["causality_lag_bins"] = lag
        if lag < 0:
            warn.append("filter is acausal (impulse response peaks at lag %d): the model is "
                        "predicting from future stimulus" % lag)
        try:
            fs = float(abs(np.corrcoef(f[:60], nonparametric_filter(stim, resp, 60))[0, 1]))
            checks["filter_vs_nonparametric"] = round(fs, 4)
            if fs < 0.8 and not glm:
                warn.append("fitted filter correlates only %.2f with the cross-correlation "
                            "estimate: likely a local minimum" % fs)
        except Exception:
            pass

    # numFilt is only identifiable while the rise is resolvable at this sampling. The 10-90%
    # rise of 1/(1+(tauR/t)^n) is ~ tauR*2*ln(9)/n, so once n exceeds roughly 22*tauR/dt the
    # discrete filter stops changing at all -- measured bit-identical for n = 500 vs 2187 at
    # dt = 10 ms. A fit reporting a large numFilt has slid along a flat direction, not found
    # a sharp rise, and the number should be reported as a bound rather than an estimate.
    nf_key = "numFilt1" if "numFilt1" in params else "numFilt"
    tr_key = "tauR1" if "tauR1" in params else "tauR"
    if nf_key in params and tr_key in params:
        n_crit = 22.0 * abs(params[tr_key]) / dt
        checks["numFilt_identifiable_below"] = round(float(n_crit), 1)
        if params[nf_key] > n_crit:
            warn.append(
                f"numFilt = {params[nf_key]:.0f} is above the resolvable limit for this "
                f"sampling (~{n_crit:.0f} = 22*tauR/dt): the rise is faster than one bin, so "
                f"the filter is unchanged above it and the value is a flat direction, not an "
                f"estimate. Report it as >= {n_crit:.0f}, or refit at finer dt to resolve it")

    spread = float(np.max(r2) - np.min(r2))
    checks["per_epoch_r2_spread"] = round(spread, 4)
    if spread > 0.15:
        warn.append("per-epoch R^2 spread is %.2f: suspect one bad trial rather than the model"
                    % spread)

    return {"ok": not warn, "warnings": warn, "checks": checks}


# --------------------------------------------------------------------------- two-arm
TWO_ARM_KEYS = ("numFilt1", "tauR1", "tauD1", "tauP1", "phi1",
                "numFilt2", "tauR2", "tauD2", "tauP2", "phi2",
                "alpha1", "beta1", "gamma1", "epsilon1",
                "alpha2", "beta2", "gamma2", "epsilon2")


def predict_two_arm(p, stim, dt):
    """NL1( filter1(stim) + NL2( filter2(stim) ) ) -- CascadeGraph TwoArmLnHyperNode."""
    n = stim.shape[1]
    f1 = make_filter(p["numFilt1"], p["tauR1"], p["tauD1"], p["tauP1"], p["phi1"], n, dt)
    f2 = make_filter(p["numFilt2"], p["tauR2"], p["tauD2"], p["tauP2"], p["phi2"], n, dt)
    if f1 is None or f2 is None:
        return None
    arm2 = nl(circular_conv(stim, f2), p["alpha2"], p["beta2"], p["gamma2"], p.get("epsilon2", 0.0))
    return nl(circular_conv(stim, f1) + arm2, p["alpha1"], p["beta1"], p["gamma1"], p["epsilon1"])


def fit_two_arm(stim, resp, dt, n_random_inits=20, n_restarts=10, n_arm2_starts=12, rng=None,
                diagnose=True):
    """Staged two-arm fit. epsilon2 is held at 0 -- it is exactly degenerate with gamma1.

    Arm 1 and NL1 are seeded from the single-arm LN fit, then arm 2 is added from several
    random starts, then everything is polished jointly. Starting arm 2 alongside an
    unconverged arm 1 gives the optimizer two ways to explain the same variance.

    n_arm2_starts defaults to 12 rather than 6: at 6 this reached 0.858 on synthetic two-arm
    data where the MATLAB implementation of the same procedure reached the 0.883 noise ceiling,
    i.e. the second arm's basin is narrow enough that six starts sometimes miss it. If a
    two-arm fit lands well short of a single-arm LN plus its expected gain, add starts before
    concluding anything about the cell.
    """
    rng = np.random.default_rng(rng)
    ln = fit_ln(stim, resp, dt, n_random_inits, n_restarts, rng=int(rng.integers(1 << 31)),
                dealias=False, diagnose=False)
    lp = ln["params"]
    n = stim.shape[1]

    def unpack(v):
        p = dict(zip(TWO_ARM_KEYS[:10], v[:10]))
        p.update(dict(zip(("alpha1", "beta1", "gamma1", "epsilon1"), v[10:14])))
        p.update(dict(zip(("alpha2", "beta2", "gamma2"), v[14:17])))
        p["epsilon2"] = 0.0
        return p

    def loss(v):
        pred = predict_two_arm(unpack(v), stim, dt)
        return _sse(pred, resp)

    base = [lp[k] for k in FILTER_KEYS]
    nl1 = [lp[k] for k in NL_KEYS]
    x1_sd = float(np.std(circular_conv(stim, make_filter(*base, n, dt))))

    best_v, best_f = None, np.inf
    for _ in range(n_arm2_starts):
        f2 = [rng.uniform(*RANDOM_RANGES[k]) for k in FILTER_KEYS]
        a2 = rng.uniform(0.2, 2.0) * x1_sd * rng.choice([-1.0, 1.0])   # comparable to arm 1
        v = np.array(base + f2 + nl1 + [a2, 2.0 / max(x1_sd, 1e-9), rng.uniform(-1, 1)])
        v, fv = nm_restarts(loss, v, n_restarts)
        if fv < best_f:
            best_v, best_f = v, fv

    params = unpack(best_v)
    pred = predict_two_arm(params, stim, dt)
    r2 = row_r2(pred, resp)
    out = {"params": params, "r2_per_epoch": [float(x) for x in r2],
           "r2_mean": float(r2.mean()), "dt": dt}
    if diagnose:
        free_keys = [k for k in TWO_ARM_KEYS if k != "epsilon2"]   # epsilon2 is held at 0

        def _loss(v):
            q = dict(zip(free_keys, v))
            q["epsilon2"] = 0.0
            return _sse(predict_two_arm(q, stim, dt), resp)

        out["diagnostics"] = _diagnose(params, stim, resp, dt, {"losses": [], "status": {}},
                                       predict_two_arm, _loss, r2, keys=free_keys)
    return out


def bic(resp, pred, n_free_params):
    """BIC for a Gaussian-residual fit -- the criterion Wilson & Collins use for model recovery."""
    n = resp.size
    sse = float(np.sum((resp - pred) ** 2))
    return n * np.log(sse / n) + n_free_params * np.log(n)


# --------------------------------------------------------------------------- reporting
def canonical_time_constants(params):
    """Force tauR > 0 and tauD > 0.

    make_filter uses abs(tauR) and abs(tauD) so the objective cannot blow up when the
    optimizer walks a time constant negative — which means their SIGNS are unconstrained and
    a fit returns whichever it happened to land on. The filter is bit-identical either way
    (verified at 0.0), but a negative time constant is meaningless to report and makes a
    perfectly good fit look like it recovered nothing.
    """
    p = dict(params)
    for k in ("tauR", "tauD", "tauR1", "tauD1", "tauR2", "tauD2"):
        if k in p and p[k] < 0:
            p[k] = -p[k]
    return p


def canonical_tauP_sign(params):
    """Force tauP > 0 by flipping (tauP, phi) together.

    cos() is even, so cos(2*pi*t/(-tauP) + 2*pi*(-phi)/360) == cos(2*pi*t/tauP + 2*pi*phi/360):
    negating BOTH tauP and phi leaves the filter bit-identical (verified at 0.0). Fits land on
    the negative branch about half the time, which makes a perfectly good fit look like it
    recovered the wrong period and the wrong phase. Apply this BEFORE dealias_tauP, which
    assumes a positive period.
    """
    p = dict(params)
    if p["tauP"] < 0:
        p["tauP"] = -p["tauP"]
        p["phi"] = -p["phi"]
    return p


def dealias_tauP(params, dt):
    """Fold tauP into the branch the sampling can resolve.

    The filter is only evaluated at t = k*dt, so nu = dt/tauP and 1-nu (with phi -> -phi)
    give the same discrete filter. An optimizer will return either; only one is
    interpretable as an oscillation period.
    """
    p = dict(params)
    nu = dt / p["tauP"]
    if nu > 0.5:
        p["tauP"] = dt / (1.0 - nu)
        p["phi"] = -p["phi"]
    return p


def canonical_sign(params):
    """Put a fit on the alpha > 0 branch of the exact sign degeneracy.

    (phi+180, alpha -> -alpha, gamma -> -gamma, epsilon -> alpha+epsilon) leaves the
    prediction bit-identical, and a fit lands on either branch at random, so alpha, gamma and
    epsilon are not comparable across cells until a branch is chosen. a_fb flips with alpha
    when present.

    There is only one binary choice here, so pinning alpha's sign and pinning the filter's
    sign are the same act: requiring the parametric filter to correlate positively with the
    cross-correlation filter is exactly equivalent to requiring alpha > 0, since the
    cross-correlation filter is proportional to alpha times the filter. This function takes
    the cheaper route and pins alpha > 0. The consequence is that the cell's polarity lives
    in the *filter's* sign, not in alpha's -- an OFF cell and an ON cell both report
    alpha > 0, and their filters are inverted relative to each other. Whichever convention
    you use, state it, and do not compare alpha across fits that used different ones.
    """
    p = dict(params)
    if p["alpha"] >= 0:
        return p
    p["phi"] = (p["phi"] + 360.0) % 360.0 - 180.0
    p["epsilon"] = p["alpha"] + p["epsilon"]
    p["alpha"] = -p["alpha"]
    p["gamma"] = -p["gamma"]
    if "a_fb" in p:
        p["a_fb"] = -p["a_fb"]
    return p


def filter_shape_check(params, stim, resp, dt, n_points=60):
    """|corr| between the fitted filter and the cross-correlation filter. Shape only.

    Use the magnitude: the sign carries the polarity convention, which canonical_sign has
    already fixed. A value below ~0.9 means the parametric fit is in a different basin than
    the data's own linear estimate, whatever the R^2 says.
    """
    f = make_filter(*[params[k] for k in FILTER_KEYS], stim.shape[1], dt)
    if f is None:
        return None
    fnp = nonparametric_filter(stim, resp, n_points)
    return float(abs(np.corrcoef(f[:n_points], fnp)[0, 1]))


def roundtrip(params, stim, resp, dt, claimed_r2_mean=None, tol=0.01, glm=False):
    """Rebuild from exactly these parameters and confirm they reproduce the fit."""
    pred = predict_glm(params, stim, dt) if glm else predict_ln(params, stim, dt)
    if pred is None or not np.all(np.isfinite(pred)):
        return {"ok": False, "reason": "parameters do not produce a finite prediction"}
    r2 = row_r2(pred, resp)
    out = {"r2_per_epoch": [float(v) for v in r2], "r2_mean": float(r2.mean()), "ok": True}
    if claimed_r2_mean is not None:
        out["gap"] = abs(float(claimed_r2_mean) - out["r2_mean"])
        out["ok"] = out["gap"] < tol
    return out


def local_optimality(loss, params_vec, rel=0.02, floor=1e-4):
    """Is this even a local minimum? Perturb each coordinate and check the loss rises.

    A fit that terminated on maxfev rather than convergence often sits on a slope, not in a
    basin. This is the cheapest possible check and it catches that outright: any coordinate
    where the loss *drops* means the optimizer simply stopped early.

    Returns (ok, worst_drop, per_coordinate) where a negative delta is a downhill direction.
    """
    p = np.asarray(params_vec, dtype=float)
    f0 = loss(p)
    deltas = []
    for i in range(len(p)):
        step = max(abs(p[i]) * rel, floor)
        up, dn = p.copy(), p.copy()
        up[i] += step
        dn[i] -= step
        deltas.append(min(loss(up) - f0, loss(dn) - f0))
    worst = min(deltas)
    return bool(worst >= 0), float(worst / max(abs(f0), 1e-30)), [float(d) for d in deltas]


def start_dispersion(stim, resp, dt, n_random_inits=20, n_restarts=10, rng=None):
    """Spread of the Stage-1 loss across independent starts.

    If the best few starts agree to a few parts in 1e3 you have real evidence the search is
    finding the same basin repeatedly. If they disagree, the reported fit is whichever start
    happened to be luckiest, and the number you quote depends on the seed.
    """
    rng = np.random.default_rng(rng)
    n_points = stim.shape[1]

    def loss(p):
        f = make_filter(p[0], p[1], p[2], p[3], p[4], n_points, dt)
        if f is None:
            return PENALTY
        return _sse(p[5] * circular_conv(stim, f), resp)

    keys = list(FILTER_KEYS) + ["scFact"]
    losses = []
    for i in range(n_random_inits):
        s = (np.array([DEFAULT_START[k] for k in keys]) if i == 0
             else np.array([rng.uniform(*RANDOM_RANGES[k]) for k in keys]))
        _p, fv = nm_restarts(loss, s, n_restarts)
        losses.append(fv)
    losses = np.sort(np.array(losses))
    best = losses[0]
    return {"best": float(best),
            "n_within_1pct": int(np.sum(losses <= best * 1.01)),
            "n_starts": int(n_random_inits),
            "spread_ratio": float(losses[min(4, len(losses) - 1)] / best) if best > 0 else float("inf")}


def parameter_recovery(params, stim, dt, resid_sd, n_reps=3, rng=None, glm=False, **fit_kw):
    """Can this pipeline recover its own parameters from data it generated?

    Simulate a response from `params` plus noise matched to the residual you actually saw,
    refit with the same pipeline, and compare. This needs no ground truth — it is the one
    check that directly asks whether the optimizer works on this problem at this SNR. If it
    cannot recover parameters from its own synthetic data, a good R^2 on the real data is not
    evidence the fit is right.

    Returns one dict per repetition with the input and recovered parameters and their
    relative error.
    """
    rng = np.random.default_rng(rng)
    out = []
    predict = predict_glm if glm else predict_ln
    fit = fit_glm if glm else fit_ln
    clean = predict(params, stim, dt)
    for _ in range(n_reps):
        sim = clean + rng.standard_normal(clean.shape) * resid_sd
        got = fit(stim, sim, dt, rng=int(rng.integers(1 << 31)), **fit_kw)["params"]
        keys = [k for k in (GLM_KEYS if glm else LN_KEYS)]
        err = {k: (abs(got[k] - params[k]) / max(abs(params[k]), 1e-12)) for k in keys}
        out.append({"recovered": {k: float(got[k]) for k in keys},
                    "rel_error": {k: float(v) for k, v in err.items()},
                    "worst_key": max(err, key=err.get),
                    "worst_rel_error": float(max(err.values()))})
    return out


def causality_check(filt, n_points=None, impulse_at=None):
    """Confirm a kernel is causal under this module's circular-convolution convention.

    Returns the lag in bins between an impulse and the peak of the response. Positive means
    the response follows the stimulus, which is what you want. Negative means the kernel is
    time-reversed and the model is predicting from future stimulus — the single most common
    silent error in this pipeline, because such a fit still reports a plausible R^2.

    Run this on any kernel you did not build with make_filter: a nonparametric estimate from
    computeFilter, anything convolved with mode="same", anything you flipped.
    """
    n = n_points or len(filt)
    at = impulse_at if impulse_at is not None else n // 4
    imp = np.zeros((1, n))
    imp[0, at] = 1.0
    y = circular_conv(imp, np.asarray(filt)[:n])[0]
    lag = int(np.argmax(np.abs(y))) - at
    if lag > n // 2:                      # unwrap the circular index
        lag -= n
    return lag


def nonparametric_filter(stim, resp, n_points=None):
    """Cross-correlation filter estimate, for checking the parametric fit's shape."""
    n_points = n_points or stim.shape[1]
    S, R = np.fft.fft(stim, axis=1), np.fft.fft(resp - resp.mean(axis=1, keepdims=True), axis=1)
    F = np.mean(R * np.conj(S), axis=0)
    F[0] = 0
    return np.real(np.fft.ifft(F))[:n_points]
