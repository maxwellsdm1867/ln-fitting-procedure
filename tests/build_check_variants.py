"""Generate a correct hand-rolled fitting script, plus one variant per single mistake."""
import os, re

BASE = '''import json, numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

DT = 0.01

def make_filter(n, tauR, tauD, tauP, phi, npts, dt):
    t = np.arange(1, npts + 1) * dt
    rise = 1.0 / (1.0 + (abs(tauR) / t) ** n)
    f = rise * np.exp(-t / abs(tauD)) * np.cos(2*np.pi*t/tauP + 2*np.pi*phi/360)
    f = f / np.max(np.abs(f))
    f = f - np.mean(f)
    return f

def conv(s, f):
    return np.real(np.fft.ifft(np.fft.fft(s, axis=1) * np.fft.fft(f)[None, :], axis=1))

def nl(x, a, b, g, e):
    return a * norm.cdf(b * x + g) + e

def row_r2(pred, meas):
    sse = np.sum((meas - pred) ** 2, axis=1)
    sst = np.sum((meas - meas.mean(axis=1, keepdims=True)) ** 2, axis=1)
    return 1.0 - sse / sst

def load():
    d = np.load("data.npz")
    stim = d["stim"].astype(float); resp = d["resp"].astype(float)
    ne, nr = stim.shape; nb = nr // 100
    stim = stim[:, :nb*100].reshape(ne, nb, 100).mean(axis=2)
    resp = resp[:, :nb*100].reshape(ne, nb, 100).mean(axis=2)
    stim = stim - stim.mean(axis=1, keepdims=True)
    return stim, resp

def fit(stim, resp, rng):
    best, bestf = None, np.inf
    starts = [np.array([4,0.02,0.01,0.02,1.0,-55.0,1.0,-0.4,18.0])]
    for _ in range(19):
        starts.append(np.array([rng.uniform(1,10), rng.uniform(.005,.1), rng.uniform(.005,.2),
                                rng.uniform(.01,.1), rng.uniform(-180,180),
                                rng.uniform(-200,200), rng.uniform(.1,3), rng.uniform(-2,2),
                                rng.uniform(-50,50)]))
    def loss(p):
        f = make_filter(p[0],p[1],p[2],p[3],p[4], stim.shape[1], DT)
        v = np.sum((resp - nl(conv(stim,f), p[5],p[6],p[7],p[8]))**2)
        return v if np.isfinite(v) else 1e12
    for s in starts:
        r = minimize(loss, s, method="Nelder-Mead", options={"maxfev":1800})
        if r.fun < bestf: best, bestf, status = r.x, r.fun, r.status
    return best, status
'''

VARIANTS = {
 "00_correct":        [],
 "01_t_origin":       [("t = np.arange(1, npts + 1) * dt", "t = np.arange(npts) * dt + 1e-12")],
 "02_phi_radians":    [("2*np.pi*t/tauP + 2*np.pi*phi/360", "2*np.pi*t/tauP + phi")],
 "03_no_unit_peak":   [("    f = f / np.max(np.abs(f))\n", "")],
 "04_no_zero_dc":     [("    f = f - np.mean(f)\n", "")],
 "05_nl_grouping":    [("a * norm.cdf(b * x + g) + e", "a * norm.cdf(b * (x + g)) + e")],
 "06_zscored_resp":   [("    return stim, resp", "    resp = (resp - resp.mean()) / resp.std()\n    return stim, resp")],
 "07_rectified":      [("    return stim, resp", "    resp = np.maximum(resp, 0.0)\n    return stim, resp")],
 "08_causal_conv":    [("    return np.real(np.fft.ifft(np.fft.fft(s, axis=1) * np.fft.fft(f)[None, :], axis=1))",
                        "    out = np.empty_like(s)\n    for i in range(s.shape[0]):\n        out[i] = np.convolve(s[i], f)[:s.shape[1]]\n    return out")],
 "09_truncated_conv": [("np.fft.fft(f)[None, :]", "np.fft.fft(np.concatenate([f[:60], np.zeros(len(f)-60)]))[None, :]")],
 "10_overflow_form":  [("rise = 1.0 / (1.0 + (abs(tauR) / t) ** n)", "rise = ((t/abs(tauR))**n) / (1 + (t/abs(tauR))**n)")],
 "11_single_start":   [("    for _ in range(19):", "    for _ in range(0):")],
 "12_no_stim_meansub":[("    stim = stim - stim.mean(axis=1, keepdims=True)\n", "")],
}

for name, subs in VARIANTS.items():
    src = BASE
    for old, new in subs:
        assert old in src, (name, old[:40])
        src = src.replace(old, new, 1)
    open(f"{name}.py", "w").write(src)
print(f"wrote {len(VARIANTS)} variants")
