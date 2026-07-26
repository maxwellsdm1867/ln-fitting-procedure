"""Standard diagnostic figures for a cascade fit. One command, one PNG, always the same panels.

    python make_figures.py results.json data.npz [out.png]

Six panels, in the order you actually need to look at them:

  1. INPUT  — the stimulus, one epoch, so you can see what the cell was shown
  2. OUTPUT — the response over the same window, with the model prediction on top
  3. FILTER — the recovered filter, with the cross-correlation estimate overlaid; if these
              disagree the parametric fit found a local minimum whatever the R^2 says
  4. NONLINEARITY — the binned (filter output, response) relationship with the fitted sigmoid.
              A sigmoid sitting in one saturated tail, or running straight through without
              curving, means the cell was never driven across its nonlinear range
  5. RESIDUALS vs prediction — structure here is the model missing something systematic
  6. PER-EPOCH R^2 — the spread, against the mean. One epoch far below the rest is usually a
              bad trial, not a bad model

The point of a fixed panel set is that you learn where to look. A figure assembled ad hoc for
each fit has to be re-read every time.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cascade_fit as cf  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    results = json.load(open(sys.argv[1]))
    data = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "fit_figures.png"

    p = {k: float(v) for k, v in results["params"].items()}
    dt = float(results.get("dt", 0.01))
    glm = "a_fb" in p
    stim, resp, info = cf.load_epochs(data, dt=dt, verbose=False)
    pred = cf.predict_glm(p, stim, dt) if glm else cf.predict_ln(p, stim, dt)
    r2 = cf.row_r2(pred, resp)

    n = stim.shape[1]
    t = np.arange(n) * dt
    show = slice(0, min(n, int(4.0 / dt)))          # first 4 s, or the whole epoch
    filt = cf.make_filter(*[p[k] for k in cf.FILTER_KEYS], n, dt)
    x = cf.circular_conv(stim, filt)

    fig, ax = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle(f"{os.path.basename(data)} — {'GLM' if glm else 'LN'} fit, "
                 f"mean per-epoch R² = {r2.mean():.4f}", fontsize=12)

    ax[0, 0].plot(t[show], stim[0][show], lw=0.7, color="0.35")
    ax[0, 0].set_title("1. input: stimulus (epoch 1)"); ax[0, 0].set_xlabel("time (s)")

    ax[0, 1].plot(t[show], resp[0][show], lw=0.9, color="0.25", label="response")
    ax[0, 1].plot(t[show], pred[0][show], lw=0.9, color="tab:red", label="model")
    ax[0, 1].set_title(f"2. output: epoch 1, R² = {r2[0]:.3f}")
    ax[0, 1].set_xlabel("time (s)"); ax[0, 1].legend(fontsize=8)

    npts = min(60, n)
    fnp = cf.nonparametric_filter(stim, resp, npts)
    fpar = filt[:npts]
    s = np.sign(np.dot(fpar, fnp)) or 1.0
    ax[1, 0].plot(np.arange(1, npts + 1) * dt * 1e3, fpar / np.max(np.abs(fpar)),
                  lw=1.4, label="fitted")
    ax[1, 0].plot(np.arange(1, npts + 1) * dt * 1e3, s * fnp / np.max(np.abs(fnp)),
                  lw=1.0, ls="--", color="0.5", label="cross-correlation")
    ax[1, 0].axhline(0, lw=0.5, color="0.8")
    corr = abs(np.corrcoef(fpar, fnp)[0, 1])
    ax[1, 0].set_title(f"3. filter (|corr| with x-corr estimate = {corr:.3f})")
    ax[1, 0].set_xlabel("lag (ms)"); ax[1, 0].legend(fontsize=8)

    xb, yb = cf.binned_nl(x, resp, 40)
    xs = np.linspace(xb.min(), xb.max(), 200)
    ax[1, 1].plot(xb, yb, "o", ms=4, color="0.35", label="binned data")
    ax[1, 1].plot(xs, cf.nl(xs, p["alpha"], p["beta"], p["gamma"], p["epsilon"]),
                  lw=1.6, color="tab:red", label="fitted")
    ax[1, 1].set_title("4. nonlinearity")
    ax[1, 1].set_xlabel("filter output"); ax[1, 1].legend(fontsize=8)

    res = (resp - pred).ravel()
    ax[2, 0].plot(pred.ravel(), res, ".", ms=1, alpha=0.25, color="0.4")
    ax[2, 0].axhline(0, lw=0.8, color="tab:red")
    ax[2, 0].set_title("5. residuals vs prediction (structure = something systematic missed)")
    ax[2, 0].set_xlabel("prediction")

    ax[2, 1].bar(np.arange(1, len(r2) + 1), r2, color="0.55")
    ax[2, 1].axhline(r2.mean(), color="tab:red", lw=1.2,
                     label=f"mean {r2.mean():.3f}")
    ax[2, 1].set_ylim(0, 1)
    ax[2, 1].set_title(f"6. per-epoch R² (spread {r2.max()-r2.min():.3f})")
    ax[2, 1].set_xlabel("epoch"); ax[2, 1].legend(fontsize=8)

    for a in ax.ravel():
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
    print(f"  mean per-epoch R² {r2.mean():.4f}   per-epoch {np.round(r2, 4).tolist()}")
    print(f"  filter vs cross-correlation |corr| = {corr:.3f}"
          + ("   <-- below 0.8, suspect a local minimum" if corr < 0.8 else ""))


if __name__ == "__main__":
    main()
