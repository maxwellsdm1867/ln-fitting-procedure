"""Blind parameter-recovery benchmark: three cells with deliberately unguessable parameters.

The point is to defeat priors. Every other dataset here uses textbook-plausible retinal
values (tauD ~ 45 ms, numFilt ~ 4-5), which a model could land near without genuinely
optimising anything. Here the generating parameters are drawn from wide ranges and are
non-round on purpose -- numFilt = 7.31, tauD = 137 ms, alpha = -212.7 -- so the only way to
report them is to run the pipeline and let it converge.

SNR is varied across the three cells so the benchmark also shows where recovery degrades:
the interesting answer is not "did it recover everything" but "which parameters go first,
and at what noise level".
"""
import json
import os

import numpy as np

from generate_data import (DECIM, DT, DT_RAW, add_noise_for_target_r2, cg_filter, circ_conv,
                           make_stim, nl, row_r2, save, upsample)

# ranges deliberately wider, and shifted away from, the textbook values used elsewhere
RANGES = dict(num_filt=(1.5, 9.5), tau_r=(0.004, 0.09), tau_d=(0.012, 0.19),
              tau_p=(0.018, 0.095), phi=(-180.0, 180.0))
CELLS = [("blind_cell_a", 0.95, 3, 900), ("blind_cell_b", 0.85, 3, 900),
         ("blind_cell_c", 0.70, 3, 900)]


def one(name, target_r2, n_ep, n_bins, rng):
    p = {k: float(np.round(rng.uniform(*v), 4)) for k, v in RANGES.items()}
    stim = make_stim(rng, n_ep, n_bins)
    f = cg_filter(p["num_filt"], p["tau_r"], p["tau_d"], p["tau_p"], p["phi"], n_bins, DT)
    x = circ_conv(stim, f)
    sign = rng.choice([-1.0, 1.0])
    alpha = float(np.round(sign * rng.uniform(20.0, 260.0), 2))
    beta = float(np.round(rng.uniform(1.2, 2.8) / x.std(), 6))
    gamma = float(np.round(rng.uniform(-1.2, 1.2), 4))
    epsilon = float(np.round(rng.uniform(-90.0, 40.0), 2))
    clean = nl(x, alpha, beta, gamma, epsilon)
    resp = add_noise_for_target_r2(rng, clean, target_r2)
    r2 = row_r2(clean, resp)
    print(f"  {name}: ceiling {r2.mean():.3f}  numFilt={p['num_filt']} tauR={p['tau_r']} "
          f"tauD={p['tau_d']} tauP={p['tau_p']} phi={p['phi']} alpha={alpha}")
    save(name, upsample(stim, DECIM), upsample(resp, DECIM),
         {"cell_type": "unspecified", "protocol": "Variable Mean Noise",
          "sample_interval_s": DT_RAW, "response_units": "mV", "n_epochs": n_ep,
          "note": "stim and resp are (epochs x time) at 0.1 ms."},
         {"filter": p, "alpha": alpha, "beta": beta, "gamma": gamma, "epsilon": epsilon,
          "dt": DT, "ceiling_r2_per_epoch": r2.tolist(), "ceiling_r2_mean": float(r2.mean()),
          "filter_10ms": f.tolist(), "target_r2": target_r2})


if __name__ == "__main__":
    print("blind recovery cells (parameters drawn wide and non-round on purpose):")
    rng = np.random.default_rng(606061)
    for name, t, n_ep, n_bins in CELLS:
        one(name, t, n_ep, n_bins, rng)
    print("done.")
