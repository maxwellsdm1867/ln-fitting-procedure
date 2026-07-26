"""Datasets for evals 3 and 4.

  no_feedback_control : a pure LN cell, handed to the "does this cell need feedback?" task.
                        Correct answer is that feedback buys nothing and the LN should be
                        reported. Tests over-claiming, which in-sample R^2 always rewards
                        because the GLM nests the LN at a_fb = 0.

  two_arm_cascade     : CascadeGraph TwoArmLnHyperNode topology,
                        NL1( filter1(s) + NL2( filter2(s) ) ), 18 free parameters.
                        epsilon2 is fixed at 0 because (epsilon2, gamma1) is an exact
                        degeneracy, so the generating parameters are the identifiable ones.
"""
import json
import os

import numpy as np

from generate_data import (DECIM, DT, DT_RAW, add_noise_for_target_r2, cg_filter, circ_conv,
                           make_stim, nl, row_r2, save, upsample)


def dataset_no_feedback_control():
    print("[4] no_feedback_control")
    rng = np.random.default_rng(31337)
    n_ep, n_bins = 3, 700
    p = dict(num_filt=6, tau_r=0.022, tau_d=0.055, tau_p=0.072, phi=-75.0)
    stim = make_stim(rng, n_ep, n_bins)
    f = cg_filter(p["num_filt"], p["tau_r"], p["tau_d"], p["tau_p"], p["phi"], n_bins, DT)
    x = circ_conv(stim, f)
    alpha, beta, gamma, epsilon = 38.0, 2.1 / x.std(), 0.25, -47.0
    clean = nl(x, alpha, beta, gamma, epsilon)
    resp = add_noise_for_target_r2(rng, clean, 0.87)
    r2 = row_r2(clean, resp)
    print(f"    ceiling per-epoch R^2 = {np.round(r2, 3)}  (mean {r2.mean():.3f})  NO feedback present")
    save("no_feedback_control", upsample(stim, DECIM), upsample(resp, DECIM),
         {"cell_type": "On parasol", "protocol": "Variable Mean Noise, ConeResponseFull",
          "sample_interval_s": DT_RAW, "response_units": "mV", "n_epochs": n_ep,
          "note": "stim and resp are (epochs x time) at 0.1 ms."},
         {"filter": p, "alpha": alpha, "beta": beta, "gamma": gamma, "epsilon": epsilon,
          "dt": DT, "ceiling_r2_per_epoch": r2.tolist(), "ceiling_r2_mean": float(r2.mean()),
          "filter_10ms": f.tolist(), "a_fb": 0.0,
          "correct_verdict": "no feedback: a GLM should recover a_fb ~ 0 and gain nothing "
                             "beyond noise over the LN"})


def dataset_two_arm():
    print("[5] two_arm_cascade")
    rng = np.random.default_rng(2468)
    n_ep, n_bins = 3, 800
    p1 = dict(num_filt=4, tau_r=0.028, tau_d=0.050, tau_p=0.070, phi=25.0)
    p2 = dict(num_filt=3, tau_r=0.014, tau_d=0.028, tau_p=0.045, phi=-95.0)
    stim = make_stim(rng, n_ep, n_bins)
    f1 = cg_filter(p1["num_filt"], p1["tau_r"], p1["tau_d"], p1["tau_p"], p1["phi"], n_bins, DT)
    f2 = cg_filter(p2["num_filt"], p2["tau_r"], p2["tau_d"], p2["tau_p"], p2["phi"], n_bins, DT)
    x1, x2 = circ_conv(stim, f1), circ_conv(stim, f2)

    # arm 2: the nonlinear correction. epsilon2 == 0 (degenerate with gamma1).
    # Arms must be COMPARABLE in magnitude to be identifiable: if either dominates,
    # NL1(NL2(f2*s)) collapses to one filter with a composed static nonlinearity and
    # a single-arm LN fits it. Headroom peaks (~0.10) near arm2_sd ~ arm1_sd.
    alpha2, beta2, gamma2, epsilon2 = 3.0, 3.0 / x2.std(), 0.5, 0.0
    arm2 = nl(x2, alpha2, beta2, gamma2, epsilon2)
    summed = x1 + arm2
    alpha1, beta1, gamma1, epsilon1 = 65.0, 1.7 / summed.std(), -0.35, -30.0
    clean = nl(summed, alpha1, beta1, gamma1, epsilon1)

    # how much of this a single-arm LN can reach, for reference
    print(f"    arm1_sd={x1.std():.2f}  arm2_sd={arm2.std():.2f}  "
          f"(comparable -> the second arm is identifiable)")
    resp = add_noise_for_target_r2(rng, clean, 0.88)
    r2 = row_r2(clean, resp)
    print(f"    ceiling per-epoch R^2 = {np.round(r2, 3)}  (mean {r2.mean():.3f})")
    save("two_arm_cascade", upsample(stim, DECIM), upsample(resp, DECIM),
         {"cell_type": "Off parasol", "protocol": "Variable Mean Noise, ConeResponseFull",
          "sample_interval_s": DT_RAW, "response_units": "mV", "n_epochs": n_ep,
          "note": "stim and resp are (epochs x time) at 0.1 ms."},
         {"filter1": p1, "filter2": p2,
          "alpha1": alpha1, "beta1": beta1, "gamma1": gamma1, "epsilon1": epsilon1,
          "alpha2": alpha2, "beta2": beta2, "gamma2": gamma2, "epsilon2": epsilon2,
          "dt": DT, "ceiling_r2_per_epoch": r2.tolist(), "ceiling_r2_mean": float(r2.mean()),
          "filter1_10ms": f1.tolist(), "filter2_10ms": f2.tolist(),
          "topology": "NL1( filter1(stim) + NL2( filter2(stim) ) )  -- CascadeGraph TwoArmLnHyperNode",
          "note": "epsilon2 fixed at 0: (epsilon2, gamma1) is an exact degeneracy",
          "best_single_arm_ln_r2_vs_clean": 0.899,
          "arm1_sd": float(x1.std()), "arm2_sd": float(arm2.std())})




def dataset_time_axis_trap():
    """A clean LN cell whose filter has an unmistakable +70 ms peak latency.

    The planted bugs are all time-axis bugs, so the eval isolates the failure mode the lab
    actually hits (losing track of which end of the kernel is causal) rather than re-testing
    the preprocessing bugs already covered by on_parasol_broken.
    """
    print("[6] time_axis_trap")
    rng = np.random.default_rng(90210)
    n_ep, n_bins = 3, 800
    p = dict(num_filt=5, tau_r=0.055, tau_d=0.090, tau_p=0.60, phi=0.0)
    stim = make_stim(rng, n_ep, n_bins)
    f = cg_filter(p["num_filt"], p["tau_r"], p["tau_d"], p["tau_p"], p["phi"], n_bins, DT)
    peak_bin = int(np.argmax(np.abs(f)))
    x = circ_conv(stim, f)
    alpha, beta, gamma, epsilon = -48.0, 1.9 / x.std(), -0.2, 12.0
    clean = nl(x, alpha, beta, gamma, epsilon)
    resp = add_noise_for_target_r2(rng, clean, 0.89)
    r2 = row_r2(clean, resp)
    print(f"    filter peak at bin {peak_bin} = {(peak_bin + 1) * DT * 1000:.0f} ms (causal, positive lag)")
    print(f"    ceiling per-epoch R^2 = {np.round(r2, 3)}  (mean {r2.mean():.3f})")
    save("time_axis_trap", upsample(stim, DECIM), upsample(resp, DECIM),
         {"cell_type": "Off parasol", "protocol": "Variable Mean Noise, ConeResponseFull",
          "sample_interval_s": DT_RAW, "response_units": "mV", "n_epochs": n_ep,
          "note": "stim and resp are (epochs x time) at 0.1 ms."},
         {"filter": p, "alpha": alpha, "beta": beta, "gamma": gamma, "epsilon": epsilon,
          "dt": DT, "ceiling_r2_per_epoch": r2.tolist(), "ceiling_r2_mean": float(r2.mean()),
          "filter_10ms": f.tolist(),
          "peak_latency_bins": peak_bin, "peak_latency_ms": (peak_bin + 1) * DT * 1000,
          "planted_bugs": [
              "np.convolve(..., mode='same') centres the kernel so half of it acts at negative lag",
              "the nonparametric filter estimate is taken from the END of the ifft output "
              "(the anticausal half) and used as if it were causal",
              "peak latency is therefore reported as a negative number",
          ]})


if __name__ == "__main__":
    dataset_no_feedback_control()
    dataset_two_arm()
    dataset_time_axis_trap()
    print("done.")
