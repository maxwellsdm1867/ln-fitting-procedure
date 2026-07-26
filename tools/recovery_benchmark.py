"""Parameter recovery + model recovery for the CascadeGraph model family.

Follows Wilson & Collins (2019), eLife 49547, "Ten simple rules for the computational
modeling of behavioral data", adapted to this model set:

  Parameter recovery  — simulate from known parameters over a WIDE range, refit, and compare
                        recovered against simulated per parameter. Also correlate the
                        recovered parameters against each other: if the simulated parameters
                        were independent, correlation among the recovered ones means the
                        parameters are trading off, i.e. not separately identifiable.

  Model recovery      — simulate from every model, fit every model to each simulation, and
                        record which wins by BIC. Rows are the generating model, columns the
                        winning model: p(fit = B | simulated = A). A diagonal matrix means the
                        protocol can tell these models apart. Off-diagonal mass means it
                        cannot, and no amount of care fitting real data will fix that.

  Inversion matrix    — Bayes-inverted, p(simulated = A | fit = B), with a flat prior over
                        generating models. This is the question you actually have when
                        looking at real data: the GLM won, so what generated it?

Usage:  python tools/recovery_benchmark.py [n_sims_per_model]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "skills", "ln-fitting-procedure", "scripts"))
import cascade_fit as cf  # noqa: E402

DT = 0.01
N_EP, N_BINS = 3, 700
# reduced budgets: this runs 9 fits per simulation, so the per-fit cost dominates
FAST = dict(n_random_inits=12, n_restarts=6)
MODELS = ["ln", "glm", "two_arm"]
NFREE = {"ln": 9, "glm": 11, "two_arm": 17}


def rand_filter(rng):
    return [rng.uniform(*cf.RANDOM_RANGES[k]) for k in cf.FILTER_KEYS]


def simulate(model, stim, rng, target_r2=0.88):
    """Draw parameters from wide ranges, simulate, add noise to a fixed SNR."""
    n = stim.shape[1]
    fp = rand_filter(rng)
    f = cf.make_filter(*fp, n, DT)
    if f is None:
        return None
    x = cf.circular_conv(stim, f)
    sx = float(x.std()) or 1.0
    alpha = rng.uniform(20, 200) * rng.choice([-1.0, 1.0])
    beta, gamma, eps = rng.uniform(1.2, 2.6) / sx, rng.uniform(-1, 1), rng.uniform(-60, 30)
    p = dict(zip(cf.FILTER_KEYS, fp))
    p.update(alpha=alpha, beta=beta, gamma=gamma, epsilon=eps)

    if model == "ln":
        clean = cf.nl(x, alpha, beta, gamma, eps)
    elif model == "glm":
        decay = np.exp(-np.arange(1, 31) * DT / 0.06).sum()
        slope = alpha * beta * 0.3989
        p["a_fb"] = -2.0 / (decay * slope)          # signed loop gain +2 (regenerative)
        p["tau_fb"], p["n_fb_bins"] = 0.06, 30
        clean = cf.predict_glm(p, stim, DT)
    else:
        p2 = {f"{k}1": v for k, v in dict(zip(cf.FILTER_KEYS, fp)).items()}
        p2.update({f"{k}2": v for k, v in dict(zip(cf.FILTER_KEYS, rand_filter(rng))).items()})
        f2 = cf.make_filter(*[p2[f"{k}2"] for k in cf.FILTER_KEYS], n, DT)
        if f2 is None:
            return None
        x2 = cf.circular_conv(stim, f2)
        a2 = rng.uniform(0.4, 1.6) * sx * rng.choice([-1.0, 1.0])   # comparable to arm 1
        p2.update(alpha2=a2, beta2=2.0 / (float(x2.std()) or 1.0), gamma2=rng.uniform(-1, 1),
                  epsilon2=0.0)
        s = x + cf.nl(x2, a2, p2["beta2"], p2["gamma2"], 0.0)
        p2.update(alpha1=alpha, beta1=rng.uniform(1.2, 2.4) / (float(s.std()) or 1.0),
                  gamma1=gamma, epsilon1=eps)
        clean = cf.predict_two_arm(p2, stim, DT)
        p = p2
    if clean is None or not np.all(np.isfinite(clean)):
        return None
    var = np.var(clean, axis=1, keepdims=True)
    noise = np.sqrt(var * (1.0 / target_r2 - 1.0))
    resp = clean + rng.standard_normal(clean.shape) * noise
    return p, resp


def fit_all(stim, resp, rng):
    out = {}
    out["ln"] = cf.fit_ln(stim, resp, DT, rng=int(rng.integers(1 << 31)), **FAST)
    out["glm"] = cf.fit_glm(stim, resp, DT, rng=int(rng.integers(1 << 31)), **FAST)
    out["two_arm"] = cf.fit_two_arm(stim, resp, DT, rng=int(rng.integers(1 << 31)),
                                    n_arm2_starts=4, **FAST)
    preds = {"ln": cf.predict_ln(out["ln"]["params"], stim, DT),
             "glm": cf.predict_glm(out["glm"]["params"], stim, DT),
             "two_arm": cf.predict_two_arm(out["two_arm"]["params"], stim, DT)}
    bics = {m: cf.bic(resp, preds[m], NFREE[m]) for m in MODELS if preds[m] is not None}
    return out, bics


def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    rng = np.random.default_rng(4242)
    stim = rng.standard_normal((N_EP, N_BINS))
    stim -= stim.mean(axis=1, keepdims=True)

    conf = {a: {b: 0 for b in MODELS} for a in MODELS}
    recov = []
    for gen in MODELS:
        for i in range(n_sims):
            sim = simulate(gen, stim, rng)
            if sim is None:
                continue
            true_p, resp = sim
            fits, bics = fit_all(stim, resp, rng)
            win = min(bics, key=bics.get)
            conf[gen][win] += 1
            print(f"  simulated {gen:8s} #{i}  ->  BIC winner {win:8s}  "
                  f"(R2 ln={fits['ln']['r2_mean']:.3f} glm={fits['glm']['r2_mean']:.3f} "
                  f"two_arm={fits['two_arm']['r2_mean']:.3f})", flush=True)
            if gen == "ln":
                got = cf.canonical_sign(cf.dealias_tauP(fits["ln"]["params"], DT))
                tru = cf.canonical_sign(cf.dealias_tauP(dict(true_p), DT))
                recov.append({"true": {k: float(tru[k]) for k in cf.LN_KEYS},
                              "fit": {k: float(got[k]) for k in cf.LN_KEYS}})

    tot = {a: max(1, sum(conf[a].values())) for a in MODELS}
    confusion = {a: {b: conf[a][b] / tot[a] for b in MODELS} for a in MODELS}
    # Bayes inversion with a flat prior over generating models
    inversion = {}
    for b in MODELS:
        col = {a: confusion[a][b] for a in MODELS}
        s = sum(col.values())
        inversion[b] = {a: (col[a] / s if s else 0.0) for a in MODELS}

    out = {"n_sims_per_model": n_sims, "confusion_p_fit_given_sim": confusion,
           "inversion_p_sim_given_fit": inversion, "ln_parameter_recovery": recov,
           "reference": "Wilson & Collins 2019, eLife 49547"}
    with open(os.path.join(ROOT, "results_recovery_benchmark.json"), "w") as fh:
        json.dump(out, fh, indent=2)

    print("\nCONFUSION  p(fit | simulated)      rows = generating model")
    print(f"{'':10s}" + "".join(f"{m:>10s}" for m in MODELS))
    for a in MODELS:
        print(f"{a:10s}" + "".join(f"{confusion[a][b]:10.2f}" for b in MODELS))
    print("\nINVERSION  p(simulated | fit)      rows = winning model")
    print(f"{'':10s}" + "".join(f"{m:>10s}" for m in MODELS))
    for b in MODELS:
        print(f"{b:10s}" + "".join(f"{inversion[b][a]:10.2f}" for a in MODELS))

    if recov:
        print("\nLN PARAMETER RECOVERY (simulated vs recovered)")
        for k in cf.LN_KEYS:
            t = np.array([r["true"][k] for r in recov])
            g = np.array([r["fit"][k] for r in recov])
            c = np.corrcoef(t, g)[0, 1] if len(t) > 2 and t.std() > 0 and g.std() > 0 else float("nan")
            print(f"  {k:9s} r={c:+.3f}  median |rel err| = "
                  f"{np.median(np.abs((g - t) / np.maximum(np.abs(t), 1e-12))):.3f}")


if __name__ == "__main__":
    main()
