# GLM with feedback (11 parameters)

Read this after `SKILL.md` — the filter, nonlinearity, convolution, preprocessing,
numerical hygiene, optimizer settings, and R² conventions are all identical. This file
only covers what feedback adds.

## Contents

- [What feedback adds](#what-feedback-adds)
- [Never teacher-force](#never-teacher-force)
- [Staged pipeline](#staged-pipeline)
- [Adaptive or regenerative?](#adaptive-or-regenerative-use-the-signed-loop-gain)
- [Making the inner loop fast](#making-the-inner-loop-fast)
- [Interpreting the result](#interpreting-the-result)

## What feedback adds

An exponential kernel that lets the model's own output influence its future output — the
only model in this family with temporal recursion:

```
h_fb[k] = a * exp(-k*dt / tau_fb),    k = 1 .. n_fb_bins
```

- `a` — amplitude. Its sign alone does **not** tell you whether the loop is adaptive or
  regenerative; see below.
- `tau_fb` — feedback time constant, in seconds; keep it at or above `dt`, since a time
  constant shorter than one bin is not resolvable and the search will drift there.
- `n_fb_bins` — history length, default 30 (300 ms at 10 ms bins).

Prediction becomes autoregressive and must be evaluated sample by sample:

```python
filtered = conv(stim, filt)                     # FFT circular convolution, as in LN
for t in range(T):
    drive   = filtered[t] + h_fb @ pred[t-1 : t-n_fb_bins-1 : -1]
    pred[t] = alpha * Phi(beta*drive + gamma) + epsilon
```

Parameter vector: `[numFilt, tauR, tauD, tauP, phi, alpha, beta, gamma, epsilon, a_fb, tau_fb]`.

## Never teacher-force

Teacher forcing means feeding the *observed* response into the feedback path during
fitting instead of the model's own prediction. Do not do it — not for fitting, not for
evaluation, not "just to get a quick number".

The reason is not stylistic. With the true response available one lag back, the feedback
kernel's cheapest strategy is to copy it: any smooth response is well predicted by its own
recent past, so the fit converges to a near-identity autoregression, reports a very high
R², and has learned essentially nothing about how the stimulus drives the cell. The filter
parameters then come out arbitrary, and the model fails the moment it has to run forward
on its own. A teacher-forced number is not comparable to an LN number, so it also silently
breaks any model comparison it appears in.

Free-running is more expensive and the R² is lower. That lower number is the honest one.

## Staged pipeline

### Stage 1 — filter only (6 params)

Identical to LN Stage 1: no feedback, no nonlinearity, default start plus 19 random starts,
10 restarts each. Feedback cannot help find the filter, and including it here just widens a
search space that is already the hard part.

### Stage 2 — nonlinearity + feedback (6 params, filter fixed, free-running)

Fix the Stage 1 filter, then fit `[alpha, beta, gamma, epsilon, a_fb, tau_fb]` jointly under
**free-running** loss. Even with the filter fixed, using free-running here matters: it is
what stops the nonlinearity and the feedback from splitting the work in a way that only
looks good when the true response is available.

Initialize the nonlinearity exactly as in LN Stage 2 — prefit the sigmoid to the binned
`(filter_output, response)` relationship, then refine.

One caveat specific to feedback data: the binned relationship is *smeared* here, because the
response at a given filter output also depends on its own recent history. The prefit is
still the right starting point — it is well conditioned and lands the parameters on the
right scale — but expect the binned cloud to look noisier than it does for a pure LN cell,
and do not read a poor binned fit as evidence the nonlinearity is wrong. That smearing is
the feedback, and it is what the recursion is there to explain.

The same smearing is why an LN model fitted to feedback data is the *fragile* half of this
comparison: its nonlinearity has to average over history it cannot represent, so its Stage 2
is the step most likely to collapse into a near-linear sigmoid. Give the LN arm the same
prefit and the same restart budget you give the GLM. An LN number that is low because its
own fit failed, rather than because the cell has feedback, makes the GLM look better than it
is — which is exactly the conclusion you are trying to test.

Feedback needs several starts because the sign and timescale are genuinely unknown, and
`a_fb = 0` is a saddle: with no feedback there is no gradient telling the search which
direction feedback should go.

```python
r_scale = resp.max() - resp.min()
fb_inits = [
    (0.0,             5*dt),    # no-feedback baseline; keeps an LN-equivalent in the pool
    (-0.01*r_scale,     dt),    # weak, fast
    (-0.05*r_scale,   5*dt),    # moderate, same sign as a_fb
    (+0.05*r_scale,   5*dt),    # moderate, opposite sign
    (-0.01*r_scale,  20*dt),    # weak, slow
]
```

Run 10 Nelder-Mead restarts from each of the 5 starts and keep the best.

The two signs are both worth starting from: which one is adaptive and which regenerative
depends on the sign of `alpha`, which Stage 2 has not settled yet.

These amplitudes assume the drive reaching the nonlinearity is on the order of the response
scale. If your filter output is on a very different scale (for example you normalized it to
unit SD), rescale the `a` values by `std(drive)/r_scale` — otherwise every start except
`a=0` saturates the nonlinearity immediately and the pool collapses to one usable start.

### Stage 3 — joint (all 11 params, free-running)

Concatenate Stage 1 and Stage 2 results, 10 Nelder-Mead restarts on all 11 parameters,
`maxfev = 200 * 11`, sum of squared errors on free-running predictions in raw response
units.

## Adaptive or regenerative? Use the signed loop gain

This is the single easiest thing to get backwards, so compute it rather than eyeballing
`a_fb`:

```
loop_gain = a_fb * sum_k exp(-k*dt/tau_fb) * (alpha * beta * phi(0)),   phi(0) = 0.3989
```

Note the slope term is **signed**. For an OFF-type fit `alpha < 0`, so the nonlinearity is
decreasing and a *negative* `a_fb` multiplies a negative slope — the loop gain comes out
**positive**, which is regeneration, not adaptation. Concretely: `alpha = -70`,
`a_fb = -0.0137` gives a loop gain of **+2.0**. Reading "negative `a_fb`" as "self-inhibition"
would get the biology exactly backwards.

Since `beta` and `phi(0)` are always positive, the sign reduces to `sign(a_fb * alpha)`:

| `sign(a_fb * alpha)` | loop | meaning |
|---|---|---|
| negative | adaptive | the cell suppresses its own recent output |
| positive | regenerative | recent output feeds back to amplify itself |

Report the product, or state which sign branch you canonicalized to (see SKILL.md) — `a_fb`
on its own is not interpretable, and neither is a plot of the kernel.

**Magnitude.** `|loop_gain|` above ~1 means the recursion is held in check only by the
saturating nonlinearity. Predictions can ring, latch, or become bistable. That regime is not
automatically wrong — strong feedback lives there — but it is where the free-running loss
becomes rough and restarts stop converging to the same answer, so check that the best
`a_fb` is not sitting exactly at the edge of where the loop stays bounded.

If Stage 2 or 3 produces non-finite predictions, return the large finite penalty rather than
`inf` (see SKILL.md). A fit that only works at the stability boundary is reporting a
numerical artifact.

## Making the inner loop fast

The sample-by-sample recursion is the entire cost of GLM fitting: the free-running loop
runs once per objective evaluation, and Stage 2 alone is 5 starts × 10 restarts ×
`200*6` evaluations.

Compile it. `numba`'s `@njit(cache=True)` gives roughly **88x** over pure Python here, which
is the difference between a minute per cell and an hour. What matters inside the compiled
function:

- **Ring buffer for the feedback history** rather than `np.roll` — the roll allocates a new
  array every sample and dominates the runtime.
- **Abramowitz & Stegun approximation for the normal CDF** rather than calling `scipy` from
  the inner loop; max error 7.5e-8, far below anything that affects a fit.
- **Separate `_free_run_1d` and `_free_run_2d`** functions, because numba specializes on
  array dimensionality.
- **Warm up the JIT once** on a tiny input before timing anything (~0.6 s one-off), and with
  `cache=True` later imports skip compilation entirely.

If numba is unavailable, a pure-numpy loop that keeps the history as a fixed-length array
and uses a single `np.dot` per sample is the fallback — slower, but workable if you shrink
the restart count while iterating and restore it for the final fit.

## Interpreting the result

**The bar is that the GLM does as well as the LN, not that it beats it.** The GLM nests the
LN at `a_fb = 0`, so with enough optimization effort it can always match it; what it cannot
be relied on to do is exceed it. A GLM that converges with the feedback effectively switched
off is a *correct result on a cell without feedback*, not a failed fit — report the LN and
say why. Treating "GLM > LN" as the success criterion guarantees you eventually manufacture
feedback that is not there, because in-sample R² can only rise with the extra parameters.

So the questions to ask, in order: does the GLM at least match the LN (if not, its
optimization failed — the LN is inside its search space); is the fitted loop gain
non-negligible; and does the advantage survive held-out data. Only the third is evidence.

Two results deserve suspicion:

- **GLM R² far above LN R²** — check that nothing is teacher-forced anywhere in the
  evaluation path, including any "quick check" helper.
- **GLM R² materially *below* LN R²** — not a finding about the cell. The GLM contains the
  LN, so this means the GLM fit did not converge; give it more starts before concluding
  anything.
- **`a_fb` fitted at essentially zero** — the data may genuinely be feedback-free, in which
  case report the LN model; two extra parameters that do nothing are worth dropping.
- **`tau_fb` pinned at the `dt` floor** — the same message in a different disguise. With no
  feedback signal to find, the search drives the time constant to its lower bound and
  inflates `a_fb` to compensate, so `a_fb` alone can look substantial while the loop is
  inert. On a genuinely feedback-free cell this is what "no feedback" looks like in the
  parameters. Judge it with the dimensionless loop gain, never with `a_fb` or `a_fb*alpha`,
  both of which ignore `tau_fb`.

One more thing about `a_fb`'s sign: it is only interpretable together with `alpha`. The sign
degeneracy in the parameterization (see SKILL.md) flips the filter, `alpha`, *and* `a_fb` at
once, so `a_fb > 0` with `alpha > 0` describes exactly the same loop as `a_fb < 0` with
`alpha < 0`. What is invariant is the product — see "Adaptive or regenerative?" above.

Budget roughly a minute per cell for 3 epochs × 5000 bins at 10 ms with the compiled loop.
