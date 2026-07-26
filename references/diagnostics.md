# What the diagnostics mean

`fit_ln`, `fit_glm` and `fit_two_arm` run every check below automatically and return
`res["diagnostics"]` with `ok`, a list of plain-language `warnings`, and the underlying
numbers. You do not need to remember any of this to fit a cell — read it when a warning
fires and you want to know what it is telling you, or when you are reimplementing.

| warning | what it means | what to do |
|---|---|---|
| optimizer did not converge | scipy exhausted `maxfev` rather than settling | more restarts; check for a runaway parameter |
| only 1 of N starts reached the best loss | the answer depends on the seed | add starts, and report the dispersion |
| not at a local minimum | the search stopped on a slope | more restarts; check `initial_simplex` |
| parameters do not reproduce the fit | the reported numbers are in a private parameterization | do not report them; find what differs |
| filter is acausal | the model predicts from future stimulus | see the time-axis section in SKILL.md |
| filter correlates poorly with the cross-correlation estimate | likely a local minimum | more starts |
| large per-epoch R² spread | usually one bad trial, not a bad model | inspect the outlier epoch |

### Is it actually optimized, or does it just look optimized?

This is the failure mode with no symptom. A fit that stopped early, or settled in a local
basin, still returns parameters and still reports an R² — and if that R² is in the range you
expected for the cell type, nothing looks wrong. R² cannot detect it, because the number is
plausible either way. Four checks can, and none needs ground truth:

**1. Did the optimizer say it converged?** `scipy`'s result object knows, and almost nobody
looks. `res.status == 1` means *"Maximum number of function evaluations exceeded"* — the fit
did not converge, it ran out of budget. Read `res.success` and `res.status` on the final
call of every stage and report them. A run that quotes an R² from a `success=False` fit is
quoting a number the optimizer itself disclaims.

**2. Is it even a local minimum?** Perturb each parameter by ±2% and confirm the loss goes
*up* in every direction. `cascade_fit.local_optimality(loss, params)` does this in one call.
A converged fit gives small positive changes (~+2e-4 relative); a fit that stopped early
gives a negative one — an outright downhill direction it never took.

**3. Do independent starts agree?** `cascade_fit.start_dispersion` reports how many of the 20
Stage-1 starts land within 1% of the best loss. Several agreeing is evidence of a real basin.
One start far better than the rest means your answer is whichever start got lucky, and the
number you quote depends on the seed — report the dispersion, not just the winner.

**4. Recover parameters from your own fit — the decisive one.** Simulate a response from the
parameters you just fitted, add noise matched to the residual you actually saw, and refit
with the same pipeline:

```python
resid_sd = np.std(resp - predict_ln(params, stim, dt))
rec = parameter_recovery(params, stim, dt, resid_sd, n_reps=3)
print([r["worst_rel_error"] for r in rec])
```

If the pipeline cannot recover parameters from data it generated itself, at the SNR you
actually have, then a good R² on the real data is not evidence the fit is right — it is
evidence the model is flexible. This separates "the optimizer works and the data are
informative" from "the loss surface is flat here and any of a thousand parameter sets would
score the same". Run it once per new protocol or SNR regime; it is the closest thing to a
guarantee available without ground truth.

**And compare against a ceiling.** Bin the nonparametric filter output against the response
(`nonparametric_filter` + `binned_nl`). That combination is model-free and upper-bounds what
any LN model with a linear front end can reach. A parametric fit sitting well below it is
underfitting, whatever its absolute R² looks like.

### Four degeneracies you will hit

None is a bug; all of them make raw fitted numbers look wrong when they are not, so resolve
them before reporting rather than after someone asks.

**This matters more than it sounds, because unresolved degeneracies masquerade as science.**
A parameter-recovery run on this model initially reported median relative errors of 205% for
`tauR` and 102% for `tauD` — which reads as "the rise and decay time constants are not
identifiable", a real and publishable-sounding claim. Re-scoring the *same fits* with the
signs canonicalized gave 5.5% and 2.7%. The fits had been correct the whole time. So when a
recovery test says a parameter cannot be recovered, check your canonicalization before you
believe it: the first hypothesis is bookkeeping, not identifiability.

**Sign.** `(phi + 180, alpha -> -alpha, gamma -> -gamma, epsilon -> alpha + epsilon)` flips
the filter and re-flips the nonlinearity, giving a bit-identical prediction — `a_fb` flips
too when there is feedback. A fit lands on either branch at random, so `alpha`, `gamma` and
`epsilon` are not comparable across cells until you pick one and say which.

There is only one binary choice here, which is worth being precise about because it is easy
to think there are two. Pinning `alpha > 0` and requiring the fitted filter to correlate
*positively* with the cross-correlation filter are the same act — the cross-correlation
filter is proportional to `alpha` times the filter, so either rule forces the other.
`cascade_fit.canonical_sign` takes the cheap route and pins `alpha > 0`. The consequence to
be aware of: the cell's polarity then lives in the **filter's** sign, not in `alpha`'s. An
OFF cell and an ON cell both report `alpha > 0`, with inverted filters. Do not read
`alpha < 0` as "OFF cell" unless you know the fit used the opposite convention.

**`tauR` and `tauD` signs.** `make_filter` takes `abs()` of both so the objective cannot blow
up when the optimizer walks a time constant negative — which leaves their signs entirely
unconstrained. Negating either is an exact no-op (verified at 0.0), and a fit returns
whichever sign it wandered into. `cascade_fit.canonical_time_constants` forces both positive.

**`tauP` sign.** `cos` is even, so negating **both** `tauP` and `phi` leaves the filter
bit-identical — verified at exactly 0.0 difference. Fits land on the negative-period branch
roughly half the time, and it reads as a catastrophe: a recovery test on a cell with true
`tauP = +0.0699, phi = -103.3` reported `tauP = -0.0699, phi = +103.3` and scored 7/9 on
parameter recovery until this was canonicalized. The fit was perfect; the bookkeeping wasn't.
Force `tauP > 0`, flipping `phi` with it (`cascade_fit.canonical_tauP_sign`), and do it
*before* de-aliasing, which assumes a positive period.

**`tauP` aliasing.** The filter is only ever evaluated at `t = k*dt`, so a period faster than
the sampling folds: `dt/tauP` and `1 - dt/tauP` (with `phi -> -phi`) give the same discrete
filter to machine precision. An optimizer will happily return the aliased branch —
`tauP = 0.0125 s` on a 10 ms grid, say — which reads as an implausible 80 Hz oscillation
when the resolvable equivalent is 50 ms. De-alias into the resolvable branch before
reporting; the fit is unchanged, the number becomes interpretable.

