---
name: ln-fitting-procedure
description: >-
  Fitting and debugging parametric cascade models of neural stimulus-response data — LN, GLM
  with feedback, LNLN, two-arm — in the CascadeGraph parameterization: parametric temporal
  filter, cumulative-normal nonlinearity, staged Nelder-Mead pipeline with random restarts,
  free-running (never teacher-forced) feedback, per-epoch variance explained. Use it whenever
  someone is fitting a temporal filter plus a static nonlinearity to a recording, or
  diagnosing one that misbehaves — "fit an LN model to this cell", "recover the temporal
  filter", "near-zero variance explained", "the filter looks wrong", "alpha came out the wrong
  sign", "why is my EV so low on the parasols", "add spike-history feedback", "port this from
  the MATLAB CascadeGraph code" — for cone, horizontal, bipolar, amacrine, or parasol/midget
  RGC data, and even when "LN model" is never said. Not for generic signal processing or
  resampling, receptive-field mapping, spike sorting, Hodgkin-Huxley fitting, or
  neural-network encoding models.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
  - AskUserQuestion
---

# Fitting parametric cascade models to retinal data

Every model in this family is the same two blocks — a parametric temporal filter and a
cumulative-normal nonlinearity — plus whatever comes after, and all of them are fitted the
same staged way. The staging exists because the joint loss surface is badly behaved: a filter
half a period off in phase produces a filter output no nonlinearity can rescue, so
gradient-free search from a bad start sits in a local minimum forever. Stage 1 finds the
filter while the objective is nearly linear in what matters, Stage 2 finds the nonlinearity on
a now-fixed input, Stage 3 lets them negotiate.

## Start here: use the bundled implementation

This model family is genuinely sensitive to initial conditions, and no amount of documentation
fixes that. What documentation *can* do is stop the sensitivity from costing you attention:
every convention, degeneracy and optimizer quirk below is either handled silently by the
bundled fitters or reported by them automatically. Use one and your time goes to improving the
fit and understanding the data, which is the only part that needs you.

There are two, and they agree to ~1e-6 on every fitted parameter on the same data.

**MATLAB** — `scripts/matlab/`. Calls `ParamFilterNode` and `SigmoidNlNode` directly, so the
model has exactly one definition and parity is automatic rather than maintained. This is the
layer CascadeGraph does not provide: the staged pipeline, restarts, and the diagnostics.

```matlab
addpath(genpath('<cascadegraph>')); addpath('<skill>/scripts/matlab');

[stim, resp, info] = cascadeLoadEpochs('cell.mat');    % dt + units read from meta.json
out = cascadeFitLN(stim, resp, info.dt);               % LN, 9 params
% out = cascadeFitGLM(stim, resp, info.dt);            % + exponential feedback, 11 params
% out = cascadeFitTwoArm(stim, resp, info.dt);         % TwoArmLnHyperNode topology, 18 params

if ~out.diagnostics.ok, disp(out.diagnostics.warnings); end
disp(out.params); disp(out.r2PerEpoch);
```

`cascadeFitGLM` also returns `out.loopGain` and `out.feedbackType` — judge feedback by the
signed loop gain, never by `a_fb` alone. `cascadeFitTwoArm` returns `out.bic` and `out.lnBic`,
because R² alone always favours the bigger model.

**Python** — `scripts/cascade_fit.py`, checked against the MATLAB kernel to 1e-14.

```python
import sys; sys.path.insert(0, "<skill>/scripts")
from cascade_fit import load_epochs, fit_ln

stim, resp, info = load_epochs("data.npz")     # sampling interval + units from meta.json
res = fit_ln(stim, resp, dt=info["dt"])

if info["unresolved"]: print(info["unresolved"])
if not res["diagnostics"]["ok"]: print(res["diagnostics"]["warnings"])
print(res["params"], res["r2_per_epoch"], res["r2_mean"])
```

**The loader infers the setup** rather than asking you to restate it: sampling interval and
response units come from the recording's own `meta.json`, the setup used is printed in one
line, and it refuses outright on the mistakes that are invisible downstream — a transposed
`(time x epochs)` array, a `dt` that is not an integer multiple of the sampling interval, an
unreadable file. Anything it cannot resolve — units missing or unrecognised, no metadata at
all — is reported rather than defaulted, because a silent default is a guess wearing a fact's
clothes. Known and checked is quiet; unknown is loud.

**The diagnostics are the point.** Every fit automatically checks whether the optimizer
converged, whether independent starts agreed, whether it is even at a local minimum, whether
the reported parameters reproduce the fit, whether the filter is causal, whether it matches
the cross-correlation estimate, and whether one epoch is dragging the rest. `ok` true with no
warnings means the mechanical failure modes are ruled out and what is left is science. A
warning is plain language and says what to do; `references/diagnostics.md` has the detail.

Both fitters also handle the staged pipeline, random restarts, the binned-nonlinearity prefit,
the finite-loss guard, `tauP` de-aliasing and all four sign degeneracies. Roughly 25 s for an
LN fit on 3 epochs x 1000 bins in either language.

Every convention below is a chance to drift, and a drifted convention produces parameters
that silently mean something else — so reimplement only when you need a variant neither
fitter covers, and read the rest of this file when you do.

## What it needs to run

Deliberately small, because a fitting pipeline that silently requires a licence is one that
does not run on a colleague's machine.

**MATLAB** — base MATLAB only, plus CascadeGraph for the model nodes. No toolboxes: `normcdf`
and `corr` are replaced by `cascadeNormcdf` (erfc, verified identical) and `cascadeCorr`.
Needs R2016b or later for `jsondecode` and implicit expansion. Verified on R2022a from a
`restoredefaultpath` with only CascadeGraph and `scripts/matlab` added.

```matlab
addpath(genpath('<cascadegraph>'));
addpath('<skill>/scripts/matlab');
```

**Python** — numpy and scipy. `numba` is optional and only accelerates the GLM inner loop;
without it the pure-numpy path uses the same exact O(1) recursion and is perfectly usable
(~20 s for a GLM fit). Parses and runs on 3.9 through 3.13. The Python needs **no** MATLAB and
no CascadeGraph — it implements the model itself and is checked against the MATLAB separately.

```python
import sys; sys.path.insert(0, "<skill>/scripts")
```

**Cross-checking the two** (`scripts/parity_dump.m` + `scripts/parity_check.py`) is the only
thing that needs both, plus `scipy.io` to read the reference file.

## Before you fit: agree what "good" means

**Ask what R² they expect for this cell type and this stimulus, before fitting anything.** In
an interactive session use the `AskUserQuestion` tool rather than burying the question in
prose — one question, a few plausible ranges for this preparation plus an "I don't know"
option. It takes the scientist seconds and it changes how every number afterwards is read:

```
AskUserQuestion({questions: [{
  header:   "Expected R²",
  question: "What per-epoch R² would you expect for an Off parasol under Variable Mean Noise?
             It decides whether this fit is good or broken.",
  options: [
    {label: "~0.6-0.7",  description: "Typical for parasols in our hands"},
    {label: "~0.8-0.95", description: "More like horizontal or bipolar cells"},
    {label: "~0.3-0.5",  description: "Expect a lot of unexplained variance here"},
    {label: "Not sure",  description: "Use the noise ceiling and a model-free bound instead"}
  ]}]})
```

Achievable R² varies a great deal across stimulus types and cell types — a number that means
"excellent" for one protocol means "something is broken" for another — so there is no built-in
threshold worth trusting. Asking also converts a vague "the fit looks bad" into a decidable
question.

If you cannot ask — a batch job, a one-shot invocation, nobody at the keyboard — do not
quietly skip it. **Say in the write-up what you compared against and that no expectation was
supplied**, e.g. "no target R² was given; this is 0.885 against a trial-to-trial noise ceiling
of 0.884." A bare R² with no reference point is not interpretable by whoever reads it next,
and the omission is invisible unless you name it.

For calibration only, these were measured under **one** protocol (Variable Mean Noise,
ConeResponseFull), per-epoch R²:

| Cell type | Linear | LN |
|-----------|--------|-----|
| Horizontal | ~80% | ~95% |
| Off bipolar | ~55% | ~75% |
| On bipolar | ~60% | ~70% |
| Off parasol | ~20% | ~65% |

Do not port these to a different stimulus. They are one protocol's numbers, recorded here so
you can see the *spread* across cell types — roughly 30 points between horizontal cells and
parasols — not so you can test against them.

What generalizes instead of an absolute number:

- **The noise ceiling.** With repeated trials, the trial-to-trial reliability bounds any
  model. R² near that ceiling is a good fit whatever its absolute value.
- **A model-free ceiling.** The binned nonparametric estimate (`nonparametric_filter` +
  `binned_nl`) upper-bounds any LN model with a linear front end. Compare against that.
- **The same cell across conditions**, or the same protocol across cells. Relative
  comparisons survive; absolute thresholds do not.

## The two building blocks

Match these definitions exactly — they are what CascadeGraph's `ParamFilterNode` and
`SigmoidNlNode` compute, and downstream parameter values are only comparable across
fits if the parameterization is identical.

**Filter (5 params).**

```python
t = np.arange(1, n_points + 1) * dt                  # starts at dt, not at 0
rise = 1.0 / (1.0 + (abs(tauR) / t) ** numFilt)      # == (t/tauR)^n / (1 + (t/tauR)^n)
f = rise * np.exp(-t / abs(tauD)) * np.cos(2*np.pi*t/tauP + 2*np.pi*phi/360)
f = f / np.max(np.abs(f))     # unit peak
f = f - np.mean(f)            # zero DC
```

Two details in that first pair of lines are worth not rediscovering. `t` starts at `dt`; a
filter built on `np.arange(n)*dt` is shifted by one sample, and the fitted `tauR` quietly
absorbs the difference, so two implementations that disagree only here will report different
parameters for the same data. And the rise term is written as a reciprocal because fits
routinely drive `numFilt` into the hundreds — the rise becomes a step, which is a legitimate
solution — and at that point `(t/tauR)**numFilt` overflows to `inf`, making `rise/(1+rise)`
NaN. The reciprocal form saturates to 0 or 1 instead and keeps the objective finite.

- `numFilt` — filter order, controls how sharply the rise turns on
- `tauR` — rise time constant (seconds)
- `tauD` — decay time constant (seconds)
- `tauP` — oscillation period (seconds)
- `phi` — phase offset in **degrees**; the `/360` in the formula is the giveaway. Treating
  it as radians silently rescales the search space and is one of the most common reasons a
  filter fit stalls.

The two normalization lines are easy to drop and they matter. Without `f/max|f|` the filter
amplitude and the gain downstream (`scFact` in Stage 1, `alpha*beta` later) are the same
degree of freedom, and the simplex wanders along that ridge instead of shaping the filter.
Without `f - mean(f)` the filter carries DC, so it can explain the response mean — which is
`epsilon`'s job — and the two fight.

**Nonlinearity (4 params).** `y = alpha * Phi(beta*x + gamma) + epsilon`, where `Phi` is the
standard normal CDF (`scipy.stats.norm.cdf`).

It is `beta*x + gamma`, **not** `beta*(x + gamma)`. Both are sigmoids, so a wrong grouping
still fits *something* — it just fits a different model, and `gamma` is then on the wrong
scale, which makes the reported threshold meaningless and any cross-cell comparison invalid.

**Convolution.** Circular, via FFT, per epoch, with the filter generated at the full
stimulus length (not truncated and zero-padded):

```python
x = np.real(np.fft.ifft(np.fft.fft(stim, axis=1) * np.fft.fft(filt)[None, :], axis=1))
```

**Verify it rather than reading it.** `verify_convolution` (Python) and `cascadeVerifyConv`
(MATLAB) probe all three properties of any convolution you did not get from this module, and
report *which* one is wrong rather than that the numbers differ:

```python
from cascade_fit import verify_convolution
verify_convolution(my_conv)     # PASS/FAIL per property, with the measurement
```
```matlab
cascadeVerifyConv(@myConv)
```

Each probe is built so a correct and an incorrect implementation differ qualitatively: an
impulse in the last sample must wrap to the start (circular, not causal); a filter tap at a
long lag must contribute (full length, not truncated); and with one epoch driven and the next
silent, the silent one must be exactly zero (per-epoch, not flattened). Checked against the
four common mistakes, each fails exactly the property it violates:

| implementation | circular | full length | per-epoch |
|---|---|---|---|
| reference | pass | pass | pass |
| `np.convolve` / `conv` truncated to N | **fail** | pass | pass |
| kernel truncated to 60 taps | pass | **fail** | pass |
| epochs flattened into one vector | pass | pass | **fail** |
| `mode='same'` | **fail** | **fail** | pass |

Time-domain causal convolution gives a slightly different answer at the edges and will not
reproduce reference fits. The wrap-around this implies — the end of an epoch's stimulus
acting as history for its first bins — is deliberate, not a bug to be fixed with zero
padding. Padding changes the model, so a fit that pads no longer reproduces or compares
against anything fit the reference way. If edge effects worry you, drop the first
`~3*tauD/dt` bins from the *loss*, and say so; do not change the convolution.

### The time axis — get this right before anything else

This is the most common way a fitting session silently goes wrong, and it produces a model
that predicts the response from the *future* stimulus while still reporting a respectable
R². Fix the convention once, then check it, every session:

```
index:   0      1      2    ...        N-2    N-1
lag:   +1*dt  +2*dt  +3*dt  ...       -2*dt  -1*dt
        \___________ past ___________/  \_ future _/
```

`filt[k]` multiplies `stim[t-k-1]`: **early indices are the causal part, and the tail of the
array wraps around to negative lag.** The parametric filter is causal by construction —
it is only defined for `t = dt, 2*dt, ...` — so if you build it with `make_filter` you
cannot get this wrong. You can get it wrong everywhere else.

**Verify it, don't assume it.** One impulse settles it in three lines:

```python
imp = np.zeros((1, N)); imp[0, 100] = 1.0
y = circular_conv(imp, filt)
print(np.argmax(np.abs(y[0])) - 100)     # must be POSITIVE: response follows stimulus
```

On the reference implementation this prints `+2` for a filter peaking at 20 ms. A negative
number means the kernel is reversed and the model is reading the future.

**The three ways it breaks:**

- `np.convolve(s, f, mode="same")` **centers** the kernel, so half of it acts at negative
  lag. `mode="full"` then truncating to `[:N]` is causal; `"same"` is not.
- Flipping the kernel — `f[::-1]`, or a `correlate` where a `convolve` was meant — reverses
  time outright. In the check above this turns `+2` into `-3`.
- Reading `computeFilter`'s output wrong. It returns **both** halves:
  `filterCausal = filterFull(1:filterPts)` from the *start* of the array,
  `filterAnticausal = filterFull(end-filterPts+1:end)` from the *end*. The anticausal half is
  not noise — it is structure from autocorrelation in the stimulus — but it is not the
  filter, and it looks like a plausible filter if you plot it alone.
  `convolveFilterWithStim`'s `filterHasAnticausalHalf` flag exists precisely because a filter
  carrying both halves must be zero-padded *in the middle*, not at the end.

One thing not to worry about: `f - mean(f)` leaves a small constant (~1.5e-3) across every
lag, including negative ones. Against a mean-subtracted stimulus its contribution is
identically zero — the filter's DC multiplies the stimulus's DC — measured at 1e-15. The
mean subtraction is not smuggling in an acausal term.

## Preprocessing

- **Mean-subtract the stimulus per epoch.** The filter is DC-free, so a stimulus mean only
  adds a constant the nonlinearity has to absorb.
- **Leave the response alone.** Do not z-score it, do not rectify it, do not baseline-shift
  it. `alpha` and `epsilon` are in the response's native units (mV, pA) and that is what
  makes fitted parameters comparable across cells. Rectifying a membrane-voltage trace
  clips the lower half of the sigmoid, so `alpha` and `gamma` become unidentifiable.
- **Rectify only for rates.** Check `meta['response_units']`; set `rectify=False` unless the
  units are `spikes/s`.
- **Decimate to 10 ms bins** (`decimation_factor=100` from 0.1 ms raw). Bin by averaging
  within each block rather than subsampling, so you keep the noise averaging.
- **Keep the epoch structure** as an `(epochs x time)` matrix. Flattening splices the end of
  one epoch onto the start of the next, and circular convolution then smears one trial's
  stimulus into another trial's prediction.

## Numerical hygiene

Nelder-Mead is unconstrained, so it will happily walk `tauD` to a negative number, at which
point `exp(-t/tauD)` overflows and the whole prediction becomes NaN. Two lines prevent most
lost afternoons:

```python
# inside the filter: use magnitudes, so a sign flip is a no-op rather than a blow-up
rise = 1.0 / (1.0 + (abs(tauR) / t) ** numFilt) ;  decay = np.exp(-t / abs(tauD))

# inside every loss: refuse to return NaN
if not np.isfinite(err):
    return 1e12          # a large finite value the simplex can retreat from
```

Returning `inf` or `nan` from the objective leaves the simplex with no gradient information
to back away with; a large finite penalty tells it "that direction is bad" and it recovers.

There is a third trap that costs whole parameters silently. **Never start a Nelder-Mead
parameter at exactly 0.** SciPy builds its initial simplex by perturbing each coordinate by
5% *relative*, and falls back to an absolute `0.00025` when the coordinate is zero. If that
parameter's natural scale is O(1), the simplex explores it four orders of magnitude too
finely, it never moves, and every restart re-inherits the dead value — with no warning and
no error. This bites `gamma` in particular: a DC-free filter convolved with a mean-subtracted
stimulus has *exactly* zero mean output, so any `gamma = -beta * x.mean()` rule initializes
it to 0.0. Either seed such parameters from a prefit (below) or pass an explicit
`initial_simplex` with a sensible absolute step per coordinate.

Worth calibrating how much this costs, because it is easy to over- or under-react. Hit in
isolation it can strand a fit several points of variance explained below where it belongs.
But once Stage 2 starts from a binned prefit, `gamma` never begins at zero and the trap
stops existing — which is the argument for the prefit, not for hand-tuning simplexes.

## The staged pipeline

### Stage 1 — filter only (6 params)

Fit `prediction = scFact * conv(stim, filter)`. No nonlinearity yet.

Default start: `[numFilt=4, tauR=0.02, tauD=0.01, tauP=0.02, phi=1, scFact=-100]`.

Then **19 more starts sampled uniformly** from:

| Parameter | Range | Units |
|-----------|-------|-------|
| numFilt | [1, 10] | – |
| tauR | [0.005, 0.1] | s |
| tauD | [0.005, 0.2] | s |
| tauP | [0.01, 0.1] | s |
| phi | [-180, 180] | degrees |
| scFact | [-500, 500] | – |

Each start gets 10 Nelder-Mead restarts and you keep the best of all 20. This stage is
where fits are won or lost. On real data the default start alone can land at **-7% variance
explained** on an Off parasol where the random-restart version reaches **65%** — the phase
term makes the surface periodic, so a start on the wrong lobe never escapes.

If you are tempted to trim the 20 starts for speed, trim restarts instead: the spread of
starting points is doing the work, the restarts only polish.

### Stage 2 — nonlinearity only (4 params, filter fixed)

Convolve the stimulus with the Stage 1 filter, then fit the sigmoid to
`(filter_output, response)`.

`scFact` is **not** carried forward. `alpha` and `beta` absorb the gain, so re-fitting it in
Stage 3 would add a redundant direction to the search. Convolve with the unscaled filter.

**Prefit to the binned nonlinearity, then refine.** Do not go straight at the raw
`(filter_output, response)` cloud from a formula-based start. Bin it first:

```python
x = conv(stim, filt).ravel()                       # filter output, unscaled filter
y = resp.ravel()
order = np.argsort(x)
xb = np.array([c.mean() for c in np.array_split(x[order], 30)])   # equal-N bins
yb = np.array([c.mean() for c in np.array_split(y[order], 30)])
```

`(xb, yb)` *is* the nonlinearity, with the noise averaged out — 30 smooth points instead of
tens of thousands of scattered ones. Fit the sigmoid to those first (least squares is fine
here; the problem is small and well conditioned), then use that solution as the start for
the full-data fit with 10 Nelder-Mead restarts.

This is what the reference implementation does — CascadeGraph's `SigmoidNlNode.fitToSample`
fits to a binned sample from `sampleNl`, not to raw traces — and it is worth doing for three
reasons. The binned objective has no local minima to speak of, so the starting point barely
matters. The resulting `gamma` is a real number rather than a formula that can land on
exactly zero and freeze (see Numerical hygiene). And `(xb, yb)` is the plot you should be
looking at anyway to see whether the fitted sigmoid actually tracks the data.

Seed the prefit from the data, not from constants, since the polarity and scale depend on
the cell:

```python
s       = np.sign(np.corrcoef(x, y)[0, 1])         # +1 ON-type, -1 OFF-type
alpha   = s * (y.max() - y.min())                  # sign follows the cell
epsilon = y.max() if s < 0 else y.min()
beta    = 2.0 / x.std()                            # sigmoid spans ~+/-2 SD of the drive
gamma   = -beta * np.median(x)                     # median, not mean: mean(x) is exactly 0
```

The historical constants `alpha = resp_min - resp_max, beta = 0.3, gamma = -2.0,
epsilon = resp_max` work when the drive happens to be O(1) and the cell is OFF-type. They
fail on ON cells (wrong `alpha` sign) and whenever the filter output is on a different
scale, which is exactly when a fit mysteriously returns a flat prediction.

If the refined `beta` comes back near zero, the sigmoid has collapsed into its linear regime
and the fit has thrown away the nonlinearity. Treat that as a failure to retry — refit from
the prefit solution with an explicit `initial_simplex` — not as a result to report.

### Stage 3 — joint (all 9 params)

Concatenate `[numFilt, tauR, tauD, tauP, phi, alpha, beta, gamma, epsilon]` from Stages 1
and 2, run 10 Nelder-Mead restarts on the full vector. Loss is sum of squared errors on raw
response values (mV or pA) — not on normalized or per-epoch-scaled values.

### Optimizer settings, every stage

```python
minimize(loss, p0, method="Nelder-Mead",
         options={"xatol": 1e-4, "fatol": 1e-4, "maxfev": 200 * n_params})
```

A "restart" means calling `minimize` again from the previous solution. The simplex collapses
as it converges, and a restart rebuilds a full-size simplex around the current best — which
is why ten cheap restarts explore more than one run with tight tolerances. Tightening
`xatol`/`fatol` past `1e-4` costs roughly 10x the time for no measurable gain.

## Reporting variance explained

CascadeGraph's `computeVarianceExplained` is **row-wise**: one R² per epoch,
`1 - SSE_epoch / SST_epoch`, using that epoch's own mean.

Report the per-epoch values and their mean. A single R² computed on the flattened
concatenation is a different quantity — it credits the model for across-epoch mean
differences — and the two can rank models differently. If you show both, label which is
which; quietly mixing a pooled number in a figure with per-epoch numbers in a table is how
model comparisons go wrong.

## Before you trust a fit

`res["diagnostics"]` has already done the mechanical part: round-trip, convergence, local
optimality, start agreement, causality, filter-versus-cross-correlation, per-epoch spread. If
it says `ok: True`, none of those is your problem and you can stop thinking about them.

What is left needs judgment, and the module cannot do it for you:

1. **Look at the nonlinearity against binned data.** You already have `(xb, yb)` from the
   Stage 2 prefit; overlay the fitted sigmoid. A sigmoid sitting entirely in one saturated
   tail, or running straight through the middle without curving, means the cell is not being
   driven across its nonlinear range — a fact about the stimulus, not a bug in the fit.
2. **Ask whether the parameters are physically sensible** for this cell type. Time constants
   pinned at an initialization bound mean the search ran out of room, and only you know
   whether a 12 ms decay is plausible here.
3. **Hold out an epoch** before claiming predictive performance. Nine free parameters on a
   smooth response can look good in-sample.
4. **Compare against what you expected** (see "agree what good means" above). A fit that is
   mechanically clean and still well below expectation is the interesting case — that is
   where the science is.

When you report parameters, say which conventions they are in (t origin, filter
normalization, degrees, `beta*x+gamma`). `cascade_fit` puts them in the CascadeGraph
conventions and resolves all four sign/aliasing degeneracies automatically, so for a fit from
this module the answer is simply "cascade_fit defaults" — but say so, because a reader cannot
tell by looking at the numbers.

## Validating the pipeline, not just the fit

Everything above checks a single fit. Before trusting a *protocol* — a new stimulus, a new
recording length, a new model in the family — validate the pipeline itself with parameter
recovery and model recovery (a BIC confusion matrix and its Bayes inversion), following
Wilson & Collins (2019), *eLife* 49547. This is a once-per-protocol activity, not a
per-cell one: read `references/validation.md`, and run
`tools/recovery_benchmark.py` in this project, which implements it for this model set.

## Implementation

Order of preference:

1. **A project-local port, if one exists** — `src/ln_model_gamma_parametric.py`
   (`GammaParametricLNModel`) and `src/glm_model_gamma_parametric.py`
   (`GammaParametricGLM`), entry point
   `model.fit_staged(stim, resp, n_restarts=10, n_random_inits=20)`. These are not present in
   every repo; check before assuming.
2. **`scripts/cascade_fit.py` bundled with this skill** — the reference implementation above.
3. **Your own**, only for a variant neither covers, and only after reading the conventions in
   this file. Cross-check it against `cascade_fit.make_filter` on a fixed parameter vector
   before trusting a single fitted number.

## Credit

The model and the fitting procedure are **Fred Rieke's**, as implemented in CascadeGraph
(`ParamFilterNode`, `SigmoidNlNode`, `LnHyperNode`, `TwoArmLnHyperNode`). The MATLAB fitters
here call those nodes directly rather than reimplementing them, so there is one definition of
the model and it is his. What this skill adds is packaging: the staged fitter in two languages,
degeneracy handling, diagnostics, and tests.

## Going further

- **Multi-component cascades** — the two-arm topology (which is *not* two symmetric LN arms),
  its exact `epsilon2`/`gamma1` degeneracy, and why the arms must be comparable in magnitude
  to be identifiable at all: read `references/two-arm-cascade.md`.

- **GLM with feedback (11 params)** — free-running prediction, feedback-kernel
  initialization, stability, and the reason teacher forcing must never be used:
  read `references/glm-feedback.md`.
- **What a diagnostic warning means** — the underlying checks, the four exact degeneracies,
  and how to tell "the optimizer stopped early" from "this parameter is not identifiable":
  read `references/diagnostics.md`.
- **MATLAB/CascadeGraph parity** — the exact reference equations, an executable cross-check
  (`scripts/parity_dump.m` + `scripts/parity_check.py`, verified to ~1e-14), and a known
  large-`numFilt` NaN bug in the MATLAB filter: read `references/cascadegraph-parity.md`.
