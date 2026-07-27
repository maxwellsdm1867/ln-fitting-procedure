---
name: ln-fitting-procedure
description: >-
  Fits and checks parametric cascade models — LN, GLM with feedback, LNLN, two-arm — against
  neural stimulus-response recordings, in the MATLAB CascadeGraph parameterization, and
  enforces the data contract a recording must satisfy before it is fitted. Deliberately narrow:
  use it when asked for it by name, when asked to fit or debug a cascade/LN/GLM model in the
  CascadeGraph parameterization, or when a recording needs its meta.json contract set up or
  checked. Do not reach for it as general help with filters, units, resampling,
  receptive-field mapping, spike sorting, Hodgkin-Huxley fitting, or neural-network encoding
  models — those overlap in vocabulary and are not what this does.
allowed-tools:
  - Read
  - Write
  - Edit
  - Bash
  - Grep
  - Glob
  - AskUserQuestion
  - Agent
---

# Fitting parametric cascade models to retinal data

Every model in this family is the same two blocks — a parametric temporal filter and a
cumulative-normal nonlinearity — plus whatever comes after, and all of them are fitted the
same staged way. The staging exists because the joint loss surface is badly behaved: a filter
half a period off in phase produces a filter output no nonlinearity can rescue, so
gradient-free search from a bad start sits in a local minimum forever. Stage 1 finds the
filter while the objective is nearly linear in what matters, Stage 2 finds the nonlinearity on
a now-fixed input, Stage 3 lets them negotiate.

## Before anything: the recording has to declare itself

A fit is about ten minutes. A response left in volts, or an array stored `(time x epochs)`,
costs you all ten — and the version that does not error but quietly fits the wrong thing costs
a great deal more, because none of it shows up in R². So the declaration comes first, and it
is written down rather than remembered.

```matlab
addpath('<skill>/scripts/matlab');

cascadePreflight('cell.mat')                  % can this be fitted? run it before the job
proposal = cascadeInferContract('cell.mat');  % what the file can and cannot tell us
cascadeWriteContract('cell.mat', fields)      % land the confirmed answers in meta.json
```

`cascadePreflight` checks the declaration **and** opens the array header to check it against
what is stored, because a `meta.json` can be internally perfect and still contradict its own
recording — an array saved `(time x epochs)` while declaring the default `epochs_x_time` passes
every check on the declaration alone and then dies at load. That is the ten-minute job this is
supposed to save you, so the shape check belongs here rather than downstream.

**When `proposal.mustAsk` is non-empty, ask with `AskUserQuestion`. Do not guess, and do not
default.** Those fields are exactly the ones the file is silent about and the answer changes
the numbers:

- `sample_interval_s` — no array can reveal its own sample rate. If the file carries a scalar
  named `dt` or `Fs`, it is still only a proposal: whether that number is an interval or a rate
  is a coin flip on the name, and getting it backwards rescales the entire time axis.
- `response_units` — magnitude only hints. Non-negative integers look like a firing rate and
  also like a rectified current; a sub-1 span looks like volts and also like a small millivolt
  deflection. Volts read as millivolts is a 1000× error that fits happily and moves every
  threshold in this skill.
- `orientation` — proposed from the array shape, which is reliable until the array is square.

Each proposal carries its `evidence`; put that in the question so the answer is informed rather
than a coin flip of its own. Then `cascadeWriteContract` validates and records it, so the next
session does not ask again. `contract off` exists, but then `cascadePreflight` is what stands
between you and a wasted job.

## Start here: use the bundled implementation

This model family is genuinely sensitive to initial conditions, and no amount of documentation
fixes that. What documentation *can* do is stop the sensitivity from costing you attention:
every convention, degeneracy and optimizer quirk below is either handled silently by the
bundled fitters or reported by them automatically. Use one and your time goes to improving the
fit and understanding the data, which is the only part that needs you.

**Fit in MATLAB.** `scripts/matlab/` calls CascadeGraph's `ParamFilterNode` and `SigmoidNlNode`
directly, so the model has exactly one definition and parity is automatic rather than
maintained. That is the whole argument: a second implementation has to be *kept* correct, and
maintained agreement decays quietly. This layer adds what CascadeGraph does not provide — the
staged pipeline, the restarts, and the diagnostics.

```matlab
addpath(genpath('<cascadegraph>')); addpath('<skill>/scripts/matlab');

[stim, resp, info] = cascadeLoadEpochs('cell.mat');    % dt + units read from meta.json
out = cascadeFitLN(stim, resp, info.dt);               % LN, 9 params
% out = cascadeFitGLM(stim, resp, info.dt);            % + exponential feedback, 11 params
% out = cascadeFitTwoArm(stim, resp, info.dt);         % TwoArmLnHyperNode topology, 18 params

if ~out.diagnostics.ok, disp(out.diagnostics.warnings); end
disp(out.params); disp(out.r2PerEpoch);
```

```bash
# then always, before the numbers leave your screen -- about one second
python <skill>/scripts/check_fit.py your_fit_script.m results.json data.mat
```

`cascadeFitGLM` also returns `out.loopGain` and `out.feedbackType` — judge feedback by the
signed loop gain, never by `a_fb` alone. `cascadeFitTwoArm` returns `out.bic` and `out.lnBic`,
because R² alone always favours the bigger model.

**`scripts/cascade_fit.py` is not a second way to fit.** It is the reference implementation
`check_fit.py` calls to verify a filter numerically, and the engine behind the Stop hook's
figures — a hook runs at the end of every turn, and paying MATLAB's ~17 s startup each time is
the one place Python earns its keep. Reach for it to *check* a fit, never to produce one:
fitting there would resurrect exactly the maintained-parity problem the MATLAB path exists to
avoid.

**The loader applies the recording's declaration** rather than guessing at it. Which variables
hold stimulus and response, whether the arrays need transposing, the sampling interval, and the
response units and therefore the scale factor — all of it comes from `meta.json` via the
contract, and the setup actually used is printed in one line. It refuses outright on the
mistakes that are invisible downstream: a layout that contradicts the declaration, a `dt` that
is not an integer multiple of the sampling interval, an unreadable file. Anything unresolved is
asked rather than defaulted, because a silent default is a guess wearing a fact's clothes.
Known and checked is quiet; unknown is loud.

**`rectify` is the one field that is decided and reported but not applied.** A rate declares
`rectify=true`, and neither loader clips the response — the stored response is measured data
and clipping it would destroy real samples, so the non-negativity constraint belongs on the
model's *prediction*, which is a modelling choice the contract does not make for you. Both
loaders raise it as unresolved when it is true so it cannot pass unnoticed. If negative
predicted rates matter for your model, enforce it in the prediction yourself.

A `Stop` hook ships with the plugin and runs the check and the standard figures after any
turn that writes a `results.json`, so this happens whether or not anyone remembered. A failed
check is raised to the model, which then gets a turn to address it; everything else arrives as
a one-line notice. It says nothing at all once it has reported on that exact file.

It finds fits by walking up to three directories down from where you are, so `out/`,
`results/` and `analysis/<cell>/` all work, and it reports on **every** results file that is
new since it last looked — fit six cells in one turn and you get six checks, not one. What it
looks for is what the fitters already write: `results.json` (or `fit_results.json` /
`ln_fit.json`) carrying a `params` block, a `*fit*.py` beside it or at the top level, and
`data.npz` / `cell.npz` / `data.mat` / `cell.mat` next to its `meta.json` for the figures.
Anything it cannot do it says out loud — a missing `meta.json`, a `.m` fitter it cannot read,
a cap it hit — because a check that skips quietly is worse than no check.

**Run `check_fit.py` every time — it is on by default, not on request.** It takes about a
second, so there is no fit cheap enough to justify skipping it and no fit important enough to
report without it. Run it:

- after any fit, before the numbers are written down or pasted into a message;
- on any script you did not write yourself — inherited, ported, or sent to you;
- after editing a fitting script, however small the edit;
- before a write-up, a figure, or a claim about a cell.

It walks the whole list — filter construction against the reference, nonlinearity grouping,
the three convolution conventions, causality, preprocessing, per-epoch vs pooled R², starts
and convergence, `numFilt` identifiability, and whether the reported parameters actually
reproduce the reported R² — and returns PASS/FAIL/SKIP with the measurement attached. It is
regression-tested against 13 single-mistake variants: each flags exactly its own mistake, and
a correct script comes back clean, so a FAIL means something.

If the fit came from `fit_ln`/`fit_glm`/`fit_two_arm`, `res["diagnostics"]` has already run
the per-fit half automatically; `check_fit.py` additionally checks the *code*, which is where
convention drift lives.

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

**Python** — numpy and scipy, 3.9 through 3.13, needed only for `check_fit.py` and the Stop
hook's figures. See [`references/python-reference-impl.md`](references/python-reference-impl.md)
if you are working on the checker itself.

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

## The model and the staged fit

The filter and nonlinearity definitions, the time-axis convention, preprocessing, numerical
hygiene and the three fitting stages are in `references/implementation.md`. You need them when
reimplementing, porting, or reading someone's fit by hand — not for ordinary fitting, where
the module does all of it and `check_fit.py` verifies it.

The one thing worth carrying in your head: **the conventions are load-bearing and invisible.**
`t` starts at `dt`; the filter is unit-peak then zero-DC; `phi` is in degrees; the
nonlinearity is `beta*x + gamma`; convolution is circular, at full stimulus length, per epoch;
R² is row-wise per epoch. Every one of them produces a plausible fit when wrong, and every one
is checked by `check_fit.py`.

## Standard figures

Always the same six panels, in the order you need them, so you learn where to look instead of
re-reading an ad-hoc figure each time:

```bash
python <skill>/scripts/make_figures.py results.json data.npz fit_figures.png   # ~3 s
```

1. **input** — the stimulus, so you can see what the cell was shown
2. **output** — the response with the prediction on top
3. **filter** — fitted, with the cross-correlation estimate overlaid. Disagreement here means
   a local minimum whatever the R² says
4. **nonlinearity** — binned data with the fitted sigmoid. Sitting in one saturated tail, or
   running straight through without curving, means the cell was never driven across its range
5. **residuals vs prediction** — structure means something systematic is missed
6. **per-epoch R²** — the spread. One epoch far below the rest is usually a bad trial

The hook draws these automatically after a fit.

## Reporting variance explained

CascadeGraph's `computeVarianceExplained` is **row-wise**: one R² per epoch,
`1 - SSE_epoch / SST_epoch`, using that epoch's own mean.

Report the per-epoch values and their mean. A single R² computed on the flattened
concatenation is a different quantity — it credits the model for across-epoch mean
differences — and the two can rank models differently. If you show both, label which is
which; quietly mixing a pooled number in a figure with per-epoch numbers in a table is how
model comparisons go wrong.

## Reviewing a fit — delegate it, and never refit to check

**Reviewing is the agent's job, not yours. Hand it over.** Any time you are asked to review,
check, sanity-check or sign off a fit — and any time you have just produced one that is about
to be reported — spawn `cascade-fit-review` rather than working through the checks in this
context:

```
Agent(subagent_type="cascade-fit-review",
      prompt="Review the fit in <dir> before it is written up: <script>, <results>, <data>.")
```

Do this even when you could do it yourself, and especially then. You produced the fit, so you
have absorbed its assumptions: you already believe the filter is right because you wrote it,
and that belief is invisible to you. A reviewer with an empty context does not have it. This
is the same reason a second person reads a manuscript.

It is also cheaper. The agent runs `check_fit.py` and reports; it does not refit, and it does
not re-derive what you already know. Reviewing inline reliably costs minutes and, measured on
a script with eight planted mistakes, caught fewer of them (6-7 of 8, a different subset each
run) than the one-second script catches deterministically (8 of 8).

Reserve inline review for when there is genuinely no agent available.

Either way, the mechanical part is one command and about a second:

```bash
python <skill>/scripts/check_fit.py their_fit.py results.json data.npz
```

It walks the whole list — filter construction against the reference, nonlinearity grouping,
the three convolution conventions, causality, preprocessing, per-epoch vs pooled R², starts
and convergence, `numFilt` identifiability, and whether the reported parameters reproduce the
reported R² — and returns PASS/FAIL/SKIP with the measurement attached.

If you do review inline, hold to the same rule the agent does: **verification must be cheap.**
Re-running the optimizer to see whether you get the same answer is the expensive way to learn
almost nothing — it costs minutes, it conflates "the fit is wrong" with "the search is
stochastic", and it is not what any of these failure modes need. Every one of them is
detectable from the artifacts you already have:

| question | cheap check (seconds) | not this |
|---|---|---|
| do the reported parameters mean anything? | `roundtrip(params, stim, resp, dt, r2)` | refit and compare |
| is the convolution right? | `verify_convolution(their_conv)` | refit with a different conv |
| is the filter built right? | diff their filter against `make_filter` on fixed parameters | refit |
| is it causal? | `causality_check(filt)` — one impulse | inspect predictions by eye |
| did the optimizer converge? | read `res.status` / `exitflag` they already recorded | rerun to see if it moves |
| is it a real optimum? | `local_optimality(loss, params)` — 2N loss evaluations | multi-start comparison |
| is `numFilt` meaningful? | compare against `22*tauR/dt` | sweep it |

A refit belongs in a review only when a cheap check has already failed and you need to show
what the right answer looks like. Reach for it last, not first.

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

- **The model and the staged fit in full** — filter and nonlinearity definitions, the time
  axis, preprocessing, numerical hygiene, and Stages 1-3: read `references/implementation.md`.
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
