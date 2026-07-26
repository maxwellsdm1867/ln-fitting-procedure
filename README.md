# ln-fitting-procedure

Fitting parametric cascade models — **LN**, **GLM with feedback**, and the **two-arm cascade** —
to retinal stimulus–response recordings, in the parameterization used by
[CascadeGraph](https://github.com/mardoum/cascadegraph).

CascadeGraph provides the model *nodes*. This provides the layer it doesn't: the staged
fitting pipeline, random restarts, resolution of the parameterization's exact degeneracies,
and a diagnostics block so every fit reports its own verdict. In **MATLAB and Python**,
cross-checked against each other to ~1e-13.

It is packaged as a Claude Code skill (`SKILL.md` + `references/`), but the fitters are
ordinary MATLAB and Python and are useful on their own.

---

## Get started

In your Claude Code terminal:

```
/plugin marketplace add maxwellsdm1867/ln-fitting-procedure
/plugin install ln-fitting-procedure@ln-fitting-procedure
```

That's the preferred way in. The skill is then available in every project and triggers on its
own — *"fit an LN model to this cell"*, *"my fit gives near-zero variance explained"*,
*"recover the temporal filter"*. Other install routes are under [Install](#install).

## Why it exists

This model family is sensitive to initial conditions, and no documentation fixes that. What
documentation *can* do is stop the sensitivity from costing you attention. The parameterization
has about eight conventions and four exact degeneracies, and getting any of them wrong still
produces a plausible-looking fit with a respectable R². So the design principle here is:

> **the code handles or reports every mechanical failure mode, so your time goes to the fit
> and the data.**

Concretely, in a benchmark of 27 independent agent runs against synthetic cells with known
ground truth, runs working from prose alone reported R² values of **0.878**, **0.785** and
**0.861** whose own parameters, rebuilt independently, actually produce **0.171**, **0.353**
and **−2.958**. Nothing about those outputs looked wrong. Runs using this pipeline reproduced
their reported parameters exactly, 6/6.

## Install

### As a Claude Code plugin (easiest)

This repo is its own plugin marketplace, so it installs in two commands:

```
/plugin marketplace add maxwellsdm1867/ln-fitting-procedure
/plugin install ln-fitting-procedure@ln-fitting-procedure
```

That's it — the skill is now available in every project. Update later with
`/plugin marketplace update ln-fitting-procedure`, and remove with
`/plugin uninstall ln-fitting-procedure`.

### As a plain skill (no plugin system)

Skills are discovered from `~/.claude/skills/` (every project) and
`<project>/.claude/skills/` (that project only). The skill payload lives in
`skills/ln-fitting-procedure/`, so clone the repo somewhere and link that subdirectory:

```bash
git clone https://github.com/maxwellsdm1867/ln-fitting-procedure ~/code/ln-fitting-procedure
ln -s ~/code/ln-fitting-procedure/skills/ln-fitting-procedure \
      ~/.claude/skills/ln-fitting-procedure
```

Or copy it, if you would rather not have a symlink:

```bash
cp -R ~/code/ln-fitting-procedure/skills/ln-fitting-procedure ~/.claude/skills/
```

Either way the directory name must stay `ln-fitting-procedure` — that is the skill name.

### Then

Start a new Claude Code session and it should appear in the available-skills list. It
triggers on its own when you are doing this kind of work — *"fit an LN model to this cell"*,
*"my fit gives near-zero variance explained"*, *"recover the temporal filter"*, *"the filter
looks wrong"*, *"port this from the MATLAB CascadeGraph code"*. You do not have to name it.

Wherever it lands, `<skill>` in the instructions means that directory, so
`<skill>/scripts/matlab` is `~/.claude/skills/ln-fitting-procedure/scripts/matlab`.

### Without Claude Code

Nothing here needs it. `SKILL.md` and `references/` are plain markdown and the fitters are
ordinary MATLAB and Python — clone the repo and use the two entry points below.

### Check it works before pointing it at real data

```bash
cd ln-fitting-procedure/tools && python3 generate_data.py    # synthetic cells, known truth
cd .. && python3 - <<'EOF'
import sys; sys.path.insert(0, "skills/ln-fitting-procedure/scripts")
import cascade_fit as cf
stim, resp, info = cf.load_epochs("data/off_parasol_ln/data.npz")
res = cf.fit_ln(stim, resp, dt=info["dt"])
print(res["r2_mean"], res["diagnostics"]["ok"])     # ~0.885 True, ceiling is 0.8844
EOF
```

## Quick start

### MATLAB

```matlab
addpath(genpath('<path to cascadegraph>'));
addpath('<path to this repo>/skills/ln-fitting-procedure/scripts/matlab');

[stim, resp, info] = cascadeLoadEpochs('cell.mat');   % dt + units read from meta.json
out = cascadeFitLN(stim, resp, info.dt);              % LN, 9 params
% out = cascadeFitGLM(stim, resp, info.dt);           % + exponential feedback, 11 params
% out = cascadeFitTwoArm(stim, resp, info.dt);        % TwoArmLnHyperNode topology, 18 params

if ~out.diagnostics.ok, disp(out.diagnostics.warnings); end
disp(out.params); disp(out.r2PerEpoch);
```

`cell.mat` holds `stim` and `resp` as `(epochs × time)`; a sibling `meta.json` supplies
`sample_interval_s` and `response_units`.

### Python

```python
import sys; sys.path.insert(0, "<path to this repo>/skills/ln-fitting-procedure/scripts")
from cascade_fit import load_epochs, fit_ln, fit_glm, fit_two_arm

stim, resp, info = load_epochs("cell.npz")
res = fit_ln(stim, resp, dt=info["dt"])

if info["unresolved"]:            print(info["unresolved"])
if not res["diagnostics"]["ok"]:  print(res["diagnostics"]["warnings"])
print(res["params"], res["r2_per_epoch"], res["r2_mean"])
```

## What the diagnostics check

Every fit, automatically. `ok: True` with no warnings means the mechanical failure modes are
ruled out and what's left is science.

| check | catches |
|---|---|
| `converged` | the optimizer exhausted its budget rather than settling (scipy status 1 / `exitflag 0`) |
| `startAgreement` | only one random start reached the best loss — the answer depends on the seed |
| `localOptimum` | perturbing a parameter *lowers* the loss: the search stopped on a slope |
| `roundtrip` | the reported parameters don't reproduce the reported R² |
| `causalityLagBins` | the filter is acausal — the model is predicting from future stimulus |
| `filterVsXcorr` | the fitted filter disagrees with the cross-correlation estimate |
| `numFiltIdentifiable` | `numFilt` is above the resolvable limit, so it's a bound not an estimate |
| `perEpochR2Spread` | one epoch is dragging the rest — usually a bad trial, not a bad model |

### Checking a fit — always, not on request

`check_fit.py` takes about a second and walks the whole list of ways this model family is set
up wrong. Run it after every fit, on any script you did not write, and before any write-up:

```bash
python skills/ln-fitting-procedure/scripts/check_fit.py your_fit.py results.json data.npz
```

```
FAIL  filter matches reference   max|diff| vs cascade_fit.make_filter = 1.396e+00
FAIL  phi in degrees             no /360 or deg2rad near the cosine
FAIL  nonlinearity grouping      beta*(x + gamma): gamma ends up on the wrong scale
FAIL  round trip                 reported 0.8849, rebuilds to -0.1315 (gap 1.0164)
PASS  conv circular_not_causal   ...
```

It is regression-tested against 13 variants that each differ from a correct script by exactly
one mistake: every variant flags its own mistake and nothing else, and correct scripts come
back clean. Run the suite with `python tests/build_check_variants.py && python tests/test_check_fit.py`.

### The reviewer agent

The plugin also ships a **`cascade-fit-review`** agent — a fast, read-only checker that runs
in its own context and reports what is wrong with a fit before its numbers are written up. Ask
for it by name, or just say *"review this fit"*:

```
> review this fit before I write it up
```

It runs the bundled verifiers against your code rather than reading it, and works down the
accumulated list of ways this model family is set up wrong — filter construction, convolution,
time axis, the four exact degeneracies, optimizer convergence, and how the R² is reported. It
does not refit and it does not fix anything; it takes about two minutes and tells you what to
look at.

Separately, `verify_convolution` / `cascadeVerifyConv` check any convolution you *didn't* get
from here against the three conventions — circular not causal, filter at full stimulus length,
per-epoch — and report which one is wrong. Useful when porting, or when someone else's script
disagrees with yours.

The loader is equally opinionated: it reads the sampling interval and response units from the
recording's own metadata rather than asking you to restate them, refuses transposed
`(time × epochs)` arrays and non-integer decimation outright, and *surfaces* anything it can't
resolve instead of defaulting. Known and checked is quiet; unknown is loud.

## Layout

```
.claude-plugin/               plugin + marketplace manifests
skills/ln-fitting-procedure/  the skill payload
  SKILL.md                    the procedure, conventions and rationale
  references/
  diagnostics.md              what each warning means; the four exact degeneracies
  glm-feedback.md             free-running feedback, why teacher forcing is never valid
  two-arm-cascade.md          the TwoArmLnHyperNode topology and its degeneracy
    cascadegraph-parity.md    MATLAB reference equations, parity checklist
    validation.md             parameter and model recovery (Wilson & Collins 2019)
  scripts/
    cascade_fit.py            Python: all three fitters, loader, diagnostics
    matlab/                   MATLAB: the same, calling CascadeGraph's nodes directly
    parity_dump.m             writes MATLAB reference values...
    parity_check.py           ...and checks the Python against them
tools/
  generate_data.py            synthetic cells with known ground truth
  generate_data2.py           negative control, two-arm, time-axis trap
  generate_blind.py           blind recovery cells (non-round parameters, 3 SNRs)
  score.py                    rebuild a model from reported parameters and re-score it
  score_recovery.py           parameter recovery, modulo the degeneracies
  recovery_benchmark.py       parameter + model recovery, BIC confusion matrix
```

## Validating a pipeline, not just a fit

Before trusting a new protocol, run `tools/recovery_benchmark.py`. It implements
[Wilson & Collins (2019)](https://elifesciences.org/articles/49547) for this model set:
simulate from every model, fit every model, and tabulate which wins by BIC.

Measured here, the confusion matrix is the identity — this protocol distinguishes LN, GLM and
two-arm cleanly. That is *not* trivial: on LN-generated data the three models score R² =
0.885 / 0.885 / **0.886**, so choosing by variance explained would pick the 17-parameter model
every time. BIC's complexity penalty is doing the work.

The parameter-recovery half is worth running for its own sake. On blind cells with
deliberately non-round parameters it recovers 9/9 — but the first time it ran it reported
**205% error on `tauR`**, which reads as a real claim about identifiability and was entirely an
unresolved sign degeneracy. When a recovery test says a parameter can't be recovered, suspect
your bookkeeping before your science.

## Requirements

**MATLAB** — base MATLAB plus CascadeGraph. **No toolboxes**: `normcdf` and `corr` are replaced
by `cascadeNormcdf` (erfc, verified identical) and `cascadeCorr`. R2016b or later. Verified
from a `restoredefaultpath`.

**Python** — numpy and scipy. `numba` is optional and only accelerates the GLM inner loop;
without it the pure-numpy path uses the same exact O(1) recursion and is perfectly usable.
Runs on 3.9–3.13. Needs no MATLAB and no CascadeGraph.

## Known divergence from CascadeGraph

`ParamFilterNode.getFilterWithParams` forms `(t/tauR)^numFilt` explicitly, which overflows to
`Inf` above `numFilt ≈ 145`, making the whole filter NaN. The Python uses the algebraically
identical reciprocal form and stays finite, so the two genuinely differ up there.

**The node is deliberately left unpatched.** `numFilt` is unidentifiable above roughly
`22·tauR/dt` — at `dt = 10 ms`, `numFilt` 500 and 2187 give *bit-identical* filters — so that
region carries no information and the overflow costs no variance explained (verified: the
MATLAB GLM fit reaches R² 0.8911 with the wall present, exactly what the Python reaches
without it). What it does cost is a meaningful number, since `fminsearch` converges *onto* the
boundary and reports success. So the fitters flag it rather than route around it. See
`references/cascadegraph-parity.md`.

## Acknowledgements

The model itself — the parametric temporal filter, the cumulative-normal nonlinearity, and the
cascade structure they sit in — is **Fred Rieke's**, as implemented in
[CascadeGraph](https://github.com/mardoum/cascadegraph). The fitting procedure this packages,
including the staged pipeline and the initialization strategy, likewise originates in the
Rieke lab. **All credit for the science and the model formulation goes to Fred and to the lab.**

What this repository adds is packaging around that work: a staged fitter in both languages, the
degeneracy handling, the diagnostics, and a test suite. The MATLAB side calls `ParamFilterNode`
and `SigmoidNlNode` directly rather than reimplementing them, so there is one definition of the
model and it is Fred's.

Any errors in the packaging, the diagnostics or the analysis here are mine, not theirs.

## Related

- [CascadeGraph](https://github.com/mardoum/cascadegraph) — the model nodes this builds on.

## License

MIT, matching CascadeGraph.
