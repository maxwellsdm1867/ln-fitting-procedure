# The model and the staged fit, in full

Read this when you are reimplementing, porting, reviewing an implementation by hand, or need
to know exactly what a parameter means. For ordinary fitting you do not need any of it:
`scripts/matlab` implements all of it on top of CascadeGraph, `scripts/cascade_fit.py` is the
reference implementation `check_fit.py` compares against, and `check_fit.py` verifies every
convention below mechanically in about a second.

That is the point of the split. These conventions are load-bearing and easy to get wrong, but
having them verified is more useful than having them memorised.

## Contents

- [Before any of this: the recording has to declare itself](#before-any-of-this-the-recording-has-to-declare-itself) — the data contract, units, layout
- [The two building blocks](#the-two-building-blocks) — the parametric filter and the cumulative-normal nonlinearity, parameter by parameter
- [Preprocessing](#preprocessing) — decimation, mean-subtraction, what stays in native units
- [Numerical hygiene](#numerical-hygiene) — overflow forms, degeneracies, the conventions that bite
- [The staged pipeline](#the-staged-pipeline) — why the fit is staged and what each stage is for

## Before any of this: the recording has to declare itself

None of the conventions below can save a fit whose input was misread. A response stored in
volts and read as millivolts fits happily with every amplitude 1000x off, and nothing in R²
shows it; an array stored `(time x epochs)` and read the other way produces a filter that is
noise. So the declaration comes first, and it is written down rather than remembered.

The rules live in `scripts/recording.contract.json` — canonical layout and units, the required
fields, the sane band for the sampling interval, and the unit table that decides both the scale
factor and whether to rectify. They are data, applied by one function, so there is exactly one
place to change what the skill accepts:

```matlab
cascadePreflight('cell.mat')                   % can this be fitted? run before the job
proposal = cascadeInferContract('cell.mat');   % what the file can and cannot tell us
cascadeWriteContract('cell.mat', fields)       % land the confirmed answers in meta.json
```

`cascadeRecordingContract` returns one of three things, and the distinction is the point.
**Reject** lists every reason at once rather than the first. **Ask** carries an answerable
question for the fields the file is genuinely silent about — a sampling interval no array can
reveal, units where magnitude only hints, an orientation that is reliable until the array is
square. **Accept** returns a plan saying exactly what to apply. `cascadeLoadEpochs` obeys that
plan; it has no unit logic of its own, because two copies of these rules is the drift the
contract exists to remove.

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

