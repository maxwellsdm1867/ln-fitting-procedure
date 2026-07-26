# CascadeGraph parity

Read this when a Python fit has to reproduce, or be compared against, the MATLAB
CascadeGraph kernel — or when fitted parameters from two implementations disagree and you
need to find out which convention drifted.

## Reference equations (MATLAB source of truth)

### `ParamFilterNode.getFilterWithParams(params, numPoints, dt)`

```matlab
t = ((1:numPoints) * dt)';
filter = (((t./abs(params.tauR)) .^ params.numFilt) ./ (1 + ((t./abs(params.tauR)) .^ params.numFilt))) ...
    .* exp(-((t./params.tauD))) .* cos(((2.*pi.*t) ./ params.tauP) + (2*pi*params.phi/360));
filter = filter/max(abs(filter));
filter = filter - mean(filter);
```

Points worth noticing:

- `t` starts at `dt`, not at 0. An implementation using `np.arange(n)*dt` is off by one
  sample and the fitted `tauR` absorbs the difference.
- `abs(tauR)` — the rise term is sign-insensitive by construction. `tauD` is not wrapped in
  `abs` in the MATLAB, which is exactly why an unconstrained optimizer can drive it
  negative and overflow; wrapping it in Python is a safe deviation (it cannot change the
  optimum, only which paths reach it).
- `phi` enters as `2*pi*phi/360` — degrees.
- Both normalizations are applied *after* the product, in this order.

### `ParamFilterNode.processTempParams`

```matlab
filter = getFilterWithParams(params, size(stim, 2), dt);
prediction = real(ifft(fft(stim') .* fft(filter)))';
```

Circular convolution at full stimulus length, per epoch. No zero-padding, no truncation of
the filter to a shorter kernel.

### `SigmoidNlNode.processTempParams`

```matlab
out = params(1) * normcdf(params(2) .* xarray + params(3), 0, 1) + params(4);
```

`[alpha, beta, gamma, epsilon]` = `alpha * Phi(beta*x + gamma) + epsilon`.
`scipy.stats.norm.cdf` is the direct equivalent of `normcdf(·, 0, 1)`.

### `computeVarianceExplained(predicted, measured)`

```matlab
responseMean   = mean(measured, 2);
sumSquareErr   = sum(((measured - predicted).^2), 2);
sumSquareTotal = sum(((measured - responseMean).^2), 2);
rSquared = 1 - (sumSquareErr ./ sumSquareTotal);
```

Row-wise: one R² per epoch, each against its own mean. The function warns if it is handed a
matrix with more rows than columns, which is its way of catching a transposed input.

### `convolveFilterWithStim(filter, stim, hasAnticausal)`

Mean-subtracts each stimulus row before convolving. If you mean-subtract the stimulus
yourself, this is a no-op; if you do not, MATLAB and Python diverge by a constant offset
that the nonlinearity partially hides.

## Known divergence: large `numFilt`, and why the node is deliberately left alone

`ParamFilterNode.getFilterWithParams` forms `(t./tauR).^numFilt` explicitly, which overflows:

| `numFilt` | `max((t/tauR)^n)` | NaN samples in the rise |
|---|---|---|
| 4 | 3.2e+08 | 0 |
| 50 | 1.8e+106 | 0 |
| 150 | Inf | 60 of 400 |
| 250 | Inf | 400 of 400 — the whole filter |

`cascade_fit.make_filter` uses the algebraically identical reciprocal form `1/(1+(tauR/t)^n)`
and stays finite, so the two implementations genuinely differ above `numFilt ≈ 145`. The
parity check reports this as a known divergence rather than a failure.

**The node is not patched, on purpose.** The tempting fix is one line, and it is correct
arithmetic — but the region it opens up carries no information:

**`numFilt` is unidentifiable above roughly `22 * tauR/dt`.** The 10–90% rise of
`1/(1+(tauR/t)^n)` is about `tauR * 2*ln(9)/n`, so once that falls below one bin the discrete
filter stops changing. Measured at `dt = 10 ms, tauR = 30 ms`: `numFilt` 500 and 2187 give
**bit-identical** filters (max difference 0.00e+00), and 100 vs 180 differ by 2e-13. A fit
reporting `numFilt = 2187` has slid along a perfectly flat direction.

So the overflow wall sits just above a flat region it was accidentally concealing, and
removing it buys nothing. Verified directly: with the wall present the MATLAB GLM fit reaches
**R² = 0.8911**, exactly what the Python reaches without it. The wall costs no variance
explained.

**What the wall does cost is a meaningful number.** `fminsearch` converges *onto* the
boundary — probed on a smooth objective minimised at `x = 180` with a barrier at `x = 100`, it
returns `x = 100.000` from every start and reports **exitflag 1**. Worth knowing: a large
*finite* penalty behaves identically to NaN here. A constant wall gives the simplex no more
gradient than a NaN does, so the finite-loss guard elsewhere in this pipeline does not rescue
this case — it only keeps the objective arithmetic well-defined.

The fitters therefore flag it rather than route around it. Both `fit_ln`/`fit_glm`/
`fit_two_arm` and their MATLAB counterparts check `numFilt` against `22*tauR/dt` and warn:

> numFilt = 120 is above the resolvable limit for this sampling (~44 = 22*tauR/dt): the rise
> is faster than one bin, so the filter is unchanged above it and the value is a flat
> direction, not an estimate. Report it as >= 44, or refit at finer dt to resolve it.

If you ever do want the node patched — because you are fitting at a much finer `dt`, where
the resolvable limit rises above the overflow threshold and the region *does* carry
information — the change is:

```matlab
% ParamFilterNode.getFilterWithParams
filter = (1 ./ (1 + ((abs(params.tauR)./t) .^ params.numFilt))) ...
```

identical to 1.1e-16 in the normal regime. Note that it changes reported `numFilt` for any
archived fit that had stopped at the wall.

## Parity checklist

Work down this list when two implementations disagree — the failures are ordered roughly by
how often they are the culprit:

1. **`phi` in degrees, both sides.** A factor of `2*pi/360` in the phase is the single most
   common divergence.
2. **`t` starts at `dt`.** Compare `filt[0]` between implementations before comparing fits.
3. **Both filter normalizations present, in order** (`/max(abs)` then `- mean`). Check
   `max(abs(filt)) == 1` and `abs(mean(filt)) < 1e-12`.
4. **`beta*x + gamma`, not `beta*(x + gamma)`.**
5. **Circular FFT convolution at full stimulus length**, not `np.convolve`, not `'same'`
   mode, not a truncated kernel.
6. **Stimulus mean-subtracted per epoch**, response untouched.
7. **R² computed row-wise**, not on the flattened array.
8. **Same decimation** — block-mean over 100 samples versus `scipy.signal.decimate` (which
   applies an anti-alias filter) give slightly different traces and therefore slightly
   different fits.
9. **Same `dt`** in the filter as in the data: fitting decimated 10 ms data with `dt=1e-4`
   silently rescales every time constant by 100.

## The executable parity check

`scripts/parity_dump.m` and `scripts/parity_check.py` are the automated version of everything
below. Run the MATLAB half, then the Python half, and it compares filters, convolutions, LN
predictions, row-wise R² and the nonlinearity across five parameter sets including the
awkward corners:

```matlab
>> parity_dump('/path/to/cascadegraph', '/tmp/cg_reference.mat')
```
```bash
$ python parity_check.py /tmp/cg_reference.mat
```

Everything agrees to ~1e-14 except the large-`numFilt` case documented above. Run it after
touching either implementation and after any MATLAB upgrade: "the Python matches the MATLAB"
is a claim with a short shelf life, and nothing about a fitted R² reveals when it expires.

## A minimal parity test

Fix a parameter vector, generate a filter on both sides, and compare numerically rather than
by eye. `scripts/cascade_fit.py` is the Python side of this comparison — it reproduces the
MATLAB filter to 1e-16 on a fixed parameter vector, so use it as the reference rather than
whatever local implementation is in question:

```python
from cascade_fit import make_filter
f = make_filter(numFilt=4, tauR=0.025, tauD=0.045, tauP=0.065, phi=35.0,
                n_points=1000, dt=0.01)
assert abs(np.max(np.abs(f)) - 1.0) < 1e-12      # unit peak
assert abs(np.mean(f)) < 1e-12                   # zero DC
# then: max|f_local - f| and max|f_matlab - f| should both be < 1e-10
```

If the filters match to 1e-10 and the fits still disagree, the difference is in
preprocessing or the optimizer schedule, not in the model.

## Deviations that are safe

- Wrapping `tauD` (and `tauP`) in `abs()` inside the filter.
- Returning a large finite penalty instead of `NaN`/`inf` from the objective.
- Data-driven nonlinearity initialization (see SKILL.md) instead of fixed constants — the
  optimum is unchanged, only the path to it.

## Deviations that break comparability

- Normalizing or rectifying the response.
- Dropping either filter normalization.
- Reporting a pooled R² where the reference reports per-epoch R².
- Fitting on flattened epochs.
