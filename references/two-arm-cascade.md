# Multi-component cascades

Read this when a single LN stage feeds something downstream — the two-arm cascade, LNLN, or
two-arm divisive. `SKILL.md` covers the shared filter, nonlinearity, conventions and staged
fit; only what the extra stages add is here.

**The two-arm cascade is not two symmetric LN arms.** CascadeGraph's `TwoArmLnHyperNode`
wires it as one *linear* arm summed with one *nonlinear* arm, and puts a nonlinearity after
the sum:

```
prediction = NL1( filter1(stim) + NL2( filter2(stim) ) )
```

18 free parameters: `numFilt1..phi1`, `numFilt2..phi2`, `alpha1..epsilon1`,
`alpha2..epsilon2`. Arm 1 has no nonlinearity and no gain of its own.

**`epsilon2` and `gamma1` are exactly degenerate.** `epsilon2` shifts arm 2's output, which
just shifts the sum, which `gamma1` shifts back:
`(epsilon2 + d, gamma1 - beta1*d)` gives a bit-identical prediction — verified to 6e-14.
Fix `epsilon2 = 0` and let `gamma1` carry the offset. Leaving both free adds a flat ridge
that the simplex will wander along, and makes both numbers meaningless to report.

**`alpha2` is genuinely identified, and only because the filters are unit-peak.** Arm 1's
contribution has no free scale — `ParamFilterNode` normalizes to unit peak — so `alpha2`
sets the *relative* weight of the two arms against a fixed reference. Drop the `f/max|f|`
line and that anchor disappears: the arms trade amplitude back and forth and neither
`alpha2` nor the arm weighting means anything. This is the strongest reason the filter
normalization is not optional.

**Staging.** Fit the single-arm LN first and use its filter and nonlinearity as
`filter1`/`NL1`, then add arm 2 from random starts with `NL1` briefly frozen. Arm 2 is the
correction term; starting it from scratch alongside an unconverged arm 1 gives the optimizer
two ways to explain the same variance.

**Unit-SD normalization.** This project's Python multi-stage models (LNLN, two-arm additive,
two-arm divisive v2) additionally normalize each filter output to unit standard deviation
after convolution. That is a convention of those implementations rather than of the MATLAB
kernel — check which one you are reproducing before comparing fitted parameters, because it
rescales every downstream `beta`.

