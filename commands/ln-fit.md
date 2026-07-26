---
description: Fit or check a cascade model (LN / GLM / two-arm) in the CascadeGraph parameterization, or set up a recording's data contract.
---

Use the `ln-fitting-procedure` skill for this request.

$ARGUMENTS

The skill's description is deliberately narrow, because its vocabulary — filters, units,
resampling — overlaps heavily with ordinary ephys talk, and a skill that interrupts is worse
than one you have to name. This command is the front door: invoking it is an explicit
instruction to load the skill and follow it.

Whatever the request, two things come before fitting:

1. **The recording must declare itself.** Run `cascadePreflight` on the data. If the contract
   is unsatisfied, use `cascadeInferContract` to propose what the file can tell us, ask the
   scientist about `proposal.mustAsk` with AskUserQuestion — never guess a sampling interval or
   a unit — and record the answers with `cascadeWriteContract`. A response left in volts fits
   happily with every amplitude 1000x off and nothing in R² shows it.

2. **The fit is checked, not just reported.** Run `check_fit.py` before any number leaves the
   screen. It takes about a second, and it catches the failure modes a good R² hides.
