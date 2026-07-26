# Validating the pipeline

Read this when you are standing up a new protocol, a new recording length, or a new model in
the family — not for every cell. It answers a different question from the per-fit checks in
`SKILL.md`: not "is this fit good" but "can this pipeline, on this kind of data, find the
right answer at all".

The recipe is Wilson & Collins (2019), *eLife* 49547, "Ten simple rules for the computational
modeling of behavioral data", adapted to the CascadeGraph model set. This project implements
it in `tools/recovery_benchmark.py`.

Everything above checks a single fit. Before trusting a *protocol* — a new stimulus, a new
recording length, a new model in the family — validate the pipeline itself. The standard
recipe is Wilson & Collins (2019), *eLife* 49547, "Ten simple rules for the computational
modeling of behavioral data"; `tools/recovery_benchmark.py` in this project implements it for
this model set. Two tests, and they answer different questions.

**Parameter recovery — "can the fit find it?"** Simulate from known parameters drawn over a
*wide* range (not a narrow plausible one — you are mapping where recovery works, not
confirming it does), refit, and compare recovered against simulated per parameter. Look at
the scatter, not just a correlation coefficient: recovery often holds in one regime and fails
in another, and a single number hides that. Then correlate the *recovered* parameters against
each other. Your simulated parameters were drawn independently, so any correlation among the
recovered ones is the model trading parameters off — those two are not separately
identifiable at this SNR, and reporting them as independent findings is a mistake.

**Model recovery — "if I pick the winner, is it the right one?"** Simulate from every model in
the family, fit every model to each simulation, and record which wins by BIC. Rows are the
generating model, columns the winner: `p(fit = B | simulated = A)`. A diagonal matrix means
this protocol can distinguish these models. Off-diagonal mass means it cannot, and no amount
of care fitting real data will fix that — the design has to change.

Then invert it. The confusion matrix gives `p(fit | simulated)`, but the question you have in
front of real data is the reverse: the GLM won, so what generated it? With a flat prior over
generating models, `p(simulated = A | fit = B)` is the normalized column. The two matrices
differ unless recovery is perfect, and the inverted one is usually the sobering one.

A caution specific to this family: model recovery depends strongly on the simulating
parameters. The two-arm model collapses to a single-arm LN whenever one arm dominates
(measured here: identifiability peaks when the arms' outputs are comparable in SD, and a
single-arm LN reaches R² 0.94+ of the two-arm signal when either arm is 3x the other). A
confusion matrix built in that regime will look alarming and mean nothing. Simulate in the
regime your real fits actually land in.

