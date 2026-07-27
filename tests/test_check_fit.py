"""Regression test for check_fit.py: one mistake per variant, flagged exactly.

    python tests/build_check_variants.py && python tests/test_check_fit.py

Every variant differs from a known-correct hand-rolled fitting script by exactly one
mistake. The checker must flag that one and nothing else -- a checker that fires on
everything is as useless as one that fires on nothing. Both failure directions matter,
which is how two real bugs were found: "filter unit peak" compared against exactly 1
when the reference itself peaks at 0.9985 (unit-peak normalisation precedes the zero-DC
subtraction), and a start loop of range(0) was rescued by a fallback that only looked
for the textual presence of random draws.
"""
"""Each variant has exactly ONE mistake. The checker must flag that one and nothing else."""
import glob, re, subprocess, sys

import os
CHECK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "skills", "ln-fitting-procedure", "scripts", "check_fit.py")
EXPECT = {  # variant -> the check name that MUST fail
 "00_correct":        None,
 "01_t_origin":       "filter t origin",
 "02_phi_radians":    "phi in degrees",
 "03_no_unit_peak":   "filter unit peak",
 "04_no_zero_dc":     "filter zero DC",
 "05_nl_grouping":    "nonlinearity grouping",
 "06_zscored_resp":   "response native units",
 "07_rectified":      "no rectification",
 "08_causal_conv":    "conv circular_not_causal",
 "09_truncated_conv": "conv filter_full_length",
 "10_overflow_form":  "filter overflow form",
 "11_single_start":   "multiple starts",
 "12_no_stim_meansub":"stimulus mean-subtracted",
}
rows=[]
for name in sorted(EXPECT):
    out = subprocess.run([sys.executable, CHECK,
                            os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         f"{name}.py")], capture_output=True, text=True).stdout
    fails = set(re.findall(r"FAIL\s+(.+?)\s\s+", out))
    fails = {f.strip() for f in fails}
    want = EXPECT[name]
    # "filter matches reference" is an aggregate: it fires for any filter-shape change, expected
    aggregate = {"filter matches reference"}
    extra = fails - aggregate - ({want} if want else set())
    caught = (want in fails) if want else True
    rows.append((name, want, caught, sorted(extra)))
print(f"{'variant':22s}{'expected fail':26s}{'caught':8s} false positives")
ok=True
for name, want, caught, extra in rows:
    flag = "yes" if caught else "NO"
    if not caught or extra: ok=False
    print(f"  {name:20s}{str(want):26s}{flag:8s}{extra if extra else '-'}")
print("\n", "ALL CORRECT" if ok else "PROBLEMS ABOVE")
