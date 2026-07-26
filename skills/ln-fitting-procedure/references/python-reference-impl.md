# The Python reference implementation

Read this only if you are working on `check_fit.py`, on the Stop hook's figures, or on the
parity harness. **Nothing here is a way to fit a cell.** Fitting is `scripts/matlab/` on top of
CascadeGraph, for the reason that decided it: a second implementation has to be *kept* correct,
and maintained agreement decays quietly, whereas the MATLAB layer calls `ParamFilterNode` and
`SigmoidNlNode` directly and inherits parity instead of maintaining it.

This file exists so that fact costs one line in SKILL.md instead of a section on every load.

## Why there is any Python at all

Two jobs, both of which run somewhere MATLAB cannot cheaply go.

**`check_fit.py`** analyses a fitting script's *source* and then executes its functions to
compare them numerically against a reference filter. It needs a reference implementation in the
same language it is checking, which is what `cascade_fit.py` provides.

**The Stop hook** runs at the end of every turn. MATLAB's cold start is ~17 s on R2022a; a hook
that paid that per turn would be worse than no hook. Python starts in ~30 ms, so the hook and
the figures it draws stay in Python. That is the whole argument, and it is a latency argument
rather than a modelling one.

## Dependencies

numpy and scipy; parses and runs on 3.9 through 3.13. `numba` is optional and only accelerates
the GLM inner loop — without it the pure-numpy path uses the same exact O(1) recursion and is
perfectly usable (~20 s for a GLM fit). h5py is needed only for MATLAB v7.3 (HDF5) `.mat`
files, and its absence produces a message saying so rather than a traceback.

`cascade_fit.py` needs **no** MATLAB and no CascadeGraph — it implements the model itself, and
is checked against the MATLAB kernel to 1e-14 rather than trusted.

```python
import sys; sys.path.insert(0, "<skill>/scripts")
```

## Loading

`load_epochs` reads `.npz` and `.mat` (v7 and v7.3), dispatching on the HDF5 magic number
rather than on which exception scipy chose to raise — scipy reports a v7.3 file as
`NotImplementedError` or as `ValueError('Unknown mat file type')` depending on whether MATLAB's
512-byte userblock is present, so catching one of them silently mishandles the other.

Arrays are forced C-contiguous on load. `loadmat` returns Fortran-ordered data and `np.mean`
follows the strides, so the decimation lands a few ulp away for a `.mat` versus the identical
`.npz` — small, but it means the same recording fits to different numbers depending on which
file it arrived in, which is not a property worth having.

The Python loader does **not** implement the recording contract. The contract is MATLAB
(`cascadeRecordingContract`), and duplicating it here would recreate exactly the drift it
exists to remove. When the checker needs a recording, it is given one that has already been
loaded through the contract.

## Cross-checking against MATLAB

`scripts/parity_dump.m` + `scripts/parity_check.py` is the only thing that needs both
languages, plus `scipy.io` to read the reference file. Run it when you change either
implementation's maths — it is what turns "they should agree" into a number.
