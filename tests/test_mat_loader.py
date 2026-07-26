"""Regression test for .mat support in cascade_fit.load_epochs.

    python tests/test_mat_loader.py

Correctness bar: a .mat must give numbers IDENTICAL to the equivalent .npz. Everything else is
a failure path, and each one has to fail loudly with a message that names the problem -- a
loader that raises IndexError from somewhere inside scipy has technically refused the file and
practically wasted an hour.

Both v7.3 layouts are exercised. scipy reports a v7.3 file as NotImplementedError or as
ValueError('Unknown mat file type') depending on whether MATLAB's 512-byte userblock is
present, which is why the reader sniffs the HDF5 magic number instead of dispatching on the
exception type.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from scipy.io import savemat

try:
    import h5py
except ImportError:                                          # pragma: no cover
    h5py = None

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "skills", "ln-fitting-procedure", "scripts")
sys.path.insert(0, SCRIPTS)
import cascade_fit as cf  # noqa: E402

DT, RAW_DT, N_EP, N_RAW = 0.01, 1e-4, 3, 60000
META = {"sample_interval_s": RAW_DT, "response_units": "mV", "n_epochs": N_EP}
TRUE = dict(numFilt=4.0, tauR=0.025, tauD=0.05, tauP=0.065, phi=-140.0,
            alpha=55.0, beta=1.0, gamma=0.4, epsilon=-38.0)

rows = []


def check(name, ok, detail=""):
    rows.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}  {name:38s} {detail}")


def raises(fn, *want):
    try:
        fn()
        return False, "did NOT raise"
    except Exception as e:                                   # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        missing = [s for s in want if s.lower() not in msg.lower()]
        if missing:
            return False, f"raised but message lacks {missing}: {msg[:110]}"
        return True, msg.split("\n")[0][:100]


def make_recording(seed=0):
    """A synthetic cone->RGC recording, generated through the bundled fitter itself."""
    rng = np.random.default_rng(seed)
    stim = rng.standard_normal((N_EP, N_RAW))
    dec = int(round(DT / RAW_DT))
    coarse = stim[:, :N_RAW // dec * dec].reshape(N_EP, -1, dec).mean(axis=2)
    coarse = coarse - coarse.mean(axis=1, keepdims=True)
    resp = cf.predict_ln(TRUE, coarse, DT)
    resp = np.repeat(resp, dec, axis=1)[:, :N_RAW]
    return stim, resp


def bed(root, name, stim, resp, fmt, meta=META):
    p = os.path.join(root, name)
    os.makedirs(p, exist_ok=True)
    if fmt == "npz":
        f = os.path.join(p, "data.npz")
        np.savez(f, stim=stim, resp=resp)
    elif fmt == "v7":
        f = os.path.join(p, "cell.mat")
        savemat(f, {"stim": stim, "resp": resp})
    else:                                                    # v73 / v73ub
        f = os.path.join(p, "cell.mat")
        kw = {"userblock_size": 512} if fmt == "v73ub" else {}
        with h5py.File(f, "w", **kw) as fh:
            # MATLAB is column-major, so HDF5 holds (epochs x time) with dims reversed.
            fh.create_dataset("stim", data=np.asarray(stim).T)
            fh.create_dataset("resp", data=np.asarray(resp).T)
        if fmt == "v73ub":
            with open(f, "r+b") as fh:
                fh.write(b"MATLAB 7.3 MAT-file, Platform: TEST")
    if meta is not None:
        json.dump(meta, open(os.path.join(p, "meta.json"), "w"))
    return f


def main():
    if h5py is None:
        print("h5py not installed -- v7.3 cases cannot run")
        return 1
    tmp = tempfile.mkdtemp()
    try:
        stim, resp = make_recording()

        print("--- 1. the numbers must be identical across formats ---")
        s_npz, r_npz, _ = cf.load_epochs(bed(tmp, "npz", stim, resp, "npz"), verbose=False)
        for fmt in ("v7", "v73", "v73ub"):
            s, r, info = cf.load_epochs(bed(tmp, fmt, stim, resp, fmt), verbose=False)
            check(f"{fmt}: shape matches npz", s.shape == s_npz.shape,
                  f"{s.shape} vs {s_npz.shape}")
            check(f"{fmt}: stim bit-identical", np.array_equal(s, s_npz),
                  f"max|diff| = {np.max(np.abs(s - s_npz)):.3e}")
            check(f"{fmt}: resp bit-identical", np.array_equal(r, r_npz),
                  f"max|diff| = {np.max(np.abs(r - r_npz)):.3e}")
            check(f"{fmt}: meta.json found", info.get("meta") is not None, "")

        print("\n--- 2. failure paths must fail, and say why ---")
        ok, m = raises(lambda: cf.load_epochs(
            bed(tmp, "t7", stim.T, resp.T, "v7"), verbose=False), "transposed")
        check("v7 transposed refused", ok, m)
        ok, m = raises(lambda: cf.load_epochs(
            bed(tmp, "t73", stim.T, resp.T, "v73"), verbose=False), "transposed")
        check("v7.3 transposed refused", ok, m)
        ok, m = raises(lambda: cf.load_epochs(
            bed(tmp, "nometa", stim, resp, "v7", meta=None), verbose=False),
            "sample_interval_s")
        check("missing meta.json refused", ok, m)
        ok, m = raises(lambda: cf.load_epochs(
            bed(tmp, "mismatch", stim, resp[:2], "v7"), verbose=False), "differ")
        check("stim/resp size mismatch refused", ok, m)

        p = os.path.join(tmp, "novar"); os.makedirs(p)
        savemat(os.path.join(p, "cell.mat"), {"stim": stim, "voltage": resp})
        json.dump(META, open(os.path.join(p, "meta.json"), "w"))
        ok, m = raises(lambda: cf.load_epochs(os.path.join(p, "cell.mat"), verbose=False),
                       "expected", "resp")
        check("missing 'resp' variable refused", ok, m)

        p = os.path.join(tmp, "corrupt"); os.makedirs(p)
        open(os.path.join(p, "cell.mat"), "w").write("not a MATLAB file")
        json.dump(META, open(os.path.join(p, "meta.json"), "w"))
        ok, m = raises(lambda: cf.load_epochs(os.path.join(p, "cell.mat"), verbose=False),
                       "not a readable MATLAB file")
        check("corrupt .mat refused", ok, m)

        # A cell array of per-epoch traces is a normal MATLAB export and reads as
        # object references, not numbers. Saying "expected 'stim'" about a variable that
        # plainly IS in the file sends you looking in the wrong place.
        p = os.path.join(tmp, "cellarray"); os.makedirs(p)
        with h5py.File(os.path.join(p, "cell.mat"), "w") as fh:
            grp = fh.create_group("epochs")
            refs = [grp.create_dataset(f"e{i}", data=stim[i]).ref for i in range(N_EP)]
            fh.create_dataset("stim", data=refs, dtype=h5py.ref_dtype)
            fh.create_dataset("resp", data=np.asarray(resp).T)
        json.dump(META, open(os.path.join(p, "meta.json"), "w"))
        ok, m = raises(lambda: cf.load_epochs(os.path.join(p, "cell.mat"), verbose=False),
                       "stim", "not a numeric array")
        check("cell-array stim explained", ok, m)

        print("\n--- 3. v7.3 without h5py must explain itself ---")
        f73 = bed(tmp, "v73c", stim, resp, "v73")
        code = (f"import sys; sys.path.insert(0,{SCRIPTS!r});"
                "import builtins; _r=builtins.__import__;"
                "builtins.__import__=lambda n,*a,**k:(_ for _ in ()).throw(ImportError(n))"
                " if n=='h5py' else _r(n,*a,**k);"
                f"import cascade_fit as cf; cf.load_epochs({f73!r}, verbose=False)")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        err = r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""
        check("h5py missing -> actionable message",
              r.returncode != 0 and "h5py" in err and "-v7" in err, err[:100])

        print("\n--- 4. the rest of the toolchain accepts .mat ---")
        p = os.path.join(tmp, "pipeline"); os.makedirs(p)
        savemat(os.path.join(p, "cell.mat"), {"stim": stim, "resp": resp})
        json.dump(META, open(os.path.join(p, "meta.json"), "w"))
        json.dump({"params": TRUE, "dt": DT}, open(os.path.join(p, "results.json"), "w"))
        r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "make_figures.py"),
                            "results.json", "cell.mat", "fig.png"],
                           capture_output=True, text=True, cwd=p)
        check("make_figures.py on .mat",
              r.returncode == 0 and os.path.exists(os.path.join(p, "fig.png")),
              (r.stdout or r.stderr).strip().splitlines()[0][:80] if (r.stdout or r.stderr) else "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    bad = [n for n, ok in rows if not ok]
    print(f"\n{len(rows) - len(bad)}/{len(rows)} passed")
    if bad:
        print("FAILED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
