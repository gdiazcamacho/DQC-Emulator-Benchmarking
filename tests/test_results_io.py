"""
Correctness tests for the results-CSV contract in common/dqc_common.py.

The suite's core invariant is that a benchmark run is identified by SIX
coordinates -- (Algorithm, Comm Mode, Noise, Emulator, Nodes, Num Qubits,
Theta) -- and that every one of them that discriminates between runs also
appears in the results filename.

This matters because the failure mode is silent. If two runs that differ only
in communication schedule (or only in error model) map to the same path, the
second overwrites the first and nothing complains: plot_results.py simply sees
fewer rows than were produced, and the missing series looks like a job that
never ran rather than a job whose output was destroyed.

Runs with NumPy alone; no emulator installation required:

    python -m pytest tests/test_results_io.py -v

or, without pytest:

    python tests/test_results_io.py
"""

import csv
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.dqc_common import (  # noqa: E402
    COMM_MODES,
    CSV_COLUMNS,
    NOISE_SETTINGS,
    write_result_row,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_BASE = dict(
    emulator="qiskit_aer",
    num_nodes=2,
    num_qubits=6,
    theta=0.33,
    shots=100,
    execution_time=1.23,
    current_kb=10.0,
    peak_kb=20.0,
    result={0.0: 100},
)


def _write(tmpdir, **overrides):
    """Write one result row into tmpdir, returning its basename.

    The default path is derived from the run coordinates, so we let
    write_result_row build it and only redirect the results root.
    """
    kwargs = dict(_BASE)
    kwargs.update(overrides)
    path = write_result_row(algorithm="qft", **kwargs)
    # Move it into the temp dir so the repo's results/ is not polluted.
    dest = os.path.join(tmpdir, os.path.basename(path))
    shutil.move(path, dest)
    return os.path.basename(dest)


def _read_row(path):
    with open(path, newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[0], rows[1]


# --------------------------------------------------------------------------- #
# Filename must discriminate on every axis
# --------------------------------------------------------------------------- #

def test_comm_mode_changes_the_filename():
    """basic and packed runs of the same configuration must not collide.

    This is the regression guard for the packing ablation: (packed - basic) on
    a fixed emulator is the measurement that isolates the saving due to gate
    packing, and it requires both runs to survive on disk.
    """
    tmp = tempfile.mkdtemp()
    try:
        names = {mode: _write(tmp, comm_mode=mode, noise="none")
                 for mode in ("basic", "packed")}
        assert names["basic"] != names["packed"], (
            "basic and packed produced the same filename: {0}".format(names))
        assert len(os.listdir(tmp)) == 2, "one file overwrote the other"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_noise_changes_the_filename():
    """A fidelity figure is meaningless without its error model.

    Noiseless and noisy runs of the same configuration must coexist, otherwise
    a correctness check silently destroys the noisy datapoint it was meant to
    validate.
    """
    tmp = tempfile.mkdtemp()
    try:
        names = {nz: _write(tmp, comm_mode="packed", noise=nz)
                 for nz in ("none", "depolarizing")}
        assert names["none"] != names["depolarizing"], (
            "noise settings produced the same filename: {0}".format(names))
        assert len(os.listdir(tmp)) == 2, "one file overwrote the other"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_all_axes_together_are_distinct():
    """Every (comm_mode, noise) pair must land on its own file."""
    tmp = tempfile.mkdtemp()
    try:
        seen = set()
        for mode in ("basic", "packed"):
            for nz in ("none", "depolarizing"):
                seen.add(_write(tmp, comm_mode=mode, noise=nz))
        assert len(seen) == 4, "expected 4 distinct filenames, got {0}".format(seen)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Row shape
# --------------------------------------------------------------------------- #

def test_header_matches_row_length():
    """plot_results.py concatenates every CSV under emulators/*/results/, so a
    row whose width disagrees with the header corrupts the combined DataFrame
    silently rather than raising."""
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "row.csv")
        kwargs = dict(_BASE)
        write_result_row(algorithm="qft", comm_mode="packed", noise="none",
                         fidelity=1.0, ebits=3, path=path, **kwargs)
        header, row = _read_row(path)
        assert header == CSV_COLUMNS, "header drifted from CSV_COLUMNS"
        assert len(row) == len(CSV_COLUMNS), (
            "row has {0} fields, header has {1}".format(len(row), len(CSV_COLUMNS)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_comm_mode_and_noise_are_recorded_in_the_row():
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, "row.csv")
        kwargs = dict(_BASE)
        write_result_row(algorithm="qft", comm_mode="basic", noise="none",
                         ebits=9, path=path, **kwargs)
        header, row = _read_row(path)
        cells = dict(zip(header, row))
        assert cells["Comm Mode"] == "basic", cells["Comm Mode"]
        assert cells["Noise"] == "none", cells["Noise"]
        assert cells["Ebits"] == "9", cells["Ebits"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Packed boolean stays consistent with Comm Mode
# --------------------------------------------------------------------------- #

def test_packed_is_derived_from_comm_mode():
    """`Packed` is retained for back-compatibility with existing figures, so it
    must never disagree with `Comm Mode`."""
    expected = {"packed": "True", "basic": "False", "dynamic": "", "native": ""}
    tmp = tempfile.mkdtemp()
    try:
        for mode, want in expected.items():
            path = os.path.join(tmp, "{0}.csv".format(mode))
            kwargs = dict(_BASE)
            write_result_row(algorithm="qft", comm_mode=mode, noise="none",
                             path=path, **kwargs)
            header, row = _read_row(path)
            got = dict(zip(header, row))["Packed"]
            assert got == want, (
                "comm_mode={0!r} gave Packed={1!r}, expected {2!r}".format(
                    mode, got, want))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_legacy_packed_boolean_still_infers_comm_mode():
    """Emulator scripts not yet ported to --comm_mode call write_result_row with
    the old `packed=` boolean. Those calls must keep working and must land on
    distinct files."""
    tmp = tempfile.mkdtemp()
    try:
        names = {}
        for packed, label in ((True, "packed"), (False, "basic")):
            kwargs = dict(_BASE)
            path = write_result_row(algorithm="qft", packed=packed, noise="none",
                                    **kwargs)
            dest = os.path.join(tmp, os.path.basename(path))
            shutil.move(path, dest)
            names[label] = os.path.basename(dest)
            header, row = _read_row(dest)
            got = dict(zip(header, row))["Comm Mode"]
            assert got == label, (
                "packed={0} inferred Comm Mode={1!r}, expected {2!r}".format(
                    packed, got, label))
        assert names["packed"] != names["basic"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Invalid values are rejected loudly
# --------------------------------------------------------------------------- #

def test_unknown_comm_mode_raises():
    """A typo in --comm_mode must fail the run, not write a file under a name
    nothing downstream will filter on."""
    tmp = tempfile.mkdtemp()
    try:
        kwargs = dict(_BASE)
        try:
            write_result_row(algorithm="qft", comm_mode="pakced", noise="none",
                             path=os.path.join(tmp, "x.csv"), **kwargs)
        except ValueError:
            return
        raise AssertionError("unknown comm_mode was accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_unknown_noise_raises():
    tmp = tempfile.mkdtemp()
    try:
        kwargs = dict(_BASE)
        try:
            write_result_row(algorithm="qft", comm_mode="packed",
                             noise="depolarising",  # British spelling typo
                             path=os.path.join(tmp, "x.csv"), **kwargs)
        except ValueError:
            return
        raise AssertionError("unknown noise setting was accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_missing_noise_raises():
    """No default error model: a default of "depolarizing" previously
    mislabelled noiseless SQUANCH and Interlin-q rows."""
    tmp = tempfile.mkdtemp()
    try:
        kwargs = dict(_BASE)
        try:
            write_result_row(algorithm="qft", comm_mode="packed",
                             path=os.path.join(tmp, "x.csv"), **kwargs)
        except ValueError:
            return
        raise AssertionError("a row without an explicit noise setting was written")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_shots_changes_the_filename():
    """Regression: a 100-shot SLURM run overwrote a 1000-shot interactive run
    of the same configuration because shots were not in the filename."""
    tmp = tempfile.mkdtemp()
    try:
        names = {sh: _write(tmp, comm_mode="packed", noise="none", shots=sh)
                 for sh in (100, 1000)}
        assert names[100] != names[1000], names
        assert len(os.listdir(tmp)) == 2, "one file overwrote the other"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_provenance_is_recorded():
    tmp = tempfile.mkdtemp()
    old = {k: os.environ.get(k) for k in ("DQC_COMMIT", "DQC_CAMPAIGN")}
    try:
        os.environ["DQC_COMMIT"] = "abc123def456"
        os.environ["DQC_CAMPAIGN"] = "campaign-v1"
        path = os.path.join(tmp, "row.csv")
        kwargs = dict(_BASE)
        write_result_row(algorithm="qft", comm_mode="packed", noise="none",
                         path=path, emulator_version="qiskit-aer==0.17.1",
                         **kwargs)
        header, row = _read_row(path)
        cells = dict(zip(header, row))
        assert cells["Code Commit"] == "abc123def456", cells
        assert cells["Campaign"] == "campaign-v1", cells
        assert cells["Emulator Version"] == "qiskit-aer==0.17.1", cells
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(tmp, ignore_errors=True)


def test_declared_vocabularies_are_covered():
    """Every declared comm mode and noise setting must be writable, so the
    capability matrix in job_creator.py cannot reference a value the writer
    rejects."""
    tmp = tempfile.mkdtemp()
    try:
        for mode in COMM_MODES:
            for nz in NOISE_SETTINGS:
                kwargs = dict(_BASE)
                write_result_row(
                    algorithm="qft", comm_mode=mode, noise=nz,
                    path=os.path.join(tmp, "{0}_{1}.csv".format(mode, nz)),
                    **kwargs)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS  {0}".format(name))
            except AssertionError as exc:
                failures += 1
                print("FAIL  {0}: {1}".format(name, exc))
    print("\n{0} failure(s)".format(failures))
    sys.exit(1 if failures else 0)
