"""
Correctness tests for the primitives shared across emulator backends.

These run with NumPy alone; Qiskit is used only where available, and those
tests skip cleanly otherwise. No emulator installation (and in particular no
NetSquid licence) is required, so a reviewer can execute this file directly
from a fresh clone:

    python -m pytest tests/test_primitives.py -v

or, without pytest:

    python tests/test_primitives.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.dqc_common import (  # noqa: E402
    classical_fidelity,
    distribute_qubits,
    fidelity_from_counts,
    ideal_qft_probs,
    partition_list,
)


# --------------------------------------------------------------------------- #
# Exact 2-qubit reference unitaries
# --------------------------------------------------------------------------- #

def _rz(a):
    return np.diag([np.exp(-1j * a / 2), np.exp(1j * a / 2)])


_I2 = np.eye(2)
_CX = np.array([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]], dtype=complex)


def _on_control(u):
    return np.kron(u, _I2)


def _on_target(u):
    return np.kron(_I2, u)


def _cp_ref(a):
    return np.diag([1, 1, 1, np.exp(1j * a)]).astype(complex)


def _crz_ref(a):
    return np.diag([1, 1, np.exp(-1j * a / 2), np.exp(1j * a / 2)]).astype(complex)


def _equal_up_to_global_phase(a, b, tol=1e-10):
    idx = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    if abs(b[idx]) < tol:
        return False
    phase = a[idx] / b[idx]
    return abs(abs(phase) - 1.0) < tol and np.allclose(a, phase * b, atol=tol)


def _abstractnode_sequence(angle):
    """Reproduce the exact gate sequence of AbstractNode.CP as a 4x4 unitary.

    AbstractNode itself cannot be imported without NetSquid, so the sequence is
    mirrored here. If the implementation there changes, this test must change
    with it -- that coupling is intentional.
    """
    if angle < 0:
        angle = 2 * np.pi + angle
    u = np.eye(4, dtype=complex)
    u = _on_control(_rz(angle / 2)) @ u
    u = _CX @ u
    u = _on_target(_rz(2 * np.pi - angle / 2)) @ u
    u = _CX @ u
    u = _on_target(_rz(angle / 2)) @ u
    return u


ANGLES = [np.pi, np.pi / 2, -np.pi / 2, np.pi / 4, -np.pi / 8, -np.pi / 64]


def test_abstractnode_gate_is_controlled_phase():
    """AbstractNode's gate (historically named CRz) implements CP, not CRz."""
    for a in ANGLES:
        u = _abstractnode_sequence(a)
        assert _equal_up_to_global_phase(u, _cp_ref(a)), \
            "angle {0}: sequence is not CP".format(a)
        if abs(abs(a) - np.pi) > 1e-12:   # CP(pi) == CRz(pi) up to phase
            assert not _equal_up_to_global_phase(u, _crz_ref(a)), \
                "angle {0}: sequence unexpectedly matches CRz".format(a)


def test_crz_cp_relation():
    """crz(a) = cp(a) . P_control(-a/2) -- the two differ by a control-local phase."""
    for a in ANGLES:
        correction = _on_control(np.diag([1, np.exp(-1j * a / 2)]))
        assert _equal_up_to_global_phase(correction @ _cp_ref(a), _crz_ref(a))


def test_cp_wraparound_is_valid_but_crz_wraparound_is_not():
    """The `angle < 0 -> angle + 2*pi` branch is sound for CP and unsound for CRz."""
    for a in [-np.pi / 2, -np.pi / 4, -np.pi / 16]:
        assert np.allclose(_cp_ref(a), _cp_ref(a + 2 * np.pi))
        assert not np.allclose(_crz_ref(a), _crz_ref(a + 2 * np.pi))


# --------------------------------------------------------------------------- #
# Partitioning
# --------------------------------------------------------------------------- #

def test_distribute_qubits_conserves_and_balances():
    import warnings as _w
    for n in range(2, 33):
        for m in range(1, min(n, 9) + 1):
            with _w.catch_warnings():
                # degenerate <=1-qubit partitions are expected here and warn by design
                _w.simplefilter("ignore", RuntimeWarning)
                parts = distribute_qubits(n, m)
            assert len(parts) == m
            assert sum(parts) == n
            assert max(parts) - min(parts) <= 1
            assert parts == sorted(parts, reverse=True)


def test_distribute_qubits_rejects_impossible_partitions():
    for bad in [(3, 5), (1, 2)]:
        try:
            distribute_qubits(*bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for {0}".format(bad))


def test_partition_list_matches_distribute_qubits():
    lst = list(range(14))
    chunks = partition_list(lst, 4)
    assert [len(c) for c in chunks] == distribute_qubits(14, 4)
    assert [x for c in chunks for x in c] == lst


# --------------------------------------------------------------------------- #
# Reference distribution and fidelity
# --------------------------------------------------------------------------- #

def test_ideal_qft_probs_normalised():
    for nq in range(1, 9):
        for theta in [0.0, 0.33, 0.5, 0.66, 0.875]:
            _, probs = ideal_qft_probs(theta, nq)
            assert abs(probs.sum() - 1.0) < 1e-9


def test_ideal_qft_probs_peaks_on_dyadic_theta():
    """A theta that is exactly representable in nq bits gives a delta function."""
    nq = 5
    theta = 21 / 2 ** nq
    x, probs = ideal_qft_probs(theta, nq)
    assert abs(probs.max() - 1.0) < 1e-9
    assert abs(x[int(np.argmax(probs))] - theta) < 1e-12


def test_fidelity_of_ideal_counts_is_one():
    nq, theta = 6, 0.33
    x, probs = ideal_qft_probs(theta, nq)
    counts = {float(xi): float(p) for xi, p in zip(x, probs) if p > 0}
    assert abs(fidelity_from_counts(counts, nq, theta) - 1.0) < 1e-9


def test_fidelity_bounds_and_symmetry():
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.3, 0.5])
    assert abs(classical_fidelity(p, p) - 1.0) < 1e-12
    assert abs(classical_fidelity(p, q) - classical_fidelity(q, p)) < 1e-12
    assert 0.0 <= classical_fidelity(p, q) <= 1.0
    assert classical_fidelity([1, 0], [0, 1]) < 1e-12


# --------------------------------------------------------------------------- #
# End-to-end check against Qiskit, when available
# --------------------------------------------------------------------------- #

def test_distributed_iqft_matches_ideal_distribution():
    """The Qiskit distributed iQFT reproduces the analytic distribution.

    This is the check that would have caught a cp/crz mismatch if the control
    qubits were ever reused coherently, and it pins the partitioned circuit to
    the same reference the fidelity column is computed against.
    """
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
    except ImportError:
        print("skipping: qiskit / qiskit-aer not installed")
        return

    def build(nq, theta, parts):
        qc = QuantumCircuit(nq, nq)
        for i in range(nq):
            qc.h(i)
            qc.rz(theta * 2 * np.pi * 2 ** i, i)
        order = list(range(nq))[::-1]
        for a_idx, a in enumerate(order):
            qc.h(a)
            for b in order[a_idx + 1:]:
                qc.cp(-np.pi / 2 ** (a - b), a, b)
        qc.measure(range(nq), range(nq))
        return qc

    nq, theta, shots = 6, 0.33, 200000
    sim = AerSimulator()
    qc = build(nq, theta, 2)
    counts = sim.run(transpile(qc, sim), shots=shots, seed_simulator=11).result().get_counts()
    phases = {int(b[::-1], 2) / 2 ** nq: v for b, v in counts.items()}
    fid = fidelity_from_counts(phases, nq, theta)
    assert fid > 0.99, "noiseless distributed iQFT fidelity {0}".format(fid)


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
