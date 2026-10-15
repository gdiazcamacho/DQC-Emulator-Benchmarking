"""
Tests for the CUNQA adapter (emulators/cunqa/scripts/qft_multinode.py).

Two tiers:
  * Layout/decoding tests need NumPy only.
  * Circuit tests need the CUNQA *Python* package importable (no C++ build, no
    deployment): `source emulators/cunqa/env/cunqa_env.sh` (sets CUNQA_PREFIX), or
    put a cunqa checkout on PYTHONPATH (or install under $HOME,
    as on CESGA). They build the real CunqaCircuits and run them on the offline
    shadow executor. Skipped cleanly otherwise.

    PYTHONPATH=/path/to/cunqa python -m pytest tests/test_cunqa_adapter.py -v
"""
import os
import sys
import warnings

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "emulators", "cunqa", "scripts"))
sys.path.append(os.path.expanduser(os.getenv("CUNQA_PREFIX", os.getenv("HOME", ""))))

import qft_multinode as cq                                   # noqa: E402
from common.dqc_common import distribute_qubits              # noqa: E402

try:
    from cunqa.circuit import CunqaCircuit                   # noqa: F401
    HAVE_CUNQA = True
except Exception:                                            # pragma: no cover
    HAVE_CUNQA = False

needs_cunqa = pytest.mark.skipif(not HAVE_CUNQA, reason="cunqa Python package not importable")

SWEEP = [(n, M) for n in range(4, 21, 2) for M in (1, 2, 4, 8)
         if n >= 2 * M]                                       # min_qubits_per_node = 2


def _sizes(n, M):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return distribute_qubits(n, M)


# --- NumPy-only -------------------------------------------------------------

@pytest.mark.parametrize("mode", cq.CUNQA_COMM_MODES)
def test_ebit_closed_forms_match_changelog(mode):
    # packed = m*C(M,2), basic = n^2 (M-1) / (2M) for equal node sizes.
    for n, M in [(4, 2), (6, 3), (8, 2), (8, 4), (12, 4), (16, 8)]:
        m = n // M
        L = cq.Layout([m] * M, mode)
        expected = {"packed": m * M * (M - 1) // 2,
                    "basic": n * n * (M - 1) // (2 * M),
                    "dynamic": 0}[mode]
        assert L.expected_ebits() == expected


def test_decode_reverses_to_global_order():
    L = cq.Layout([2, 2], "packed")
    # Global qubits (q0..q3) = 1,0,1,1 -> MSB-first "1011" = 11/16.
    # CUNQA key: highest saved clbit leftmost -> q3 q2 q1 q0 = "1101".
    assert cq.decode_joint({"1101": 7}, L) == {11 / 16: 7}


def test_decode_rejects_wrong_width():
    L = cq.Layout([2, 2], "packed")
    with pytest.raises(RuntimeError):
        cq.decode_joint({"01": 3}, L)        # the pre-fix symptom: relayed bits missing


# --- needs the cunqa Python package -----------------------------------------

def _max_clbit(circuit):
    top = -1
    for ins in circuit.instructions:
        cl = ins.get("clbits")
        if cl is None or ins["name"] == "endcif":
            continue
        cl = cl if isinstance(cl, list) else [cl]
        top = max([top] + list(cl))
    return top


@needs_cunqa
@pytest.mark.parametrize("mode", cq.CUNQA_COMM_MODES)
def test_clbits_in_range_over_whole_sweep(mode):
    """Regression: the old fixed 2-bit scratch region overflowed at 8 nodes."""
    for n, M in SWEEP:
        circs, L, ebits = cq.build_circuits(_sizes(n, M), 0.33, mode)
        assert ebits == L.expected_ebits()
        for c in circs:
            assert _max_clbit(c) < L.num_clbits, (n, M, mode)


@needs_cunqa
@pytest.mark.parametrize("mode", cq.CUNQA_COMM_MODES)
@pytest.mark.parametrize("n,M", [(4, 1), (4, 2), (6, 3), (7, 2), (8, 4), (10, 4)])
def test_deterministic_theta_exact(mode, n, M):
    from cunqa_shadow import run_shadow
    k = int("10" * n, 2) >> n | 1                 # non-palindromic, odd
    theta = k / 2 ** n
    circs, L, ebits = cq.build_circuits(_sizes(n, M), theta, mode)
    counts, stats = run_shadow(circs, 12, comm="quantum" if L.quantum else "classical", seed=5)
    assert cq.decode_joint(counts[0], L) == {theta: 12}
    assert sum((s - 1) * v for s, v in stats["gen_ent_sizes"].items()) == ebits
    assert cq.marginal_mismatch([dict(c) for c in counts], L) == 0


@needs_cunqa
def test_statevector_sizing():
    for mode in cq.CUNQA_COMM_MODES:
        L = cq.Layout([5, 5, 4, 4], mode)
        assert L.statevector_qubits() == (18 + 4 if L.quantum else 5)
        assert cq.default_mem_per_qpu_gb(L) >= 2
