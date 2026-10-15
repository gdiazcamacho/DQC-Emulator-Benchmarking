"""
Offline check of SquidASM's 'dynamic' comm mode, without NetSquid.

NetSquid comes from a private index and cannot be installed in CI or most
dev containers, so this test stubs the handful of netqasm/squidasm symbols
that emulators/squidasm/scripts/{AbstractNode,qft_multinode}.py import and
runs the REAL QFTNode.run_dynamic generators under a tiny cooperative
scheduler:

  * each node gets its own NumPy statevector (dynamic mode never entangles
    two nodes, so the joint state is a product and per-node simulation is
    exact);
  * Qubit implements H / rot_Z / measure with netqasm's conventions
    (rot_Z(angle) = exp(-i angle Z / 2));
  * csockets are FIFO queues; recv() blocks by yielding until a message
    arrives, like squidasm's csocket.recv();
  * connection.flush() is a no-op generator (operations execute eagerly).

What this validates: the program's control flow (message order, no
deadlock), angles, global qubit offsets and the bit order of the returned
outputs. What it does not: NetSquid timing or noise. Run the real thing on
the cluster at an exact theta before trusting a sweep.

    python -m pytest tests/test_squidasm_dynamic_mock.py -v
"""
import os
import sys
import types
from collections import deque

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SQ = os.path.join(ROOT, "emulators", "squidasm", "scripts")


# --------------------------------------------------------------------------- #
# Minimal stand-ins for netqasm / squidasm
# --------------------------------------------------------------------------- #

class _NodeSim:
    """Statevector of one node's qubits (qubit k = tensor axis k)."""

    def __init__(self, rng):
        self.rng = rng
        self.psi = np.ones(1, dtype=complex)
        self.n = 0

    def alloc(self):
        self.psi = np.kron(self.psi, np.array([1, 0], dtype=complex)).reshape(-1)
        self.n += 1
        return self.n - 1

    def _apply(self, U, k):
        v = self.psi.reshape([2] * self.n)
        v = np.moveaxis(np.tensordot(U, np.moveaxis(v, k, 0), axes=([1], [0])), 0, k)
        self.psi = v.reshape(-1)

    def measure(self, k):
        v = np.moveaxis(self.psi.reshape([2] * self.n), k, 0).copy()
        p1 = float(np.sum(np.abs(v[1]) ** 2))
        b = int(self.rng.random() < p1)
        v[1 - b] = 0
        v /= np.sqrt(p1 if b else 1 - p1)
        self.psi = np.moveaxis(v, 0, k).reshape(-1)
        return b


class _Conn:
    def __init__(self, sim):
        self.sim = sim

    def flush(self):
        return
        yield  # pragma: no cover  (makes this a generator)


class _Qubit:
    def __init__(self, connection, *a, **k):
        self.sim = connection.sim
        self.k = self.sim.alloc()

    def H(self):
        self.sim._apply(np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2), self.k)

    def rot_Z(self, angle=0.0, n=None, d=None):
        if angle is None or (n is not None):
            angle = n * np.pi / 2 ** d
        self.sim._apply(np.diag([np.exp(-1j * angle / 2), np.exp(1j * angle / 2)]), self.k)

    def measure(self, *a, **k):
        return self.sim.measure(self.k)


class _CSocket:
    def __init__(self, outq, inq):
        self.outq, self.inq = outq, inq

    def send(self, msg):
        self.outq.append(msg)

    def recv(self):
        while not self.inq:
            yield "blocked"
        return self.inq.popleft()


class _ProgramContext:
    def __init__(self, connection, csockets):
        self.connection = connection
        self.csockets = csockets
        self.epr_sockets = {}


class _Program:
    def __init__(self, *a, **k):
        pass


def _install_stubs():
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    for pkg in ("netqasm", "squidasm", "squidasm.util", "squidasm.sim",
                "squidasm.sim.stack", "squidasm.run", "squidasm.run.stack"):
        sys.modules.setdefault(pkg, types.ModuleType(pkg))
    mod("netqasm.sdk", Qubit=_Qubit)
    mod("netqasm.sdk.qubit", Qubit=_Qubit)
    mod("netqasm.sdk.connection", BaseNetQASMConnection=object)
    mod("squidasm.util.util", get_qubit_state=lambda *a, **k: None)
    mod("squidasm.sim.stack.common",
        LogManager=types.SimpleNamespace(get_stack_logger=lambda *_: None))
    mod("squidasm.sim.stack.program", Program=_Program,
        ProgramContext=_ProgramContext, ProgramMeta=dict)
    mod("squidasm.run.stack.run", run=None)
    mod("squidasm.run.stack.config", StackNetworkConfig=None)


@pytest.fixture(scope="module")
def sq():
    pytest.importorskip("yaml")
    saved = dict(sys.modules)
    _install_stubs()
    sys.path.insert(0, SQ)
    sys.path.insert(0, ROOT)
    try:
        import importlib
        for name in ("AbstractNode", "qft_multinode"):
            sys.modules.pop(name, None)
        yield importlib.import_module("qft_multinode")
    finally:
        sys.path.remove(SQ)
        for k in list(sys.modules):
            if k not in saved:
                del sys.modules[k]


def _run_shot(sq, node_qubits, theta, rng):
    names = ["Node%d" % i for i in range(len(node_qubits))]
    net = dict(zip(names, node_qubits))
    queues = {(a, b): deque() for a in names for b in names if a != b}
    gens, done = {}, {}
    for nm in names:
        prog = sq.QFTNode(name=nm, peers=[p for p in names if p != nm],
                          network_structure=net, theta=theta,
                          qubits=net[nm], comm_mode="dynamic")
        socks = {p: _CSocket(queues[(nm, p)], queues[(p, nm)])
                 for p in names if p != nm}
        ctx = _ProgramContext(_Conn(_NodeSim(rng)), socks)
        gens[nm] = prog.run(ctx)
    for _ in range(100000):
        for nm, g in list(gens.items()):
            try:
                next(g)
            except StopIteration as stop:
                done[nm] = stop.value
                del gens[nm]
        if not gens:
            break
    else:
        raise AssertionError("deadlock: %s still blocked" % sorted(gens))
    bits = [b for nm in names for b in done[nm]]       # node order = global order
    return int("".join(map(str, bits)), 2) / 2 ** len(bits)


@pytest.mark.parametrize("node_qubits", [[2, 2], [3, 3], [2, 2, 2], [4, 3], [3, 3, 2, 2]])
def test_exact_theta_is_deterministic(sq, node_qubits):
    n = sum(node_qubits)
    k = int("10" * n, 2) >> n | 1           # non-palindromic, odd
    theta = k / 2 ** n
    rng = np.random.default_rng(7)
    outs = {_run_shot(sq, node_qubits, theta, rng) for _ in range(5)}
    assert outs == {theta}, (node_qubits, theta, outs)


def test_statistics_match_ideal(sq):
    sys.path.insert(0, ROOT)
    from common.dqc_common import fidelity_from_counts
    rng = np.random.default_rng(11)
    counts = {}
    for _ in range(1500):
        v = _run_shot(sq, [3, 2, 2], 0.33, rng)
        counts[v] = counts.get(v, 0) + 1
    # Ideal 1500-shot sampling at n=7, theta=0.33 gives ~0.987 +/- 0.0015.
    assert fidelity_from_counts(counts, 7, 0.33) > 0.975


def test_dynamic_uses_no_epr(sq):
    assert sq.expected_ebits([3, 3, 2], "dynamic") == 0
    assert sq.expected_ebits([3, 3, 2], "packed") == 3 * 1 + 2 * 2


def test_noiseless_network_zeroes_every_noise_source(sq):
    """--noise none is the device network with noise zeroed, nothing else."""
    import copy
    with open(os.path.join(SQ, "config_template.yaml")) as fh:
        noisy = yaml_mod().safe_load(fh)
    clean = sq.make_noiseless(copy.deepcopy(noisy))
    for st in clean["stacks"]:
        q = st["qdevice_cfg"]
        assert q["single_qubit_gate_depolar_prob"] == 0
        assert q["two_qubit_gate_depolar_prob"] == 0
        assert q["T1"] == 0 and q["T2"] == 0
    for ln in clean["links"]:
        assert ln["cfg"]["fidelity"] == 1.0
        assert ln["typ"] == next(l["typ"] for l in noisy["links"])  # same routing
    assert len(clean["stacks"]) == len(noisy["stacks"])
    assert clean["clinks"] == noisy["clinks"]


def yaml_mod():
    import yaml
    return yaml
