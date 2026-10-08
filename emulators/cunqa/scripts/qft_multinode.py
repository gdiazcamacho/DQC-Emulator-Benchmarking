"""
Distributed inverse QFT on CUNQA (github.com/CESGA-Quantum-Spain/cunqa, 3.0.0).

Same benchmark as the other emulator scripts: qubits are partitioned across
`num_nodes` vQPUs (common.dqc_common.distribute_qubits), every node prepares
its share of the Fourier state locally, and the inverse-QFT ladder runs node
by node in descending order. What differs between comm modes is only how a
controlled-phase rotation that crosses a node boundary is realised:

  packed   quantum_comm. One telegate per control qubit, shared with EVERY
           lower node at once: cat_entangler over [control node, *lower nodes]
           (one GHZ round = R e-bits for R receivers), all the column's `cp`
           gates applied locally on the receivers' comm qubits, then
           cat_disentangler. E-bits = sum_i size_i * i.
  basic    quantum_comm. One two-party telegate per individual cross-node gate.
           E-bits = number of cross-node controlled-phase gates.
  dynamic  classical_comm, zero e-bits. Semiclassical (deferred-measurement)
           inverse QFT: each qubit is measured right after its Hadamard and
           every later rotation it controlled becomes a classically
           conditioned phase `cif(bit) p(angle) endcif` on the target node,
           the bit travelling with send/recv. Valid ONLY because the transform
           is terminal (job_creator.DYNAMIC_ONLY_FOR). Same pattern as the
           iterative QPE of CUNQA's examples/cl_comm/03 and chapter 20.3 of the
           QCE26 tutorial. The Fourier input is a product state, so no
           entanglement ever needs to cross a node.

Primitive equivalence (stated once, used everywhere below): this suite's
cat_entangle / cat_disentangle == CUNQA's qc_protocols.cat_entangler /
cat_disentangler == NetQMPI's expose / unexpose (QCE26 tutorial, ch. 23.3:
"expose = gen_ent over N circuits, then cat_entangler"). A packed column is
one expose window with every lower node as receiver.

CUNQA 3.0.0 behaviour this script depends on (read from src/, not the docs):

  1. Default backends are tiny: quantum_comm declares (2 data, 1 comm) qubits,
     classical_comm (6, 0), and `run` rejects larger circuits. A backend JSON
     with "num_qubits": [max node size, comm qubits] is therefore written per
     run and passed to qraise.
  2. Quantum communication runs ONE centralized statevector over all data and
     comm qubits of all vQPUs. Memory is requested from the global size, not
     the per-node size (tutorial ch. 17 warns about exactly this).
  3. Counts are per circuit. There is no joint distribution across vQPUs.
     The fidelity metric needs the joint one, so every non-zero node relays
     its measured bits to node 0 with send/recv (dynamic mode gets them for
     free: node 0 already receives every higher bit for its corrections).
  4. A received bit is NOT reported in counts: quantum_comm marks it
     save=False, classical_comm never marks it saved. So node 0 re-measures
     each relayed bit through a scratch data qubit: cif(b) x(r); measure(r, b);
     cif(b) x(r). Without this, node 0's counts silently contain only its own
     qubits (the bug in the previous version of this script).
  5. Counts keys hold only saved clbits, highest index leftmost. Node 0's saved
     bits are exactly the data region [scratch, scratch+N), with global qubit
     g at clbit scratch+g, so reversing the key gives global qubit order with
     qubit 0 most significant -- the same convention as the Qiskit Aer script
     after classical_reverse().
  6. Telegate scratch clbits: cat_disentangler receives one correction bit per
     receiver into distinct clbits on the control node, so the scratch region
     is num_nodes bits wide. The previous fixed width of 2 overflowed the
     register at every 8-node sweep point.
  7. No noise on the communication paths (backend noise only applies to the
     no-communication scheme): this script is noiseless only, Noise="none".

Measured quantities, and how they differ from the other emulators (disclose
in Methods):
  * Execution Time: wall clock of run()+gather(), excluding qraise/qdrop
    queueing. CUNQA's own per-circuit `time_taken` is printed for reference.
  * Peak Memory: the DRIVER process only (resource.getrusage). The simulation
    runs inside the vQPU/executor processes of the qraise allocation, which
    tracemalloc cannot see. Use `seff` on the qraise job for the real figure.
  * Fidelity: noiseless, so it is a correctness check, not a noise result.

Validation: `--shadow` runs the identical circuits on cunqa_shadow.py (NumPy,
no deployment) and prints fidelity and e-bits without writing a CSV. Do this
first; then on the cluster run 4 qubits / 2 nodes per mode at a
non-palindromic theta and compare against noiseless Qiskit Aer before
submitting the sweep.
"""

import argparse
import json
import math
import os as _os
import resource
import sys as _sys
import tempfile
import time
import uuid

import numpy as np

# --- shared helpers -----------------------------------------------------------
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_REPO_ROOT = _os.path.abspath(_os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _REPO_ROOT not in _sys.path:
    _sys.path.insert(0, _REPO_ROOT)
from common.dqc_common import distribute_qubits, fidelity_from_counts, write_result_row
# ------------------------------------------------------------------------------

# Where CUNQA is installed (python package at $CUNQA_PREFIX/cunqa, CLI tools
# such as `qraise` at $CUNQA_PREFIX/bin). Set by emulators/cunqa/env/cunqa_env.sh;
# falls back to $HOME, CUNQA's own default install prefix.
CUNQA_PREFIX = _os.path.expanduser(_os.getenv("CUNQA_PREFIX", _os.getenv("HOME", "")))
if CUNQA_PREFIX and CUNQA_PREFIX not in _sys.path:
    # Appended, not prepended: with the $HOME fallback, stray modules in $HOME
    # must not shadow the standard library or numpy.
    _sys.path.append(CUNQA_PREFIX)
_bin = _os.path.join(CUNQA_PREFIX, "bin")
if _os.path.isdir(_bin) and _bin not in _os.getenv("PATH", "").split(_os.pathsep):
    # cunqa.qpu.qraise shells out to the `qraise` binary.
    _os.environ["PATH"] = _bin + _os.pathsep + _os.getenv("PATH", "")

CUNQA_COMM_MODES = ("packed", "basic", "dynamic")
QUANTUM_MODES = ("packed", "basic")


# --------------------------------------------------------------------------- #
# Circuit construction
# --------------------------------------------------------------------------- #

class Layout:
    """Qubit/clbit bookkeeping shared by builder, decoder and tests."""

    def __init__(self, node_sizes, comm_mode):
        if comm_mode not in CUNQA_COMM_MODES:
            raise ValueError("comm_mode %r not in %s" % (comm_mode, CUNQA_COMM_MODES))
        self.sizes = list(node_sizes)
        self.num_nodes = len(node_sizes)
        self.num_qubits = sum(node_sizes)
        self.comm_mode = comm_mode
        self.quantum = comm_mode in QUANTUM_MODES
        self.comm_per_node = 1 if self.quantum else 0
        # clbits [0, scratch): telegate corrections; [scratch, scratch+N): data
        self.scratch = self.num_nodes if self.quantum else 0
        self.num_clbits = self.scratch + self.num_qubits
        self.first = np.cumsum([0] + self.sizes[:-1]).tolist()   # global index of each node's qubit 0

    def node_of(self, g):
        for i in range(self.num_nodes - 1, -1, -1):
            if g >= self.first[i]:
                return i
        raise ValueError(g)

    def global_qubits(self, node):
        return list(range(self.first[node], self.first[node] + self.sizes[node]))

    def clbit(self, g):
        return self.scratch + g

    def expected_ebits(self):
        if self.comm_mode == "dynamic":
            return 0
        if self.comm_mode == "packed":
            return sum(self.sizes[i] * i for i in range(self.num_nodes))
        return sum(self.sizes[i] * sum(self.sizes[:i]) for i in range(self.num_nodes))

    def statevector_qubits(self):
        """Qubits in CUNQA's simulated state: global for quantum_comm, largest
        node for classical_comm (each vQPU simulates only itself)."""
        if self.quantum:
            return self.num_qubits + self.num_nodes * self.comm_per_node
        return max(self.sizes)


def _angle(control_g, target_g):
    # Inverse-QFT controlled-phase angle between global qubits (control > target).
    return -np.pi / 2 ** (control_g - target_g)


def _local_inverse_qft(circuit, layout, node):
    """Local part of the ladder on one node. Mirrors myqft_inverse() in the
    Qiskit Aer script; `cp`, not `crz` (see CHANGELOG, phase 1)."""
    gs = layout.global_qubits(node)
    for j in range(len(gs) - 1, -1, -1):
        circuit.h(j)
        for k in range(j - 1, -1, -1):
            circuit.cp(_angle(gs[j], gs[k]), k, j)


def _rematerialise(circuit, readout_qubit, clbit):
    """Turn a received (hence unsaved) bit into a saved measurement.

    readout_qubit must be in |0>; it is returned to |0>. See docstring item 4.
    """
    circuit.cif(clbit)
    circuit.x(readout_qubit)
    circuit.endcif()
    circuit.measure(readout_qubit, clbit)          # save=True
    circuit.cif(clbit)
    circuit.x(readout_qubit)
    circuit.endcif()


def build_circuits(node_sizes, theta, comm_mode):
    """Return (circuits, layout, ebits). circuits[i] runs on vQPU i."""
    from cunqa.circuit import CunqaCircuit
    from cunqa.qc_protocols import cat_entangler, cat_disentangler

    L = Layout(node_sizes, comm_mode)
    M = L.num_nodes
    token = uuid.uuid4().hex[:8]                    # CunqaCircuit ids are process-global
    shape = (lambda s: (s, 1)) if L.quantum else (lambda s: s)
    circs = [CunqaCircuit(shape(s), L.num_clbits, id="dqft_%s_n%d" % (token, i))
             for i, s in enumerate(L.sizes)]
    comm = [c.get_qubits()[1][0] if L.quantum else None for c in circs]
    ebits = 0

    # 1. Fourier-state preparation (common to all modes; matches fourier_state()).
    for i in range(M):
        for j, g in enumerate(L.global_qubits(i)):
            circs[i].h(j)
            circs[i].rz(theta * 2 * np.pi * 2 ** g, j)

    if L.quantum:
        # 2a. Distributed ladder with telegates, highest node first.
        for i in range(M - 1, -1, -1):
            _local_inverse_qft(circs[i], L, i)
            lower = list(range(i - 1, -1, -1))
            if not lower:
                continue
            for lj, gc in enumerate(L.global_qubits(i)):
                if comm_mode == "packed":
                    # One expose window: control lent to every lower node at once.
                    block = [circs[i]] + [circs[t] for t in lower]
                    cat_entangler(block, lj, [comm[i]] + [comm[t] for t in lower],
                                  [0] * (1 + len(lower)), tag="pk_%s_%d_%d" % (token, i, lj))
                    for t in lower:
                        for lk, gt in enumerate(L.global_qubits(t)):
                            circs[t].cp(_angle(gc, gt), comm[t], lk)
                    cat_disentangler(block, lj, [comm[t] for t in lower],
                                     list(range(1, len(lower) + 1)), [0] * len(lower))
                    ebits += len(lower)
                else:  # basic: one fresh two-party telegate per gate
                    for t in lower:
                        for lk, gt in enumerate(L.global_qubits(t)):
                            pair = [circs[i], circs[t]]
                            cat_entangler(pair, lj, [comm[i], comm[t]], [0, 0],
                                          tag="bs_%s_%d_%d_%d_%d" % (token, i, lj, t, lk))
                            circs[t].cp(_angle(gc, gt), comm[t], lk)
                            cat_disentangler(pair, lj, [comm[t]], [1], [0])
                            ebits += 1

        # 3a. Measure locally; relay every non-zero node's bits to node 0.
        for i in range(M):
            for j, g in enumerate(L.global_qubits(i)):
                circs[i].measure(j, L.clbit(g))
        for t in range(1, M):
            for g in L.global_qubits(t):
                circs[t].send(L.clbit(g), circs[0])
                circs[0].recv(L.clbit(g), circs[t])
        relayed = [g for t in range(1, M) for g in L.global_qubits(t)]

    else:
        # 2b. Semiclassical ladder. Node i first receives every bit of every
        # higher node (each sender emits in descending global order, so the
        # per-pair FIFO order matches), then processes its own qubits from the
        # highest down: corrections, H, measure, broadcast to lower nodes.
        for i in range(M - 1, -1, -1):
            for h in range(M - 1, i, -1):
                for g in reversed(L.global_qubits(h)):
                    circs[i].recv(L.clbit(g), circs[h])
            gs = L.global_qubits(i)
            for lj in range(len(gs) - 1, -1, -1):
                gj = gs[lj]
                for gk in range(L.num_qubits - 1, gj, -1):     # every higher qubit
                    circs[i].cif(L.clbit(gk))
                    # u1, not p: CUNQA 3.1.0's Aer adapter implements U1 but
                    # not P (same matrix). An unsupported gate kills the vQPU
                    # and every peer then blocks forever on recv.
                    circs[i].u1(_angle(gk, gj), lj)
                    circs[i].endcif()
                circs[i].h(lj)
                circs[i].measure(lj, L.clbit(gj))
                for t in range(i - 1, -1, -1):
                    circs[i].send(L.clbit(gj), circs[t])
        # node 0 already holds every higher bit (unsaved).
        relayed = [g for t in range(1, M) for g in L.global_qubits(t)]

    # 4. Node 0: free a data qubit (qubit 0, already measured) and use it to
    #    re-materialise every relayed bit as a saved measurement.
    if relayed:
        r = 0
        c0 = L.clbit(L.global_qubits(0)[0])
        circs[0].cif(c0)
        circs[0].x(r)
        circs[0].endcif()
        for g in relayed:
            _rematerialise(circs[0], r, L.clbit(g))

    return circs, L, ebits


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #

def decode_joint(counts0, layout):
    """Node-0 counts -> {phase value: count}, qubit 0 most significant."""
    phases = {}
    N = layout.num_qubits
    for key, n in counts0.items():
        bits = key.replace(" ", "")
        if len(bits) != N:
            raise RuntimeError(
                "node-0 count key %r has %d bits, expected %d: the saved-clbit "
                "layout is not what decode_joint assumes" % (key, len(bits), N))
        ordered = bits[::-1]                      # char g == global qubit g
        val = int(ordered, 2) / 2 ** N
        phases[val] = phases.get(val, 0) + n
    return phases


def marginal_mismatch(all_counts, layout):
    """Total-variation distance between each node's own counts and the
    marginal of the joint distribution on its qubits. Same shots, so this is
    zero up to nothing at all when the relay pairs shots correctly."""
    joint = all_counts[0]
    worst = 0.0
    shots = float(sum(joint.values()))
    for t in range(1, layout.num_nodes):
        gs = layout.global_qubits(t)
        marg = {}
        for key, n in joint.items():
            ordered = key.replace(" ", "")[::-1]
            sub = "".join(ordered[g] for g in gs)
            marg[sub] = marg.get(sub, 0) + n
        own = {}
        for key, n in all_counts[t].items():
            sub = key.replace(" ", "")[::-1]
            own[sub] = own.get(sub, 0) + n
        keys = set(marg) | set(own)
        tv = 0.5 * sum(abs(marg.get(k, 0) - own.get(k, 0)) for k in keys) / shots
        worst = max(worst, tv)
    return worst


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

def default_mem_per_qpu_gb(layout):
    """Memory per vQPU in GB. quantum_comm: every vQPU is given room for the
    GLOBAL statevector (16 B/amplitude, x3 for simulator workspace), because
    the centralized executor's placement inside the allocation is not under
    our control. classical_comm: per-node state only."""
    amp_bytes = 16 * 2 ** layout.statevector_qubits()
    return max(2, int(math.ceil(3 * amp_bytes / 1e9)) + 1)


def estimate_seconds(circuits, layout, shots, ns_per_amp_op=2.0):
    """Crude single-thread cost model: shots x quantum ops x 2^statevector
    qubits x ~2 ns. Only used to size the qraise wall-clock budget. Under
    quantum_comm the centralized executor steps ONE global state for every
    shot, so the 8-node, 18-20-qubit points are hours, not minutes."""
    ops = sum(1 for c in circuits for i in c.instructions
              if i["name"] not in ("send", "recv", "cif", "endcif"))
    if not layout.quantum:
        ops = max(ops // max(layout.num_nodes, 1), 1)   # vQPUs run concurrently
    return shots * ops * 2 ** layout.statevector_qubits() * ns_per_amp_op * 1e-9


def default_max_time(est_seconds):
    """qraise budget: 3x the estimate + 10 min, clamped to [30 min, 2 days]."""
    secs = int(min(max(3 * est_seconds + 600, 1800), 2 * 86400))
    d, rem = divmod(secs, 86400)
    h, rem = divmod(rem, 3600)
    return "%d-%02d:%02d:00" % (d, h, rem // 60)


def write_backend_json(layout):
    bdir = _os.path.join(_SCRIPT_DIR, "..", "backends")
    _os.makedirs(bdir, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="dqft_", suffix=".json", dir=bdir)
    with _os.fdopen(fd, "w") as fh:
        json.dump({
            "name": "dqc-benchmark-%s" % layout.comm_mode,
            "description": "Generated by DQC-Emulator-Benchmarking for one run",
            "num_qubits": [max(layout.sizes), layout.comm_per_node],
        }, fh)
    return _os.path.abspath(path)


def _time_to_seconds(t):
    """'D-HH:MM:SS' or 'HH:MM:SS' -> seconds."""
    days, _, hms = t.rpartition("-")
    h, m, sec = (int(x) for x in hms.split(":"))
    return int(days or 0) * 86400 + h * 3600 + m * 60 + sec


def _gather_with_timeout(qjobs, timeout_s):
    """cunqa.qjob.gather, but give up after timeout_s.

    If a vQPU dies (e.g. an unsupported gate), its peers block on recv and
    gather() never returns. The wait runs in a daemon thread so the driver can
    still raise, qdrop the family and exit."""
    import threading
    from cunqa.qjob import gather
    box = {}

    def _wait():
        try:
            box["results"] = gather(qjobs)
        except BaseException as exc:          # surfaced in the caller
            box["error"] = exc

    th = threading.Thread(target=_wait, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        raise TimeoutError(
            "CUNQA run did not finish within %ds. A vQPU probably failed and "
            "its peers are blocked; check the newest files in "
            "$STORE/.cunqa/logs/." % timeout_s)
    if "error" in box:
        raise box["error"]
    return box["results"]


def run_on_cunqa(circuits, layout, shots, max_time, mem_per_qpu, seed=None,
                 run_timeout=None):
    from cunqa.qpu import get_QPUs, qraise, qdrop, run

    if run_timeout is None:
        run_timeout = _time_to_seconds(max_time)
    backend = write_backend_json(layout)
    try:
        family = qraise(
            layout.num_nodes, max_time,
            quantum_comm=layout.quantum, classical_comm=not layout.quantum,
            simulator="Aer", co_located=True, backend=backend,
            mem_per_qpu=mem_per_qpu,
        )
        try:
            qpus = get_QPUs(co_located=True, family=family)
            kwargs = {"shots": shots}
            if seed is not None:
                kwargs["seed"] = seed
            start = time.time()
            results = _gather_with_timeout(run(circuits, qpus, **kwargs),
                                           run_timeout)
            elapsed = time.time() - start
        finally:
            qdrop(family)
    finally:
        try:
            _os.remove(backend)
        except OSError:
            pass
    counts = [dict(r.counts) for r in results]
    taken = [r.time_taken for r in results]
    return counts, elapsed, taken


def run_on_shadow(circuits, layout, shots, seed=None):
    from cunqa_shadow import run_shadow
    start = time.time()
    counts, stats = run_shadow(circuits, shots,
                               comm="quantum" if layout.quantum else "classical",
                               seed=seed)
    return [dict(c) for c in counts], time.time() - start, stats


def main():
    ap = argparse.ArgumentParser(description="Distributed inverse QFT on CUNQA.")
    ap.add_argument("--num_qubits", type=int, required=True)
    ap.add_argument("--num_nodes", type=int, required=True)
    ap.add_argument("--theta", type=float, required=True)
    ap.add_argument("--shots", type=int, default=100)
    ap.add_argument("--comm_mode", choices=CUNQA_COMM_MODES, default="packed")
    ap.add_argument("--noise", choices=["none"], default="none",
                    help="CUNQA applies no noise on communication paths; accepted "
                         "only so the scaffolding can pass the axis uniformly.")
    ap.add_argument("--max_time", default=None,
                    help="qraise wall-clock budget D-HH:MM:SS (default: from "
                         "estimate_seconds, clamped to [30 min, 2 days])")
    ap.add_argument("--mem_per_qpu", type=int, default=None,
                    help="GB per vQPU (default: from the global statevector size)")
    ap.add_argument("--seed", type=int, default=None,
                    help="shadow runs only (CUNQA 3.1.0 re-seeds every shot)")
    ap.add_argument("--shadow", action="store_true",
                    help="Run on the offline NumPy shadow executor; writes no CSV.")
    ap.add_argument("--run_timeout", type=int, default=None,
                    help="seconds to wait for results before giving up "
                         "(default: the qraise budget)")
    args = ap.parse_args()

    if not args.shadow:
        # Live runs must use the pinned build, never a stray cunqa package
        # found via the $HOME fallback (e.g. an old ~/cunqa from 2.x).
        if _sys.version_info < (3, 9):
            _sys.exit("Python %s is too old for CUNQA 3.x. Run "
                      "`source emulators/cunqa/env/cunqa_env.sh` first."
                      % _sys.version.split()[0])
        if args.seed is not None:
            # CUNQA 3.1.0 re-seeds the simulator at the start of EVERY shot
            # (AERSimulatorAdapter::initialize -> set_seed(config.seed)), so a
            # fixed seed makes every shot of a dynamic circuit identical: all
            # shots land on one outcome and the fidelity collapses to that
            # outcome's ideal probability. Observed on QMIO: theta=0.33 gave
            # 0.768 (= p_ideal of the modal outcome) instead of ~0.99.
            _sys.exit("--seed is not supported for live CUNQA runs: CUNQA "
                      "3.1.0 reuses the seed for every shot. Use it with "
                      "--shadow only.")
        if not _os.getenv("CUNQA_PREFIX"):
            _sys.exit("CUNQA_PREFIX is not set. Run "
                      "`source emulators/cunqa/env/cunqa_env.sh` first.")

    sizes = distribute_qubits(args.num_qubits, args.num_nodes)
    circuits, layout, ebits = build_circuits(sizes, args.theta, args.comm_mode)
    if ebits != layout.expected_ebits():
        raise RuntimeError("e-bit count %d != closed form %d" % (ebits, layout.expected_ebits()))
    mem = args.mem_per_qpu or default_mem_per_qpu_gb(layout)
    est = estimate_seconds(circuits, layout, args.shots)
    max_time = args.max_time or default_max_time(est)

    if not args.shadow:
        import cunqa
        where = _os.path.dirname(_os.path.abspath(cunqa.__file__))
        print("cunqa package: %s" % where)
        if not where.startswith(_os.path.abspath(CUNQA_PREFIX) + _os.sep):
            _sys.exit("cunqa was imported from %s, not from CUNQA_PREFIX=%s. "
                      "Another cunqa install shadows the pinned build."
                      % (where, CUNQA_PREFIX))
    print("partition=%s comm_mode=%s ebits=%d statevector_qubits=%d "
          "mem_per_qpu=%dG est=%.0fs max_time=%s"
          % (sizes, args.comm_mode, ebits, layout.statevector_qubits(), mem,
             est, max_time))

    if args.shadow:
        counts, elapsed, stats = run_on_shadow(circuits, layout, args.shots, args.seed)
        if stats["gen_ent_rounds"] and sum((k - 1) * v for k, v in stats["gen_ent_sizes"].items()) != ebits:
            raise RuntimeError("shadow gen_ent audit disagrees with ebits: %s" % stats)
    else:
        counts, elapsed, taken = run_on_cunqa(circuits, layout, args.shots,
                                              max_time, mem, args.seed,
                                              run_timeout=args.run_timeout)
        print("CUNQA time_taken per vQPU (s):", taken)

    phases = decode_joint(counts[0], layout)
    fidelity = fidelity_from_counts(phases, args.num_qubits, args.theta)
    mismatch = marginal_mismatch(counts, layout)
    print("Fidelity: %.6f   marginal TV mismatch: %.3g   run+gather: %.3f s"
          % (fidelity, mismatch, elapsed))
    if mismatch > 1e-12:
        print("WARNING: node marginals disagree with the relayed joint "
              "distribution -- shot pairing in the relay is broken.")

    if args.shadow:
        return

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss   # driver only
    path = write_result_row(
        emulator="cunqa", num_nodes=args.num_nodes, num_qubits=args.num_qubits,
        theta=args.theta, shots=args.shots, execution_time=elapsed,
        current_kb=peak_kb, peak_kb=peak_kb, result=phases, algorithm="qft",
        fidelity=fidelity, ebits=ebits, comm_mode=args.comm_mode, noise="none",
    )
    print("Wrote %s (memory column = driver process only)" % path)


if __name__ == "__main__":
    main()
