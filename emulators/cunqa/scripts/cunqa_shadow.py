"""
Offline shadow executor for CUNQA circuits.

Runs the instruction streams that the REAL `cunqa.circuit.CunqaCircuit` and
`cunqa.qc_protocols` record, but executes them with a small NumPy statevector
instead of qraise'd vQPUs. Purpose: validate the distributed circuits built by
qft_multinode.py (correctness, clbit layout, readout decoding, e-bit counts)
without a CUNQA deployment, before any cluster time is spent. It is the CUNQA
counterpart of the Qiskit shadow runtime used for SquidASM.

Only the CUNQA *Python* package is needed (it imports without the C++ build
once `cunqa/utils/constants.py` exists). Nothing here talks to SLURM.

Semantics reproduced from CUNQA 3.0.0 (src/, read 2026-10-01), because they
decide what ends up in `Result.counts`:

  * Per-shot lockstep. Every circuit advances until it blocks on `recv` (no
    message yet) or on `gen_ent` (not every participant has arrived). Mirrors
    the cooperative stream scheduler in
    src/sim/backends/quantum_comm/qc_executor.cpp.
  * `gen_ent` with the same tag on N circuits prepares a GHZ state on their
    comm qubits. The shadow asserts those comm qubits are |0> first, so a
    protocol that forgets to reset one fails loudly here.
  * `measure(..., save=False)` stores the bit but keeps it out of the counts.
  * quantum_comm: `recv` writes the bit AND marks it save=False
    (qc_executor.cpp, ClassicalComm branch). classical_comm: `recv` writes the
    bit and leaves its save flag untouched (cc_executor.cpp), which for a bit
    never measured locally means "absent", i.e. also not in the counts.
  * Counts key: every saved clbit in ascending index order, each new one
    PREPENDED (Simulator::get_measures / update_measures_), i.e. the highest
    saved clbit is the leftmost character, unsaved bits are dropped entirely.
  * `cif` evaluates and/or/xor/... over the listed clbits; no nesting.
  * classical_comm `send`/`recv` require one clbit per instruction
    (cc_executor checks clbits.size() == qpus.size()); enforced here.
"""

from collections import Counter, defaultdict, deque

import numpy as np

_SQ2 = 1 / np.sqrt(2)
_GATES_1Q = {
    "h": np.array([[_SQ2, _SQ2], [_SQ2, -_SQ2]], dtype=complex),
    "x": np.array([[0, 1], [1, 0]], dtype=complex),
    "y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "z": np.array([[1, 0], [0, -1]], dtype=complex),
    "s": np.diag([1, 1j]).astype(complex),
    "sdg": np.diag([1, -1j]).astype(complex),
    "t": np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex),
    "tdg": np.diag([1, np.exp(-1j * np.pi / 4)]).astype(complex),
}


def _rz(t):
    return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)]).astype(complex)


def _p(t):
    return np.diag([1, np.exp(1j * t)]).astype(complex)


_PARAM_1Q = {"rz": _rz, "p": _p, "u1": _p}
_CONTROLLED = {"cx": lambda _: _GATES_1Q["x"], "cz": lambda _: _GATES_1Q["z"],
               "cp": _p, "cu1": _p, "crz": _rz}


class ShadowError(RuntimeError):
    pass


def _as_list(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


class _State:
    def __init__(self, n, rng):
        self.n = n
        self.rng = rng
        self.psi = np.zeros(2 ** n, dtype=complex)
        self.psi[0] = 1.0

    def _view(self):
        return self.psi.reshape([2] * self.n)

    def apply_1q(self, U, q):
        v = np.moveaxis(self._view(), q, 0)
        v = np.tensordot(U, v, axes=([1], [0]))
        self.psi = np.moveaxis(v, 0, q).reshape(-1)

    def apply_controlled(self, U, c, t):
        v = self._view().copy()
        idx = [slice(None)] * self.n
        idx[c] = 1
        sub = v[tuple(idx)]                      # control = 1 slice (t shifts if t > c)
        tt = t - 1 if t > c else t
        sub = np.moveaxis(np.tensordot(U, np.moveaxis(sub, tt, 0), axes=([1], [0])), 0, tt)
        v[tuple(idx)] = sub
        self.psi = v.reshape(-1)

    def swap(self, a, b):
        self.psi = np.swapaxes(self._view(), a, b).reshape(-1)

    def prob_one(self, q):
        v = np.moveaxis(self._view(), q, 0)
        return float(np.sum(np.abs(v[1]) ** 2))

    def measure(self, q):
        p1 = self.prob_one(q)
        bit = int(self.rng.random() < p1)
        v = np.moveaxis(self._view(), q, 0).copy()
        v[1 - bit] = 0
        norm = np.sqrt(p1 if bit else 1 - p1)
        self.psi = (np.moveaxis(v, 0, q) / norm).reshape(-1)
        return bit


def _cif_value(creg, clbits, op):
    bits = [creg.get(c, False) for c in clbits]
    if op == "and":
        r = all(bits)
    elif op == "or":
        r = any(bits)
    elif op == "xor":
        r = sum(bits) % 2 == 1
    elif op == "nand":
        r = not all(bits)
    elif op == "nor":
        r = not any(bits)
    elif op == "xnor":
        r = sum(bits) % 2 == 0
    else:
        raise ShadowError("unknown cif operation %r" % op)
    return int(r)


def run_shadow(circuits, shots, comm="quantum", seed=None, max_qubits=22):
    """Execute CunqaCircuits jointly, shot by shot.

    Args:
        circuits: list of CunqaCircuit (the same list passed to cunqa.qpu.run).
        shots: number of shots.
        comm: "quantum" (qraise quantum_comm=True) or "classical"
              (classical_comm=True; any gen_ent is an error).
    Returns:
        list of Counter, one per circuit, keyed exactly like Result.counts
        for a single classical register.
    Also returns stats dict: gen_ent rounds and their sizes (e-bit audit).
    """
    if comm not in ("quantum", "classical"):
        raise ValueError(comm)
    rng = np.random.default_rng(seed)

    ids = [c.id for c in circuits]
    pos = {cid: i for i, cid in enumerate(ids)}
    sizes = [c.num_qubits[0] + c.num_qubits[1] for c in circuits]
    offs = np.cumsum([0] + sizes[:-1]).tolist()
    n_total = sum(sizes)
    if n_total > max_qubits:
        raise ShadowError("%d qubits exceeds shadow limit %d" % (n_total, max_qubits))
    progs = [c.instructions for c in circuits]

    counts = [Counter() for _ in circuits]
    stats = {"gen_ent_rounds": 0, "gen_ent_sizes": Counter()}

    for _shot in range(shots):
        st = _State(n_total, rng)
        creg = [dict() for _ in circuits]
        save = [dict() for _ in circuits]
        pc = [0] * len(circuits)
        chan = defaultdict(deque)               # (sender, receiver) -> FIFO of bits
        waiting = {}                            # tag -> {circuit index: comm qubit}

        def gq(i, q):
            return offs[i] + int(q)

        progress = True
        while progress:
            progress = False
            for i, prog in enumerate(progs):
                while pc[i] < len(prog):
                    ins = prog[pc[i]]
                    name = ins["name"]

                    if name == "recv":
                        clbits = _as_list(ins["clbits"])
                        src = pos[ins["circuits"][0]]
                        if comm == "classical" and len(clbits) != 1:
                            raise ShadowError("classical_comm recv must carry one clbit")
                        if len(chan[(src, i)]) < len(clbits):
                            break                # blocked
                        for c in clbits:
                            creg[i][c] = chan[(src, i)].popleft()
                            if comm == "quantum":
                                save[i][c] = False
                    elif name == "send":
                        clbits = _as_list(ins["clbits"])
                        dst = pos[ins["circuits"][0]]
                        if comm == "classical" and len(clbits) != 1:
                            raise ShadowError("classical_comm send must carry one clbit")
                        for c in clbits:
                            chan[(i, dst)].append(creg[i].get(c, False))
                    elif name == "gen_ent":
                        if comm != "quantum":
                            raise ShadowError("gen_ent under classical_comm")
                        tag = ins["tag"]
                        parts = [pos[c] for c in ins["circuits"]]
                        w = waiting.setdefault(tag, {})
                        if i not in w:
                            w[i] = ins["comm_qubit"]
                        if set(w) != set(parts):
                            break                # wait for the other participants
                        # last arrival: build GHZ and release everyone
                        qs = [gq(j, w[j]) for j in parts]
                        for q in qs:
                            if st.prob_one(q) > 1e-9:
                                raise ShadowError("gen_ent on comm qubit not in |0> (tag %s)" % tag)
                        st.apply_1q(_GATES_1Q["h"], qs[0])
                        for q in qs[1:]:
                            st.apply_controlled(_GATES_1Q["x"], qs[0], q)
                        stats["gen_ent_rounds"] += 1
                        stats["gen_ent_sizes"][len(parts)] += 1
                        for j in parts:
                            if j != i:
                                pc[j] += 1       # their gen_ent is now complete
                        del waiting[tag]
                    elif name == "cif":
                        r = _cif_value(creg[i], _as_list(ins["clbits"]),
                                       ins.get("operation", "and"))
                        if r != ins.get("condition", 1):
                            while prog[pc[i]]["name"] != "endcif":
                                pc[i] += 1
                                if prog[pc[i]]["name"] == "cif":
                                    raise ShadowError("nested cif is not supported by CUNQA")
                    elif name == "endcif":
                        pass
                    elif name == "measure":
                        q, c = ins["qubits"], ins["clbits"]
                        creg[i][c] = bool(st.measure(gq(i, q)))
                        save[i][c] = bool(ins.get("save", True))
                    elif name == "reset":
                        for q in _as_list(ins["qubits"]):
                            if st.measure(gq(i, q)):
                                st.apply_1q(_GATES_1Q["x"], gq(i, q))
                    elif name in _GATES_1Q:
                        st.apply_1q(_GATES_1Q[name], gq(i, _as_list(ins["qubits"])[0]))
                    elif name in _PARAM_1Q:
                        t = float(ins["params"][0])
                        st.apply_1q(_PARAM_1Q[name](t), gq(i, _as_list(ins["qubits"])[0]))
                    elif name in _CONTROLLED:
                        c, t_ = ins["qubits"]
                        par = float(ins["params"][0]) if ins.get("params") else None
                        st.apply_controlled(_CONTROLLED[name](par), gq(i, c), gq(i, t_))
                    elif name == "swap":
                        a, b = ins["qubits"]
                        st.swap(gq(i, a), gq(i, b))
                    else:
                        raise ShadowError("shadow does not implement %r" % name)
                    pc[i] += 1
                    progress = True

        if any(pc[i] < len(p) for i, p in enumerate(progs)):
            stuck = {ids[i]: progs[i][pc[i]]["name"] for i in range(len(progs)) if pc[i] < len(progs[i])}
            raise ShadowError("deadlock: circuits blocked at %s" % stuck)

        for i in range(len(circuits)):
            key = ""
            for c in sorted(creg[i]):
                if save[i].get(c, False):
                    key = ("1" if creg[i][c] else "0") + key
            if key:
                counts[i][key] += 1

    for k in stats["gen_ent_sizes"]:
        stats["gen_ent_sizes"][k] //= shots
    stats["gen_ent_rounds"] //= shots
    return counts, stats
