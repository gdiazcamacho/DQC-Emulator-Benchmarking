#!/usr/bin/env python3
"""Diagnostic: is SQUANCH's inter-agent quantum channel carrying entanglement?

Motivation
----------
The distributed iQFT on SQUANCH returns a WRONG output distribution. For
theta = 0 the inverse QFT of a Fourier state must yield the all-zero string
with probability 1, but the adapter returns a spread distribution (observed:
{0.0: 28, 0.25: 2, 0.5: 1, 0.75: 19} over 50 shots, n=4, k=2).

This script isolates the cause with three tests of increasing scope.

Results observed on squanch 1.1.0:

  1. core, single process, no agents      -> correlation 1.000  (CORRECT)
  2. two agents, separate QStreams        -> correlation 0.500  (NO entanglement)
  3. two agents, one shared QStream       -> correlation 0.700  (race condition)

Test 1 shows the simulator itself is fine. Test 2 reproduces what the adapter
does: run_DQFT_squanch builds `qstreams = [QStream(...) for i in range(P)]`,
one per node. A Bell pair created in one node's QStream and qsend()'d to
another arrives UNENTANGLED, so every cat-entangle/disentangle in the adapter
teleports nothing. Test 3 shows that sharing the QStream restores partial
correlation but leaves a race: both agents touch the same shared-memory
statevector concurrently.

Implication: SQUANCH results produced with this adapter do not reflect a
working distributed protocol, and the fidelity figures derived from them are
not meaningful.
"""

import numpy as np
from squanch import Agent, QStream, H, CNOT

N = 200


def test_core():
    qs = QStream(2, N)
    same = 0
    for q in qs:
        b0, b1 = q.qubit(0), q.qubit(1)
        H(b0)
        CNOT(b0, b1)
        if b0.measure() == b1.measure():
            same += 1
    return same / N


class _Source(Agent):
    def run(self):
        res = []
        for qs in self.qstream:
            b0, b1 = qs.qubit(0), qs.qubit(1)
            H(b0)
            CNOT(b0, b1)
            self.qsend(self.peer, b1)
            res.append(b0.measure())
        self.output(res)


class _Sink(Agent):
    def run(self):
        res = []
        for _ in self.qstream:
            q = self.qrecv(self.peer)
            res.append(q.measure() if q is not None else -1)
        self.output(res)


def test_agents(shared_qstream: bool):
    out = Agent.shared_output()
    if shared_qstream:
        stream_a = stream_b = QStream(2, N)
    else:
        stream_a, stream_b = QStream(2, N), QStream(2, N)
    a = _Source(stream_a, out, name="src")
    b = _Sink(stream_b, out, name="snk")
    a.qconnect(b)
    a.cconnect(b)
    a.peer, b.peer = b, a
    a.start(); b.start(); a.join(); b.join()
    va, vb = np.array(out["src"]), np.array(out["snk"])
    return float(np.mean(va == vb))


if __name__ == "__main__":
    print("Bell-pair correlation (1.000 = entanglement intact):\n")
    print(f"  1. core, single process, no agents : {test_core():.3f}")
    print(f"  2. agents, SEPARATE QStreams       : {test_agents(False):.3f}"
          f"   <- what the adapter does")
    print(f"  3. agents, SHARED QStream          : {test_agents(True):.3f}")
    print("\nTest 2 at ~0.5 means the qsend'd qubit arrives unentangled.")
