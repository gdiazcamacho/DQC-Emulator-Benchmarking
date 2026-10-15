"""
Ideal-sampling baseline for the fidelity metric.

The suite's fidelity is the classical (Bhattacharyya) fidelity between the
measured histogram of S shots and the exact inverse-QFT distribution. Even a
perfect emulator does not score 1: S samples from a spread distribution leave
probability mass unobserved. At S = 100 and n = 4, theta = 0.33 the ceiling is
~0.95 and it falls as n grows. A fidelity is therefore only interpretable next
to this baseline, which depends on (n, theta, S) alone -- not on the emulator,
the node count or the communication mode.

For each (n, theta, S) this draws `reps` ideal samples and records the mean,
standard deviation and the central 95% interval of their fidelity.

    python scripts/ideal_baseline.py configs/campaign_v1_wave1_noiseless.yaml
    # -> results/baselines/ideal_sampling.csv

It works in O(S) per repetition (inverse-CDF sampling plus a fidelity sum
over the observed outcomes only), so n = 20 (10^6 outcomes) is cheap.
"""
import argparse
import csv
import os
import sys

import numpy as np
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, REPO_ROOT)
from common.dqc_common import ideal_qft_probs  # noqa: E402

COLUMNS = ["Num Qubits", "Theta", "Shots", "Reps", "Fidelity Mean",
           "Fidelity SD", "Fidelity P2.5", "Fidelity P97.5"]


def ideal_sampling_fidelity(num_qubits, theta, shots, reps=2000, seed=0):
    """Fidelities of `reps` ideal `shots`-sample histograms (array)."""
    _, p = ideal_qft_probs(theta, num_qubits)
    p = p / p.sum()
    cdf = np.cumsum(p)
    cdf[-1] = 1.0
    rng = np.random.default_rng(seed)
    out = np.empty(reps)
    for r in range(reps):
        idx = np.searchsorted(cdf, rng.random(shots), side="right")
        ks, cnt = np.unique(idx, return_counts=True)
        out[r] = np.sum(np.sqrt(cnt / shots * p[ks])) ** 2
    return out


def baseline_rows(num_qubits_range, theta_range, shots_list, reps=2000):
    rows = []
    for n in num_qubits_range:
        for th in theta_range:
            for s in shots_list:
                f = ideal_sampling_fidelity(n, th, s, reps=reps,
                                            seed=hash((n, th, s)) & 0xFFFF)
                rows.append({
                    "Num Qubits": n, "Theta": th, "Shots": s, "Reps": reps,
                    "Fidelity Mean": f.mean(), "Fidelity SD": f.std(),
                    "Fidelity P2.5": np.percentile(f, 2.5),
                    "Fidelity P97.5": np.percentile(f, 97.5),
                })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("configs", nargs="+", help="campaign config(s) defining the grid")
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "baselines",
                                                  "ideal_sampling.csv"))
    a = ap.parse_args()
    qs, ths, shots = set(), set(), set()
    for path in a.configs:
        with open(path) as fh:
            c = yaml.safe_load(fh)
        qs.update(c["num_qubits_range"]); ths.update(c["theta_range"])
        shots.add(c["shots"])
    rows = baseline_rows(sorted(qs), sorted(ths), sorted(shots), reps=a.reps)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print("n={0:>2} theta={1:<5} shots={2:<5} F = {3:.4f} +/- {4:.4f}  [{5:.4f}, {6:.4f}]".format(
            r["Num Qubits"], r["Theta"], r["Shots"], r["Fidelity Mean"],
            r["Fidelity SD"], r["Fidelity P2.5"], r["Fidelity P97.5"]))
    print("Wrote", os.path.relpath(a.out, REPO_ROOT))


if __name__ == "__main__":
    main()
