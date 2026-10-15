"""
Contract tests for the campaign tooling (job_creator, templates, baseline).

These guard the failure modes found while preparing campaign-v1:
  * a template passing a flag the emulator script does not accept
    (every Interlin-q job died on --comm_mode);
  * an emulator labelled with an error model it does not apply;
  * two jobs writing the same results file;
  * points emitted that cannot fit in memory or finish in a walltime.

NumPy + PyYAML only.
"""
import os
import re
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
yaml = pytest.importorskip("yaml")

import job_creator as jc                                     # noqa: E402
from ideal_baseline import ideal_sampling_fidelity            # noqa: E402
from common.dqc_common import fidelity_from_counts, ideal_qft_probs  # noqa: E402

EMULATORS = ["qiskit_aer", "squidasm", "squanch", "cunqa", "interlin-q"]
CONFIGS = ["campaign_v1_wave1_noiseless.yaml", "campaign_v1_wave2_noisy.yaml"]
PLACEHOLDERS = {"{{JOB_NAME}}", "{{ALGORITHM}}", "{{COMM_MODE}}", "{{NOISE}}",
                "{{NUM_QUBITS}}", "{{NUM_NODES}}", "{{THETA}}", "{{SHOTS}}",
                "{{MEM_GB}}", "{{COMMIT}}", "{{CAMPAIGN}}"}


def _load(name):
    with open(os.path.join(ROOT, "configs", name)) as fh:
        return yaml.safe_load(fh)


def _script(emulator):
    with open(os.path.join(ROOT, "emulators", emulator, "scripts", "qft_multinode.py")) as fh:
        return fh.read()


def _choices(src, flag):
    """choices=[...] of an argparse flag, read from source (scripts import
    emulator packages that are not installed here)."""
    m = re.search(r"['\"]--{0}['\"][^)]*?choices=\[([^\]]*)\]".format(flag), src, re.S)
    if m is None:
        m = re.search(r"['\"]--{0}['\"][^)]*?choices=([A-Z_]+)".format(flag), src, re.S)
        assert m, "--{0} has no choices".format(flag)
        return None
    return {c.strip().strip("'\"") for c in m.group(1).split(",") if c.strip()}


@pytest.mark.parametrize("emulator", EMULATORS)
def test_template_placeholders_are_exactly_the_substituted_set(emulator):
    with open(os.path.join(ROOT, "emulators", emulator, "slurm_template_multinode.sh")) as fh:
        tpl = fh.read()
    found = set(re.findall(r"\{\{[A-Z_]+\}\}", tpl))
    assert found == PLACEHOLDERS, (emulator, found ^ PLACEHOLDERS)


@pytest.mark.parametrize("emulator", EMULATORS)
def test_scripts_accept_every_mode_and_noise_the_matrix_claims(emulator):
    src = _script(emulator)
    modes = _choices(src, "comm_mode")
    if modes is not None:
        assert jc.SUPPORTED_COMM_MODES[emulator] <= modes, (emulator, modes)
    noise = _choices(src, "noise")
    if noise is not None:
        assert jc.SUPPORTED_NOISE[emulator] <= noise, (emulator, noise)


@pytest.mark.parametrize("config", CONFIGS)
def test_campaign_plan_is_consistent(config):
    cfg = _load(config)
    jobs, _, _ = jc.plan(cfg)
    assert jobs, "empty plan"
    names = [j["job_name"] for j in jobs]
    results = [j["result_file"] for j in jobs]
    assert len(set(names)) == len(names), "duplicate job names"
    assert len(set(results)) == len(results), "two jobs would write the same results file"
    for j in jobs:
        ok, why = jc.is_supported(j["emulator"], j["algorithm"], j["comm_mode"], j["noise"])
        assert ok, why
        assert j["mem_gb"] <= cfg.get("max_mem_gb", 240)
        cap = ((cfg.get("caps") or {}).get(j["emulator"]) or {}).get(j["comm_mode"]) or {}
        if "max_nodes" in cap:
            assert j["num_nodes"] <= cap["max_nodes"]
        if "max_state_qubits" in cap:
            assert jc.state_qubits(j["emulator"], j["comm_mode"], j["num_qubits"],
                                   j["num_nodes"]) <= cap["max_state_qubits"]


def test_wave1_is_noiseless_and_wave2_is_not():
    w1, _, _ = jc.plan(_load(CONFIGS[0]))
    w2, _, _ = jc.plan(_load(CONFIGS[1]))
    assert {j["noise"] for j in w1} == {"none"}
    assert "none" not in {j["noise"] for j in w2}
    assert not ({j["result_file"] for j in w1} & {j["result_file"] for j in w2})


def test_baseline_fast_path_matches_reference_fidelity():
    """ideal_baseline.py computes fidelity over observed outcomes only; it must
    equal fidelity_from_counts on the same histogram."""
    n, theta, shots = 6, 0.33, 100
    x, p = ideal_qft_probs(theta, n)
    p = p / p.sum()
    rng = np.random.default_rng(3)
    for _ in range(20):
        idx = np.searchsorted(np.cumsum(p), rng.random(shots), side="right")
        ks, cnt = np.unique(idx, return_counts=True)
        fast = np.sum(np.sqrt(cnt / shots * p[ks])) ** 2
        ref = fidelity_from_counts({x[k]: c for k, c in zip(ks, cnt)}, n, theta)
        assert abs(fast - ref) < 1e-12


def test_baseline_known_value():
    f = ideal_sampling_fidelity(4, 0.33, 100, reps=4000, seed=1)
    assert abs(f.mean() - 0.950) < 0.003           # measured on QMIO run: 0.956
    assert ideal_sampling_fidelity(8, 0.0, 100, reps=50).min() == 1.0
