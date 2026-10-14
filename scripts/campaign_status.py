"""
Progress and sanity report for a campaign.

    python scripts/campaign_status.py campaign-v1
    python scripts/campaign_status.py campaign-v1 --missing     # list jobs with no result
    python scripts/campaign_status.py campaign-v1 --resubmit    # print sbatch lines for them
    python scripts/campaign_status.py campaign-v1 --submit [--emulator E] [--mode M] [--limit N]
        # sbatch every job that has no result and is not already queued

Reads campaigns/<campaign>/manifest_*.csv (written by job_creator.py) and, for
every job, checks that its results CSV exists and:

  * was produced by the campaign's commit (Code Commit column);
  * for noiseless runs, that its fidelity is not below the ideal-sampling
    interval (results/baselines/ideal_sampling.csv, from ideal_baseline.py).
    A noiseless emulator should land inside [P2.5, P97.5] about 95% of the
    time; one far below it is a correctness problem, not noise.
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_baseline():
    path = os.path.join(REPO_ROOT, "results", "baselines", "ideal_sampling.csv")
    if not os.path.exists(path):
        return None
    return {(int(r["Num Qubits"]), float(r["Theta"]), int(r["Shots"])): r
            for r in _read_csv(path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("campaign")
    ap.add_argument("--missing", action="store_true", help="list jobs without a result")
    ap.add_argument("--resubmit", action="store_true", help="print sbatch commands for them")
    ap.add_argument("--submit", action="store_true",
                    help="sbatch missing jobs that are not already in the queue")
    ap.add_argument("--emulator", action="append", help="restrict --submit/--missing")
    ap.add_argument("--mode", action="append", help="restrict --submit/--missing")
    ap.add_argument("--limit", type=int, default=None, help="submit at most N jobs")
    ap.add_argument("--slack", type=float, default=0.02,
                    help="flag noiseless fidelity below P2.5 minus this (default 0.02)")
    a = ap.parse_args()

    import glob
    mdir = os.path.join(REPO_ROOT, "campaigns", a.campaign)
    manifest = []
    for mp in sorted(glob.glob(os.path.join(mdir, "manifest_*.csv"))):
        manifest += _read_csv(mp)
    if not manifest:
        sys.exit("no manifest_*.csv in {0}; run job_creator.py first".format(mdir))
    commits = set()
    import yaml
    for cp in glob.glob(os.path.join(mdir, "config_*.yaml")):
        with open(cp) as fh:
            commits.add((yaml.safe_load(fh) or {}).get("commit"))
    if len(commits) > 1:
        print("[warn] waves were generated from different commits: {0}".format(sorted(commits)))
    commit = commits.pop() if len(commits) == 1 else None
    baseline = load_baseline()

    done, missing, wrong_commit, low = Counter(), [], [], []
    total = Counter()
    for job in manifest:
        key = (job["emulator"], job["comm_mode"], job["noise"])
        total[key] += 1
        path = os.path.join(REPO_ROOT, job["result_file"])
        if not os.path.exists(path):
            missing.append(job)
            continue
        done[key] += 1
        row = _read_csv(path)[0]
        if commit and row.get("Code Commit") not in (commit, None):
            wrong_commit.append((job["job_name"], row.get("Code Commit")))
        if baseline and job["noise"] == "none" and row.get("Fidelity"):
            b = baseline.get((int(job["num_qubits"]), float(job["theta"]), int(job["shots"])))
            if b and float(row["Fidelity"]) < float(b["Fidelity P2.5"]) - a.slack:
                low.append((job["job_name"], float(row["Fidelity"]),
                            float(b["Fidelity P2.5"])))

    print("Campaign {0}  (commit {1})".format(a.campaign, commit or "?"))
    print("{0:<12} {1:<8} {2:<13} {3:>9}".format("emulator", "mode", "noise", "done/total"))
    for key in sorted(total):
        print("{0:<12} {1:<8} {2:<13} {3:>4}/{4:<4}".format(*key, done[key], total[key]))
    print("Total: {0}/{1} results present".format(sum(done.values()), len(manifest)))
    if baseline is None:
        print("[info] no results/baselines/ideal_sampling.csv: run scripts/ideal_baseline.py")
    if wrong_commit:
        print("\n[warn] {0} result(s) from a different commit, e.g. {1}".format(
            len(wrong_commit), wrong_commit[:3]))
    if low:
        print("\n[warn] {0} noiseless result(s) below the ideal-sampling interval:".format(len(low)))
        for name, f, p in sorted(low, key=lambda t: t[1])[:30]:
            print("  {0:<55} F={1:.4f}  (ideal P2.5={2:.4f})".format(name, f, p))
    if a.emulator:
        missing = [j for j in missing if j["emulator"] in a.emulator]
    if a.mode:
        missing = [j for j in missing if j["comm_mode"] in a.mode]
    if a.submit:
        import getpass
        import subprocess
        try:
            queued = set(subprocess.check_output(
                ["squeue", "-u", getpass.getuser(), "-h", "-o", "%j"],
                text=True).split())
        except Exception as exc:
            sys.exit("squeue failed ({0}); not submitting".format(exc))
        todo = [j for j in missing if j["job_name"] not in queued]
        if a.limit is not None:
            todo = todo[:a.limit]
        for j in todo:
            subprocess.check_call(["sbatch", "-D", os.path.join(
                REPO_ROOT, "emulators", j["emulator"], "jobs"),
                os.path.join(REPO_ROOT, j["job_script"])])
        print("\nSubmitted {0} job(s); {1} already queued, {2} left.".format(
            len(todo), len(missing) - len([j for j in missing if j["job_name"] not in queued]),
            len([j for j in missing if j["job_name"] not in queued]) - len(todo)))
    if a.missing or a.resubmit:
        by = defaultdict(list)
        for j in missing:
            by[j["emulator"]].append(j)
        for emu in sorted(by):
            print("\nMissing for {0}: {1}".format(emu, len(by[emu])))
            for j in by[emu]:
                if a.resubmit:
                    print("sbatch -D emulators/{0}/jobs {1}".format(emu, j["job_script"]))
                else:
                    print("  " + j["job_name"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
