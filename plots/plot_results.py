"""
Figure generation for the DQC emulator benchmarking suite.

Run from anywhere:

    python plots/plot_results.py --save-path plots/scaling_2panel
"""

import argparse
import ast
import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Shared fidelity reference, so figures and per-run CSV columns cannot drift apart.
import sys
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from common.dqc_common import fidelity_from_counts


def parse_csv_results(base_dir=None, algorithm="qft"):
    """Concatenate every one-row result CSV under emulators/*/results/.

    Rows are filtered by `Algorithm`. Legacy files written before that column
    existed are backfilled as "qft", which is what they were. Without this
    filter, adding a second benchmark silently contaminates the QFT figures.
    """
    if base_dir is None:
        base_dir = os.path.join(REPO_ROOT, "emulators")

    data_frames = []
    for emulator_dir in sorted(os.listdir(base_dir)):
        results_dir = os.path.join(base_dir, emulator_dir, "results")
        if not os.path.isdir(results_dir):
            continue
        for result_file in sorted(os.listdir(results_dir)):
            if not result_file.endswith(".csv"):
                continue
            df = pd.read_csv(os.path.join(results_dir, result_file))
            data_frames.append(df)

    if not data_frames:
        raise SystemExit(
            "No result CSVs found under {0}. Run the sweep first "
            "(scripts/job_creator.py then scripts/job_submitter_all.sh).".format(base_dir)
        )

    combined_df = pd.concat(data_frames, ignore_index=True)

    if "Algorithm" not in combined_df.columns:
        combined_df["Algorithm"] = "qft"
    else:
        combined_df["Algorithm"] = combined_df["Algorithm"].fillna("qft")

    if algorithm is not None:
        combined_df = combined_df[combined_df["Algorithm"] == algorithm].reset_index(drop=True)
        if combined_df.empty:
            raise SystemExit("No rows with Algorithm == {0!r}.".format(algorithm))

    return combined_df


def truncate_colormap(cmap, minval=0.0, maxval=0.8, n=100):
    """
    Truncate a colormap to avoid very light or very dark edges.
    """
    new_cmap = LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{minval:.2f},{maxval:.2f})",
        cmap(np.linspace(minval, maxval, n))
    )
    return new_cmap
    
    
def time_memory_vs_qubits(
    df,
    log_scale=True,
    save_path=None,
    figsize=(16, 8),
    linewidth=2.0,
    alpha=0.95,
):
    """
    Side-by-side plot:
      (0) Execution Time vs Qubits
      (1) Memory Usage vs Qubits
    Color encodes Nodes (per-emulator gradient)
    One legend entry per emulator (marker + color)
    Error bars: mean +- std over repetitions.
    """

    # Desired emulator order and normalization
    emulator_order = ["qiskit-aer", "squidasm", "squanch", "interlin-q"]
    def norm_name(name): return str(name).lower().replace("_", "-")

    # Filter and order DataFrame emulators
    emulators_raw_all = df["Emulator"].unique()
    emulators_raw = [e for e in emulators_raw_all if norm_name(e) in emulator_order]
    emulators_raw.sort(key=lambda e: emulator_order.index(norm_name(e)))
    emulators = [norm_name(e) for e in emulators_raw]

    nodes_list = sorted(df["Nodes"].unique())
    qubit_list = sorted(df["Num Qubits"].unique())

    # Color maps per emulator
    emulator_cmaps = {
        "squidasm": truncate_colormap(cm.ocean, 0.2, 0.8),
        "qiskit-aer": truncate_colormap(cm.Purples_r, 0.0, 0.6),
        "interlin-q": truncate_colormap(cm.Greens_r, 0.0, 0.6),
        "squanch": truncate_colormap(cm.Oranges_r, 0.0, 0.8),
    }

    # Custom markers and line styles
    emulator_styles = {
        "qiskit-aer": ('o', 'solid'),
        "squidasm":   ('s', 'dashdot'),
        "squanch":    ('^', 'dashed'),
        "interlin-q": ('D', 'dotted'),
    }

    # Helper to sample cmap evenly for k steps
    def samples_for_k(k):
        return [0.6] if k == 1 else list(np.linspace(0.1, 0.95, k))

    # Create figure and axes
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    titles = ["Execution Time vs Qubits", "Memory Usage vs Qubits"]
    ycols = ["Execution Time (s)", "Peak Memory (KB)"]
    ylabels = ["Execution Time (s)", "Peak Memory (KB)"]

    for i, ax in enumerate(axs):
        ycol, ylabel = ycols[i], ylabels[i]
        for emu_raw, emu in zip(emulators_raw, emulators):
            marker, linestyle = emulator_styles.get(emu, ('o', 'solid'))
            cmap = emulator_cmaps.get(emu, cm.viridis)
            tvals = samples_for_k(len(nodes_list))

            for t, nodes in zip(tvals, nodes_list):
                subset = df[
                    (df["Emulator"].apply(norm_name) == emu) &
                    (df["Nodes"] == nodes)
                ]
                if subset.empty:
                    continue

                stats = subset.groupby("Num Qubits")[ycol].agg(['mean', 'std']).reset_index()
                stats['std'] = stats['std'].fillna(0.0)

                ax.errorbar(
                    stats["Num Qubits"], stats["mean"], yerr=stats["std"],
                    color=cmap(t), marker=marker, linestyle=linestyle,
                    linewidth=linewidth, alpha=alpha, capsize=3,
                )

        ax.set_title(titles[i], fontsize=17)
        ax.set_xlabel("Qubits", fontsize=15)
        ax.set_ylabel(ylabel, fontsize=15)
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)

    # Legend (inside the second plot, boxed)
    handles, labels = [], []
    for emu_raw, emu in zip(emulators_raw, emulators):
        marker, linestyle = emulator_styles.get(emu, ('o', 'solid'))
        mid_color = emulator_cmaps.get(emu, cm.viridis)(0.6)
        h = Line2D(
            [0], [0],
            color=mid_color, marker=marker, linestyle=linestyle,
            linewidth=2.4
        )
        handles.append(h)
        labels.append(emu_raw)

    axs[1].legend(
        handles, labels,
        loc="upper left",  # tweak if overlapping
        frameon=True,
        framealpha=0.9,
        facecolor="white",
        edgecolor="gray",
        title="Emulators",
        fontsize=13,
        title_fontsize=14,
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])

    if save_path:
        fig.savefig(f"{save_path}.png", dpi=200, bbox_inches="tight")
        fig.savefig(f"{save_path}.svg", format="svg", bbox_inches="tight")
        print(f"Saved to {save_path}.png and {save_path}.svg")
    else:
        plt.show()

def fidelity_per_nq_nodes(emulator, df):
    """Mean fidelity over thetas for each (num_qubits, nodes) pair.

    Uses the per-run `Fidelity` column when present. Older CSVs predate that
    column, so fidelity is recomputed from the stored counts as a fallback --
    both paths call the same `fidelity_from_counts`, so they cannot diverge.
    """
    qubit_values = sorted(df["Num Qubits"].unique())
    node_values = sorted(df["Nodes"].unique())
    theta_values = sorted(df["Theta"].unique())

    has_column = "Fidelity" in df.columns
    fid_dict = {}

    for nq in qubit_values:
        for nodes in node_values:
            fid_thetas = []
            for theta in theta_values:
                row = df.query(
                    "Emulator == @emulator and `Num Qubits` == @nq and "
                    "Nodes == @nodes and Theta == @theta"
                )
                if row.empty:
                    continue

                value = row.iloc[0]["Fidelity"] if has_column else None
                if value is None or pd.isna(value):
                    counts = ast.literal_eval(row.iloc[0]["Result"])
                    value = fidelity_from_counts(counts, nq, theta)
                fid_thetas.append(float(value))

            if fid_thetas:
                fid_dict[(nq, nodes)] = float(np.mean(fid_thetas))

    return fid_dict


def plot_fidelity_curves(fid_dicts, emulator_cmaps, emulator_styles,
                         variable="Num Qubits", save_path=None):
    if isinstance(fid_dicts, dict):
        fid_dicts = [("emulator", fid_dicts)]

    plt.figure(figsize=(8, 6))

    for label, fids in fid_dicts:
        if not fids:  # skip empty results
            print(f"Skipping {label} - no fidelity data found.")
            continue

        key = label.lower()
        cmap = emulator_cmaps.get(key, cm.viridis)
        marker, linestyle = emulator_styles.get(key, ('o', 'solid'))

        nq_vals = sorted({nq for (nq, _) in fids.keys()})
        node_vals = sorted({nodes for (_, nodes) in fids.keys()})
        if not node_vals or not nq_vals:
            print(f"Skipping {label} - empty nq/node values.")
            continue

        norm = plt.Normalize(min(node_vals), max(node_vals))
        colors = [cmap(norm(n)) for n in node_vals]
        legend_color = cmap(0.5)

        if variable == "Num Qubits":
            for n_idx, nodes in enumerate(node_vals):
                x, y = [], []
                for nq in nq_vals:
                    if (nq, nodes) in fids:
                        x.append(nq)
                        y.append(fids[(nq, nodes)])
                if x:
                    plt.plot(x, y, marker=marker, linestyle=linestyle,
                             color=colors[n_idx], alpha=0.9, lw=2, label=None)

        elif variable == "Nodes":
            for q_idx, nq in enumerate(nq_vals):
                x, y = [], []
                for nodes in node_vals:
                    if (nq, nodes) in fids:
                        x.append(nodes)
                        y.append(fids[(nq, nodes)])
                if x:
                    plt.plot(x, y, marker=marker, linestyle=linestyle,
                             color=colors[q_idx], alpha=0.9, lw=2, label=None)

        else:
            raise ValueError("variable must be 'Num Qubits' or 'Nodes'")

        plt.plot([], [], color=legend_color, marker=marker,
                 linestyle=linestyle, lw=2, label=label)

    plt.xlabel(variable)
    plt.ylabel("Mean Fidelity")
    plt.title(f"Fidelity vs {variable}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}.png", dpi=200, bbox_inches="tight")
        plt.savefig(f"{save_path}.svg", format="svg", bbox_inches="tight")
        print(f"Saved to {save_path}.png and {save_path}.svg")
    else:
        plt.show()

   
   
def main():
    parser = argparse.ArgumentParser(description="Generate benchmark figures.")
    parser.add_argument("--base-dir", default=None,
                        help="emulators/ directory (default: repo-relative)")
    parser.add_argument("--algorithm", default="qft",
                        help="Algorithm to plot (default: qft)")
    parser.add_argument("--save-path", default=None,
                        help="Path prefix for saved figures; omit to show interactively")
    args = parser.parse_args()

    df = parse_csv_results(args.base_dir, algorithm=args.algorithm)

    save_path = args.save_path
    if save_path is None and os.environ.get("MPLBACKEND", "") == "Agg":
        save_path = os.path.join(REPO_ROOT, "plots", "scaling_2panel")

    time_memory_vs_qubits(df, save_path=save_path)

    emulator_cmaps = {
        "squidasm":   truncate_colormap(cm.ocean,     0.2, 0.8),
        "qiskit_aer": truncate_colormap(cm.Purples_r, 0.0, 0.6),
        "interlin-q": truncate_colormap(cm.Greens_r,  0.0, 0.6),
        "squanch":    truncate_colormap(cm.Oranges_r, 0.0, 0.8),
    }
    emulator_styles = {
        "qiskit_aer": ("o", "solid"),
        "squidasm":   ("s", "dashdot"),
        "squanch":    ("^", "dashed"),
        "interlin-q": ("D", "dotted"),
    }

    fid_dicts = [(name, fidelity_per_nq_nodes(name, df))
                 for name in ("squidasm", "qiskit_aer", "squanch", "interlin-q")]

    plot_fidelity_curves(
        fid_dicts, emulator_cmaps, emulator_styles, variable="Num Qubits",
        save_path=None if save_path is None else save_path + "_fidelity",
    )


if __name__ == "__main__":
    main()
