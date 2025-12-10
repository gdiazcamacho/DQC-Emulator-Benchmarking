import os
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from qiskit.quantum_info import hellinger_fidelity
import ast
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D



def parse_csv_results(base_dir):
    data_frames = []

    for emulator_dir in os.listdir(base_dir):
        results_dir = os.path.join(base_dir, emulator_dir, 'results')
        if os.path.isdir(results_dir):
            for result_file in os.listdir(results_dir):
                if result_file.endswith('.csv'):
                    file_path = os.path.join(results_dir, result_file)
                    df = pd.read_csv(file_path)
                    data_frames.append(df)

    # Concatenate all data frames
    combined_df = pd.concat(data_frames, ignore_index=True)
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

def fidelity_emulator(emulator, df):

    qubit_values = df["Num Qubits"].unique()
    theta_values = df["Theta"].unique()

    fid_nq = {}
    for nq in sorted(qubit_values):
        fid = []
        for theta in sorted(theta_values):
            result_row1 = df.query(
                "Emulator == @emulator and Nodes == 1 and `Num Qubits` == @nq and Theta == @theta"
            )
            result_row2 = df.query(
                "Emulator == @emulator and Nodes == 2 and `Num Qubits` == @nq and Theta == @theta"
            )

            if not result_row1.empty:
                if not result_row2.empty:
                    counts1 = ast.literal_eval(result_row1.iloc[0]["Result"])
                    counts2 = ast.literal_eval(result_row2.iloc[0]["Result"])
                    fid.append(hellinger_fidelity(counts1, counts2))

        if fid:
            fid_nq[nq]=np.mean(fid)
    return fid_nq


def ideal_qft_probs(theta, nq, eps=1e-12):
    N = 2**nq
    ks = np.arange(N)
    x = ks / N
    angle = np.pi * (theta - x)
    
    probs = np.zeros(N)
    small = np.abs(angle) < eps          # indices very close to zero
    not_small = ~small
    
    probs[small] = 1.0                   # probability concentrated at the peak
    probs[not_small] = np.sin(N * angle[not_small])**2 / (N**2 * np.sin(angle[not_small])**2)
    
    return x, probs


def classical_fidelity(Psim, Pideal):
    """
    Compute classical fidelity between two discrete probability distributions.
    Both Psim and Pideal must be arrays of the same length and sum to 1.
    """
    Psim = np.array(Psim)
    Pideal = np.array(Pideal)
    
    # normalize just in case
    Psim = Psim / Psim.sum()
    Pideal = Pideal / Pideal.sum()
    
    fid = np.sum(np.sqrt(Psim * Pideal))**2
    return fid

def compute_full_fidelity(counts, nq, theta):
    """
    Compute fidelity between measured counts and ideal IQFT distribution.
    Pads counts with zeros for missing outcomes.
    """
    N = 2**nq
    x_full = np.arange(N) / N

    # Build Psim_full (padded with zeros)
    Psim_full = np.zeros(N)
    for k, v in counts.items():
        # If keys are floats already normalized in [0,1], convert to index
        idx = int(round(k * N))
        Psim_full[idx] = v
    Psim_full /= Psim_full.sum()

    # Ideal distribution
    _, Pideal_full = ideal_qft_probs(theta, nq)

    # Classical fidelity
    fid = np.sum(np.sqrt(Psim_full * Pideal_full))**2
    return fid

def fidelity_per_nq_nodes(emulator, df):
    """
    Loop over all (nq, nodes) tuples and compute mean fidelity across thetas.
    """
    qubit_values = sorted(df["Num Qubits"].unique())
    node_values  = sorted(df["Nodes"].unique())
    theta_values = sorted(df["Theta"].unique())

    fid_dict = {}

    for nq in qubit_values:
        for nodes in node_values:
            fid_thetas = []

            for theta in theta_values:
                row = df.query(
                    "Emulator == @emulator and `Num Qubits` == @nq and Nodes == @nodes and Theta == @theta"
                )
                if row.empty:
                    continue

                counts = ast.literal_eval(row.iloc[0]["Result"])

                fid = compute_full_fidelity(counts, nq, theta)
                fid_thetas.append(fid)

            if fid_thetas:
                fid_dict[(nq, nodes)] = np.mean(fid_thetas)

    return fid_dict

def plot_fidelity_curves(fid_dicts, emulator_cmaps, emulator_styles, variable="Num Qubits"):
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
    plt.show()

   
   
# Base directory containing the emulators
base_dir = "../emulators"

# Parse results and plot scaling
df = parse_csv_results(base_dir)

time_memory_vs_qubits(df, save_path="scaling_2panel")

# Colormaps
emulator_cmaps = {
    "squidasm":   truncate_colormap(cm.ocean,     0.2, 0.8),
    "qiskit_aer": truncate_colormap(cm.Purples_r, 0.0, 0.6),
    "interlin-q": truncate_colormap(cm.Greens_r,  0.0, 0.6),
    "squanch":    truncate_colormap(cm.Oranges_r, 0.0, 0.8),
}

# Markers and line styles
emulator_styles = {
    "qiskit_aer": ('o', 'solid'),
    "squidasm":   ('s', 'dashdot'),
    "squanch":    ('^', 'dashed'),
    "interlin-q": ('D', 'dotted'),
}

# Example data
fid_squanch = fidelity_per_nq_nodes("squanch", df)
fid_squidasm = fidelity_per_nq_nodes("squidasm", df)
fid_interlinq = fidelity_per_nq_nodes("interlin-q", df)
fid_aer = fidelity_per_nq_nodes("qiskit_aer", df)

plot_fidelity_curves(
    [
        ("squidasm", fid_squidasm),
        ("qiskit_aer", fid_aer),
        ("squanch", fid_squanch),
        ("interlin-q", fid_interlinq),
    ],
    emulator_cmaps,
    emulator_styles,
    variable="Num Qubits"
)
