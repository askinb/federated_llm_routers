result_out_folder = "./paper_results/"

## Imports
import json
import os
if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# NEW: tick formatting helpers
import matplotlib.ticker as mticker

# Choose ONE: mathtext (no LaTeX needed) or full LaTeX rendering
USE_TEX = False   # set to True if you have a LaTeX distribution installed

CFG_MATHTEXT_CM = {
    "text.usetex": False,            # don't call external LaTeX
    "mathtext.fontset": "cm",        # Computer Modern math
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
    "axes.unicode_minus": False,     # show proper minus sign with serif
    "pdf.fonttype": 42,              # better text embedding
    "ps.fonttype": 42,
}

CFG_USETEX_CM = {
    "text.usetex": True,             # render ALL text via LaTeX
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

plt.rcParams.update(CFG_MATHTEXT_CM)

result_colors = {
    "mlp": "tab:blue",
    "kmeans": "tab:orange",
    "global": "tab:red",
    "centralized": "tab:green",
    "client-local": "tab:brown",
    "personalized": "tab:purple",
    "ensemble": "tab:gray",
    "adaptive-ensemble": "tab:olive",
}
result_linestyles = {
    "mlp": "-",
    "kmeans": "--",}


# Define font sizes at the top of the file or here as parameters.
FONT_AXISLABEL = 12  # default +2
FONT_TITLE = 12      # default +2
FONT_TICK = 12       # default +2
FONT_LEGEND = 10     # default +2
FONT_SUPTITLE = 14  # default +2

# NEW: Consistent scientific cost-axis formatting across all plots.
# Goal: show ticks like "2.5" and an axis scale like "1e-3" (instead of decimals like 0.0025).
COST_SCI_POWER = -3  # Force a 1e-3 scale factor on the x-axis.

def format_cost_axis(ax, show_offset_text: bool = True):
    """
    Apply a fixed scientific notation scale on the x-axis (cost), e.g. ticks "2.5"
    with an offset "1e-3" shown near the axis. Use show_offset_text=False to hide
    the "1e-3" on crowded subplot grids while keeping tick labels short.
    """
    # Use a ScalarFormatter so Matplotlib shows an offset like "1e-3" rather than long decimals.
    fmt = mticker.ScalarFormatter(useMathText=False)
    fmt.set_scientific(True)
    fmt.set_powerlimits((COST_SCI_POWER, COST_SCI_POWER))
    fmt.set_useOffset(True)

    ax.xaxis.set_major_formatter(fmt)
    ax.ticklabel_format(
        axis="x",
        style="sci",
        scilimits=(COST_SCI_POWER, COST_SCI_POWER),
        useMathText=False,
    )

    # Make offset text (e.g., "1e-3") match tick font size, and optionally hide it.
    ax.xaxis.get_offset_text().set_fontsize(FONT_TICK)
    ax.xaxis.get_offset_text().set_visible(show_offset_text)


if not os.path.exists(result_out_folder):
    os.makedirs(result_out_folder)

## Load results
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"base dir: {BASE_DIR}")
MLP_RESULTS_PATH = os.path.join(BASE_DIR, "mlp_out", "mlp_experiment_results.json")
KMEANS_RESULTS_PATH = os.path.join(BASE_DIR, "kmeans_out", "kmeans_experiment_results.json")

mlp_results = None
if os.path.exists(MLP_RESULTS_PATH):
    with open(MLP_RESULTS_PATH, "r", encoding="utf-8") as f:
        mlp_results = json.load(f)
else:
    print(f"Missing results: {MLP_RESULTS_PATH}")

kmeans_results = None
if os.path.exists(KMEANS_RESULTS_PATH):
    with open(KMEANS_RESULTS_PATH, "r", encoding="utf-8") as f:
        kmeans_results = json.load(f)
else:
    print(f"Missing results: {KMEANS_RESULTS_PATH}")


########################################################
# Plot Global vs Centralized (MLP and K-Means) on the same plot
########################################################

if (mlp_results is not None) and (kmeans_results is not None):
    evals_mlp = mlp_results["evaluations"]["global_test"]
    evals_km = kmeans_results["evaluations"]["global_test"]
    g_mlp = evals_mlp["global_model"]
    c_mlp = evals_mlp["centralized_model"]
    g_km = evals_km["global_model"]
    c_km = evals_km["centralized_model"]

    color_mlp_global = "tab:blue"
    color_mlp_centralized = "tab:orange"
    color_km_global = "tab:green"
    color_km_centralized = "tab:red"

    fig, ax = plt.subplots(figsize=(6, 4))
    # Plot lines
    ax.plot([p[0] for p in g_mlp["curve"]], [p[1] for p in g_mlp["curve"]], marker="o", markevery=5,
            label=f"MLP-Federated ({g_mlp['auc']:.2f})", color=color_mlp_global)
    ax.plot([p[0] for p in c_mlp["curve"]], [p[1] for p in c_mlp["curve"]], marker="o", markevery=5,
            label=f"MLP-Centralized ({c_mlp['auc']:.2f})", color=color_mlp_centralized)
    ax.plot([p[0] for p in g_km["curve"]], [p[1] for p in g_km["curve"]], marker="o", markevery=5,
            label=f"K-Means-Federated ({g_km['auc']:.2f})", color=color_km_global)
    ax.plot([p[0] for p in c_km["curve"]], [p[1] for p in c_km["curve"]], marker="o", markevery=5,
            label=f"K-Means-Centralized ({c_km['auc']:.2f})", color=color_km_centralized)
    ax.set_ylim(bottom=0.5)

    ax.set_xlabel("Avg Cost (in $)", fontsize=FONT_AXISLABEL+1)
    ax.set_ylabel("Accuracy", fontsize=FONT_AXISLABEL+1)
    ax.set_title("Federated vs Centralized Models (MLP & K-Means)", fontsize=FONT_TITLE+1)
    ax.tick_params(axis='both', labelsize=FONT_TICK)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=FONT_LEGEND+1)

    # NEW: consistent scientific formatting for cost axis
    format_cost_axis(ax, show_offset_text=True)

    plt.tight_layout()
    plt.savefig(os.path.join(result_out_folder, "global_vs_centralized_both.pdf"))
    plt.close()
else:
    print("Skip combined plot (MLP & K-Means global vs centralized): at least one result not loaded.")

########################################################
# Plot both MLP and K-Means global vs all client-local on global test in a single figure with subplots
########################################################

if (mlp_results is not None) and (kmeans_results is not None):
    # Gather MLP data
    mlp_global_entry = mlp_results["evaluations"]["global_test"]["global_model"]
    mlp_per_client_global = mlp_results["evaluations"]["per_client_on_global_test"]
    mlp_client_ids = sorted(mlp_per_client_global.keys(), key=lambda x: int(x))

    # Gather K-Means data
    kmeans_global_entry = kmeans_results["evaluations"]["global_test"]["global_model"]
    kmeans_per_client_global = kmeans_results["evaluations"]["per_client_on_global_test"]
    kmeans_client_ids = sorted(kmeans_per_client_global.keys(), key=lambda x: int(x))

    fig, axs = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    plt.subplots_adjust(hspace=0.01, top=1.4)  # reduce white space above plots

    # ========== Top: MLP ==========
    ax = axs[0]
    mlp_global_color = result_colors.get("mlp", "tab:blue")
    ax.plot(
        [p[0] for p in mlp_global_entry["curve"]],
        [p[1] for p in mlp_global_entry["curve"]],
        marker="o",
        markevery=5,
        markersize=6,
        linewidth=2.5,
        label=f"Federated ({mlp_global_entry['auc']:.2f})",
        color=mlp_global_color
    )
    cmap_mlp = plt.get_cmap("tab20")
    for i, cid in enumerate(mlp_client_ids):
        entry = mlp_per_client_global[cid]["local_only"]
        color = cmap_mlp(i % 20)
        ax.plot(
            [p[0] for p in entry["curve"]],
            [p[1] for p in entry["curve"]],
            marker="o",
            markevery=5,
            markersize=4,
            linewidth=1.3,
            label=f"Client {int(cid)+1} ({entry['auc']:.2f})",
            color=color
        )
    ax.set_ylabel("Accuracy (MLP)", fontsize=FONT_AXISLABEL+2)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_ylim(bottom=0.4)

    # NEW: consistent scientific formatting for cost axis
    # (hide offset on top axis since the shared x-axis label is on the bottom subplot)
    format_cost_axis(ax, show_offset_text=False)

    # Top legend at bottom of top plot, 3 cols, small spacing, semi-transparent gray bg, -2 font
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles, labels,
        fontsize=max(FONT_LEGEND , 1)+1.2, ncol=3, loc="lower center",
        bbox_to_anchor=(0.56, 0),  # move below plot area, close to axis
        columnspacing=0.08, handletextpad=0.13, borderaxespad=0.14
    )
    legend.get_frame().set_edgecolor("none")

    # ========== Bottom: K-Means ==========
    ax = axs[1]
    kmeans_global_color = result_colors.get("mlp", "tab:blue")
    ax.plot(
        [p[0] for p in kmeans_global_entry["curve"]],
        [p[1] for p in kmeans_global_entry["curve"]],
        marker="o",
        markevery=5,
        markersize=6,
        linewidth=2.5,
        label=f"Federated ({kmeans_global_entry['auc']:.2f})",
        color=kmeans_global_color
    )
    cmap_kmeans = plt.get_cmap("tab20")
    for i, cid in enumerate(kmeans_client_ids):
        entry = kmeans_per_client_global[cid]["local_only"]
        color = cmap_kmeans(i % 20)
        ax.plot(
            [p[0] for p in entry["curve"]],
            [p[1] for p in entry["curve"]],
            marker="o",
            markevery=5,
            markersize=4,
            linewidth=1.3,
            label=f"Client {int(cid)+1} ({entry['auc']:.2f})",
            color=color
        )
    ax.set_ylabel("Accuracy (K-Means)", fontsize=FONT_AXISLABEL+2)
    ax.set_xlabel("Avg Cost (in $)", fontsize=FONT_AXISLABEL+2)
    ax.tick_params(axis="both", labelsize=FONT_TICK)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.set_ylim(bottom=0.4)

    # NEW: consistent scientific formatting for cost axis
    format_cost_axis(ax, show_offset_text=True)

    # Bottom legend at bottom of lower plot, similar style, just further down so they don't overlap
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles, labels,
        fontsize=max(FONT_LEGEND , 1)+1.2, ncol=3, loc="lower center",
        bbox_to_anchor=(0.56, 0),  # push a bit lower than above
        columnspacing=0.08, handletextpad=0.13, borderaxespad=0.14
    )
    legend.get_frame().set_edgecolor("none")

    # Title as close as possible to top plot
    fig.suptitle(
        "Global Test Performance of Federated vs Client-Local (No-FL) Models", fontsize=FONT_SUPTITLE+1, y=0.93
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(
        os.path.join(result_out_folder, "global_vs_local_only_global_test_MLP_KMEANS_subplot.pdf"),
        bbox_inches="tight"
    )
    plt.close()


########################################################
## Plot: Client-Local (No-FL) vs Global on LOCAL TEST splits (Grid, MLP)
########################################################
if mlp_results is None:
    print("Skip MLP local vs global on local test grid: results not loaded.")
else:
    per_client = mlp_results["evaluations"]["per_client_local_test"]
    client_ids = sorted(per_client.keys(), key=lambda x: int(x))

    nrows, ncols = 4, 3
    cap = nrows * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8), sharex=False, sharey=False)
    axes = axes.flatten()

    local_aucs = []
    global_aucs = []
    used = 0
    for idx, cid in enumerate(client_ids[:cap]):
        ax = axes[idx]
        used += 1
        entry = per_client[cid]

        loc = entry["local_only"]
        glob = entry["global"]
        local_aucs.append(loc["auc"])
        global_aucs.append(glob["auc"])

        ax.plot([p[0] for p in loc["curve"]],
                [p[1] for p in loc["curve"]],
                marker="o", markersize=3.5, markevery=5, linestyle="-", linewidth=2.2,
                color=result_colors.get("client-local", "tab:brown"),
                label=f"Client-Local ({loc['auc']:.2f})")
        ax.plot([p[0] for p in glob["curve"]],
                [p[1] for p in glob["curve"]],
                marker="o", markersize=3.5, markevery=5, linestyle="-", linewidth=2.2,
                color=result_colors.get("global", "tab:red"),
                label=f"Federated ({glob['auc']:.2f})")
        ax.set_title(f"Client {int(cid) + 1}", fontsize=FONT_TICK)
        ax.grid(True, linestyle="-", alpha=0.4)
        ax.legend(fontsize=max(FONT_LEGEND - 2, 8)+2)

        ax.set_ylim(bottom=0.4)

        # Set ylabel "Accuracy" on the leftmost plots of each row
        if (idx % ncols) == 0:
            ax.set_ylabel("Accuracy", fontsize=FONT_AXISLABEL)
        # Set xlabel "Cost" on the bottom row plots of each column
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Avg Cost (in $)", fontsize=FONT_AXISLABEL)

        # NEW: consistent scientific formatting for cost axis
        # Show the "1e-3" offset only on the bottom row (where we also show the x-label).
        show_offset = (idx >= (nrows - 1) * ncols)
        format_cost_axis(ax, show_offset_text=show_offset)

    for j in range(used, cap):
        fig.delaxes(axes[j])

    avg_local = float(np.mean(local_aucs)) if local_aucs else float("nan")
    avg_global = float(np.mean(global_aucs)) if global_aucs else float("nan")
    fig.suptitle(
        f"Local Test Performance of Federated vs Client-Local (No-FL) with MLP\n"
        f"Mean AUC: Client-Local (No-FL)={avg_local:.2f}, Federated={avg_global:.2f}",
        fontsize=FONT_SUPTITLE, y=1
    )
    plt.tight_layout(rect=(0, 0, 1, 1))  # remove unnecessary whitespaces
    plt.savefig(os.path.join(result_out_folder, "local_vs_global_local_test_grid_mlp.pdf"), bbox_inches="tight")
    plt.close()

########################################################
## Plot: Client-Local (No-FL) vs Global on LOCAL TEST splits (Grid, K-Means)
########################################################
if kmeans_results is None:
    print("Skip K-Means local vs global on local test grid: results not loaded.")
else:
    per_client = kmeans_results["evaluations"]["per_client_local_test"]
    client_ids = sorted(per_client.keys(), key=lambda x: int(x))

    nrows, ncols = 4, 3
    cap = nrows * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8), sharex=False, sharey=False)
    axes = axes.flatten()

    local_aucs = []
    global_aucs = []
    used = 0
    for idx, cid in enumerate(client_ids[:cap]):
        ax = axes[idx]
        used += 1
        entry = per_client[cid]

        loc = entry["local_only"]
        glob = entry["global"]
        local_aucs.append(loc["auc"])
        global_aucs.append(glob["auc"])

        ax.plot([p[0] for p in loc["curve"]],
                [p[1] for p in loc["curve"]],
                marker="o", markersize=3.5, markevery=5, linestyle="-", linewidth=2.2,
                color=result_colors.get("client-local", "tab:brown"),
                label=f"Client-Local ({loc['auc']:.2f})")
        ax.plot([p[0] for p in glob["curve"]],
                [p[1] for p in glob["curve"]],
                marker="o", markersize=3.5, markevery=5, linestyle="-", linewidth=2.2,
                color=result_colors.get("global", "tab:red"),
                label=f"Federated ({glob['auc']:.2f})")
        ax.set_title(f"Client {int(cid) + 1}", fontsize=FONT_TICK)
        ax.grid(True, linestyle="-", alpha=0.4)
        ax.legend(fontsize=max(FONT_LEGEND - 2, 8)+2)
        ax.set_ylim(bottom=0.4)

        if (idx % ncols) == 0:
            ax.set_ylabel("Accuracy", fontsize=FONT_AXISLABEL)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Avg Cost (in $)", fontsize=FONT_AXISLABEL)

        # NEW: consistent scientific formatting for cost axis
        show_offset = (idx >= (nrows - 1) * ncols)
        format_cost_axis(ax, show_offset_text=show_offset)

    for j in range(used, cap):
        fig.delaxes(axes[j])

    avg_local = float(np.mean(local_aucs)) if local_aucs else float("nan")
    avg_global = float(np.mean(global_aucs)) if global_aucs else float("nan")
    fig.suptitle(
        f"Local Test Performance of Federated vs Client-Local (No-FL) with K-Means\n"
        f"Mean AUC: Client-Local (No-FL)={avg_local:.2f}, Federated={avg_global:.2f}",
        fontsize=FONT_SUPTITLE, y=1
    )
    plt.tight_layout(rect=(0, 0, 1, 1))  # remove unnecessary whitespaces
    plt.savefig(os.path.join(result_out_folder, "local_vs_global_local_test_grid_kmeans.pdf"), bbox_inches="tight")
    plt.close()


########################################################
## Plot: Representative Clients (4, 6, 8) on LOCAL TEST (MLP top, K-Means bottom)
########################################################
if (mlp_results is None) and (kmeans_results is None):
    print("Skip selected client plot: no results loaded.")
else:
    selected_display = (4, 6, 8)
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 6), sharex=False, sharey=False)
    axes = axes.flatten()

    # Top row: MLP
    if mlp_results is None:
        for col, display_id in enumerate(selected_display):
            ax = axes[col]
            ax.set_title(f"MLP Client {display_id}", fontsize=FONT_TICK)
            ax.text(0.5, 0.5, "Missing MLP results", ha="center", va="center", fontsize=FONT_TICK)
            ax.set_axis_off()
    else:
        per_client = mlp_results["evaluations"]["per_client_local_test"]
        for col, display_id in enumerate(selected_display):
            ax = axes[col]
            cid = display_id - 1
            key = str(cid)
            if key not in per_client:
                ax.set_title(f"MLP Client {display_id}", fontsize=FONT_TICK)
                ax.text(0.5, 0.5, "Missing client", ha="center", va="center", fontsize=FONT_TICK)
                ax.set_axis_off()
                continue
            entry = per_client[key]
            loc = entry["local_only"]
            glob = entry["global"]
            ax.plot([p[0] for p in loc["curve"]],
                    [p[1] for p in loc["curve"]],
                    marker="o", markersize=6, markevery=5, linestyle="-", linewidth=3,
                    color=result_colors.get("client-local", "tab:brown"),
                    label=f"Client-Local ({loc['auc']:.2f})")
            ax.plot([p[0] for p in glob["curve"]],
                    [p[1] for p in glob["curve"]],
                    marker="o", markersize=6, markevery=5, linestyle="-", linewidth=3,
                    color=result_colors.get("global", "tab:red"),
                    label=f"Federated ({glob['auc']:.2f})")
            ax.set_ylim(bottom=0.45)
            ax.set_title(f"MLP Client {display_id}", fontsize=FONT_TICK+2.5)
            ax.tick_params(axis="both", labelsize=FONT_TICK+1.5)
            ax.grid(True, linestyle="-", alpha=0.4)
            ax.legend(fontsize=max(FONT_LEGEND - 2, 8)+4, handletextpad=0.10, borderaxespad=0.10, columnspacing=0.15)
            if col == 0:
                ax.set_ylabel("Accuracy", fontsize=FONT_AXISLABEL+2)

            # NEW: consistent scientific formatting for cost axis
            # Top row has no x-label; hide the offset text to reduce clutter.
            format_cost_axis(ax, show_offset_text=False)
            # Xlabel only on bottom row, done below

    # Bottom row: K-Means
    if kmeans_results is None:
        for col, display_id in enumerate(selected_display):
            ax = axes[col + 3]
            ax.set_title(f"K-Means Client {display_id}", fontsize=FONT_TICK)
            ax.text(0.5, 0.5, "Missing K-Means results", ha="center", va="center", fontsize=FONT_TICK)
            ax.set_axis_off()
    else:
        per_client = kmeans_results["evaluations"]["per_client_local_test"]
        for col, display_id in enumerate(selected_display):
            ax = axes[col + 3]
            cid = display_id - 1
            key = str(cid)
            if key not in per_client:
                ax.set_title(f"K-Means Client {display_id}", fontsize=FONT_TICK)
                ax.text(0.5, 0.5, "Missing client", ha="center", va="center", fontsize=FONT_TICK)
                ax.set_axis_off()
                continue
            entry = per_client[key]
            loc = entry["local_only"]
            glob = entry["global"]
            ax.plot([p[0] for p in loc["curve"]],
                    [p[1] for p in loc["curve"]],
                    marker="o", markersize= 6, markevery=5, linestyle="-", linewidth=3,
                    color=result_colors.get("client-local", "tab:brown"),
                    label=f"Client-Local ({loc['auc']:.2f})")
            ax.plot([p[0] for p in glob["curve"]],
                    [p[1] for p in glob["curve"]],
                    marker="o", markersize=6, markevery=5, linestyle="-", linewidth=3,
                    color=result_colors.get("global", "tab:red"),
                    label=f"Federated ({glob['auc']:.2f})")
            ax.set_ylim(bottom=0.45)
            ax.set_title(f"K-Means Client {display_id}", fontsize=FONT_TICK+2.5)
            ax.tick_params(axis="both", labelsize=FONT_TICK+1.5)
            ax.grid(True, linestyle="-", alpha=0.4)
            ax.legend(fontsize=max(FONT_LEGEND - 2, 8)+4, handletextpad=0.10, borderaxespad=0.10, columnspacing=0.15)
            if col == 0:
                ax.set_ylabel("Accuracy", fontsize=FONT_AXISLABEL+3)
            ax.set_xlabel("Avg Cost (in $)", fontsize=FONT_AXISLABEL+3)

            # NEW: consistent scientific formatting for cost axis
            # Bottom row has x-label; show offset text.
            format_cost_axis(ax, show_offset_text=True)

    fig.suptitle(
        f"Representative Clients {tuple(selected_display)}: Local Test Performance of Federated vs Client-Local", fontsize=FONT_SUPTITLE+2.5, y=0.97
    )
    plt.tight_layout(rect=(0, 0, 1, 1)) # remove unnecessary whitespaces
    plt.savefig(os.path.join(result_out_folder, "local_vs_global_local_test_selected_clients_mlp_kmeans.pdf"), bbox_inches="tight")
    plt.close()


########################################################
## Plot: Bubble chart of per-client model heterogeneity
########################################################
if (mlp_results is not None) and ("per_client_model_active_samples" in mlp_results):
    active_samples = mlp_results["per_client_model_active_samples"]
    model_names = mlp_results.get("model_names", [])
    client_ids = sorted(active_samples.keys(), key=lambda x: int(x))
    n_clients = len(client_ids)
    n_models = len(model_names) if model_names else (len(active_samples[client_ids[0]]) if client_ids else 0)

    if n_models > 0 and n_clients > 0:
        # Build matrix: rows = clients, cols = models
        matrix = np.zeros((n_clients, n_models))
        for i, cid in enumerate(client_ids):
            counts = active_samples[cid]
            for j in range(min(len(counts), n_models)):
                matrix[i, j] = counts[j]

        # Normalize within each client (row)
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        norm_matrix = matrix / row_sums

        fig, ax = plt.subplots(figsize=(max(6, n_models * 0.6), max(4, n_clients * 0.5)))
        max_bubble = 800
        for i in range(n_clients):
            for j in range(n_models):
                size = norm_matrix[i, j] * max_bubble
                if size > 0:
                    ax.scatter(j, i, s=size, color="tab:blue", alpha=0.6, edgecolors="black", linewidths=0.5)

        ax.set_xticks(range(n_models))
        short_names = [m.split("/")[-1][:20] for m in model_names] if model_names else [str(j) for j in range(n_models)]
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=max(FONT_TICK - 2, 6))
        ax.set_yticks(range(n_clients))
        ax.set_yticklabels([f"Client {int(cid)+1}" for cid in client_ids], fontsize=FONT_TICK)
        ax.set_xlabel("Model", fontsize=FONT_AXISLABEL)
        ax.set_ylabel("Client", fontsize=FONT_AXISLABEL)
        ax.set_title("Per-Client Model Coverage (Bubble Size = Proportion)", fontsize=FONT_TITLE)
        ax.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(result_out_folder, "client_model_activesamples_bubble.pdf"), bbox_inches="tight")
        plt.close()

print("Plotting complete. Results saved to:", result_out_folder)
