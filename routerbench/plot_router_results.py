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



# plt.rcParams.update({
#     "pdf.fonttype": 42,   # embed TrueType as Type 42 (not Type 3)
#     "ps.fonttype": 42,
#     "font.family": "serif",
#     "font.serif": ["STIX Two Text"],  # or "Times New Roman", "Georgia", etc.
# })
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
BASE_DIR = os.path.dirname(__file__)
print(f"base dir: {BASE_DIR}")
MLP_RESULTS_PATH = os.path.join(BASE_DIR, "mlp_out", "mlp_experiment_results.json")
KMEANS_RESULTS_PATH = os.path.join(BASE_DIR, "kmeans_out", "kmeans_experiment_results.json")
MLP_TSNE_PATH = os.path.join(BASE_DIR, "mlp_out", "tsne.json")
KMEANS_TSNE_PATH = os.path.join(BASE_DIR, "kmeans_out", "tsne.json")

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

mlp_tsne = None
if os.path.exists(MLP_TSNE_PATH):
    with open(MLP_TSNE_PATH, "r", encoding="utf-8") as f:
        mlp_tsne = json.load(f)
else:
    print(f"Missing t-SNE stats: {MLP_TSNE_PATH}")

kmeans_tsne = None

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
    
    try:
        color_mlp_global = COLORS.get("mlp_global", "tab:blue")
        color_mlp_centralized = COLORS.get("mlp_centralized", "tab:orange")
        color_km_global = COLORS.get("kmeans_global", "tab:green")
        color_km_centralized = COLORS.get("kmeans_centralized", "tab:red")
    except Exception:
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
        # frameon=True, fancybox=True, facecolor=(0.7, 0.7, 0.7, 0.35),
        columnspacing=0.08, handletextpad=0.13, borderaxespad=0.14
    )
    legend.get_frame().set_edgecolor("none")

    # ========== Bottom: K-Means ==========
    ax = axs[1]
    kmeans_global_color = result_colors.get("kmeans", "tab:orange")
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
        # frameon=True, fancybox=True, facecolor=(0.7, 0.7, 0.7, 0.35),
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
                marker="o", markersize=3.5, markevery=5, linestyle="-", linewidth=2.2,  # smaller dots
                color=result_colors.get("client-local", "tab:brown"),
                label=f"Client-Local ({loc['auc']:.2f})")
        ax.plot([p[0] for p in glob["curve"]],
                [p[1] for p in glob["curve"]],
                marker="o", markersize=3.5, markevery=5, linestyle="-", linewidth=2.2,  # smaller dots
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
                marker="o", markersize=3.5, markevery=5, linestyle="-", linewidth=2.2,  # smaller dots
                color=result_colors.get("client-local", "tab:brown"),
                label=f"Client-Local ({loc['auc']:.2f})")
        ax.plot([p[0] for p in glob["curve"]],
                [p[1] for p in glob["curve"]],
                marker="o", markersize=3.5, markevery=5, linestyle="-", linewidth=2.2,  # smaller dots
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
## Plot: t-SNE client grid (MLP)
########################################################
if mlp_tsne is None:
    print("Skip t-SNE grid: stats not loaded.")
else:
    X_tsne = np.asarray(mlp_tsne.get("X_tsne", []), dtype=float)
    tsne_indices = np.asarray(mlp_tsne.get("tsne_indices", []), dtype=int)
    task_keys = np.asarray(mlp_tsne.get("task_keys", []), dtype=str)
    unique_keys = mlp_tsne.get("unique_keys", [])
    clients = mlp_tsne.get("clients", [])

    if X_tsne.size == 0 or tsne_indices.size == 0:
        print("Skip t-SNE grid: empty stats.")
    else:
        num_colors = len(unique_keys)
        base = "tab20" if num_colors <= 20 else "gist_ncar"
        cmap = plt.get_cmap(base, num_colors)
        key_to_color = {k: i for i, k in enumerate(unique_keys)}
        idx_map = {int(idx): j for j, idx in enumerate(tsne_indices.tolist())}
        rng = np.random.default_rng(42)

        max_pts_per_client = 1000
        fig, axes = plt.subplots(4, 3, figsize=(12, 12), sharex=True, sharey=True)
        axes = axes.flatten()

        ax0 = axes[0]
        for k in unique_keys:
            kid = key_to_color[k]
            mask = (task_keys == k)
            if not np.any(mask):
                continue
            # Subsample 1/5th
            X_k = X_tsne[mask]
            n_pts = X_k.shape[0]
            if n_pts > 0:
                take = max(1, int(np.ceil(n_pts / 5)))
                sel = rng.choice(n_pts, size=take, replace=False)
                X_plot = X_k[sel]
                ax0.scatter(
                    X_plot[:, 0],
                    X_plot[:, 1],
                    s=8,
                    color=cmap(kid),
                    alpha=0.7,
                    label=k,
                )
        ax0.set_title("All clients (used for t-SNE)", fontsize=FONT_TICK)
        ax0.set_xticks([]); ax0.set_yticks([])

        for subplot_idx, spec in enumerate(clients, start=1):
            if subplot_idx >= len(axes):
                break
            ax = axes[subplot_idx]
            cid = spec.get("cid", subplot_idx - 1)
            client_train_idx = np.asarray(spec.get("train_idx", []), dtype=int)
            client_train_idx = np.intersect1d(tsne_indices, client_train_idx, assume_unique=False)

            if client_train_idx.size == 0:
                ax.set_title(f"Client {subplot_idx} (no TRAIN in t-SNE subset)", fontsize=FONT_TICK)
                ax.set_xticks([]); ax.set_yticks([])
                continue

            # Subsample 1/5th for each client
            n_pts = client_train_idx.size
            take = max(1, int(np.ceil(n_pts / 3)))
            client_train_idx = rng.choice(client_train_idx, size=take, replace=False)

            plot_client_pos = np.array([idx_map[int(i)] for i in client_train_idx], dtype=int)
            for k in unique_keys:
                kid = key_to_color[k]
                mask = (task_keys[plot_client_pos] == k)
                if not np.any(mask):
                    continue
                ax.scatter(
                    X_tsne[plot_client_pos[mask], 0],
                    X_tsne[plot_client_pos[mask], 1],
                    s=8,
                    color=cmap(kid),
                    alpha=0.7,
                )

            # Set title with client numbers 1-10 (not 0-based)
            ax.set_title(f"Client {subplot_idx}", fontsize=FONT_TICK)
            ax.set_xticks([]); ax.set_yticks([])

        for j in range(1 + len(clients), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(
            "t-SNE Visualization of (Subsampled) Data Points\nColors: per-task, with MMLU/Chinese Subtasks Merged",
            fontsize=FONT_SUPTITLE, y=0.93
        )
        handles, labels = ax0.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="upper center",
            ncol=8,
            fontsize=max(FONT_LEGEND, 11),
            frameon=False,
            bbox_to_anchor=(0.5, 0.9),
            columnspacing=0.5,  # squeeze space between columns
            handletextpad=0.03   # reduce space between marker and label text
        )
        plt.tight_layout(rect=(0, 0, 1, 0.90))
        plt.savefig(os.path.join(result_out_folder, "tsne_clients_grid.png"), dpi=200, bbox_inches="tight")
        plt.savefig(os.path.join(result_out_folder, "tsne_clients_grid.pdf"), bbox_inches="tight")
        plt.close()


import matplotlib.colors as mcolors

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by mixing it with white.

    color: any matplotlib color spec (name, hex, rgb/rgba tuple)
    amount: 0 -> no change, 1 -> white
    """
    r, g, b, a = mcolors.to_rgba(color)
    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount
    return (r, g, b, a)
# ########################################################
# ## Plot: t-SNE client grid (MLP) with color-by-subtask
# ########################################################
# if mlp_tsne is None:
#     print("Skip t-SNE grid: stats not loaded.")
# else:
#     import matplotlib.colors as mcolors

#     X_tsne = np.asarray(mlp_tsne.get("X_tsne", []), dtype=float)
#     tsne_indices = np.asarray(mlp_tsne.get("tsne_indices", []), dtype=int)
#     task_keys = np.asarray(mlp_tsne.get("task_keys", []), dtype=str)
#     unique_keys = mlp_tsne.get("unique_keys", [])
#     clients = mlp_tsne.get("clients", [])

#     # Group subtasks: keep all separate, but ensure ANY MMLU-labeled key becomes "MMLU"
#     mmlu_subtasks = [
#         k for k in unique_keys
#         if isinstance(k, str) and ("mmlu" in k.lower())  # catches "MMLU (all)" and "mmlu/..."
#     ]

#     chinese_subtasks = [
#         k for k in unique_keys
#         if isinstance(k, str) and ("chinese" in k.lower())
#         and (k not in mmlu_subtasks)
#     ]

#     other_tasks = [
#         k for k in unique_keys
#         if (k not in mmlu_subtasks) and (k not in chinese_subtasks)
#     ]

#     # (optional but recommended) stable ordering so colors don't reshuffle run-to-run
#     mmlu_subtasks = sorted(mmlu_subtasks, key=str)
#     chinese_subtasks = sorted(chinese_subtasks, key=str)
#     other_tasks = sorted(other_tasks, key=str)

#     from matplotlib.cm import get_cmap
#     import matplotlib.patches as mpatches

#     # --- Colormap customizations for MMLU subtasks ---
#     def red_palette(n):
#         # Use Reds colormap, pick evenly across good spread (avoid too light and too dark)
#         cmap = get_cmap('Reds')
#         # For a single subtask: just pick center
#         if n == 1:
#             samples = [0.6]
#         else:
#             # Avoid the very lightest and darkest
#             samples = np.linspace(0.35, 0.85, n)
#         colors = [cmap(x) for x in samples]
#         # always return RGBA
#         return [tuple(list(mcolors.to_rgb(c)) + [1.0]) for c in colors]
#     import numpy as np
#     import matplotlib.colors as mcolors
#     def red_palette_ultra_separated(n, seed=42):
#         """
#         Strongly separated 'red family' palette.
#         Uses multiple nearby hue bands (red -> orange-red -> crimson -> magenta-red),
#         plus interleaved brightness + alternating saturation.
#         Everything still reads as "reddish", but much more distinct.
#         """
#         if n <= 0:
#             return []
#         rng = np.random.default_rng(seed)

#         # Hue bands (in HSV): 0 is red. Values near 1 wrap back to red.
#         # These bands stay in the red family but cover orange-red to magenta-red.
#         hue_bands = [
#             (0.00, 0.04),   # red
#             (0.96, 1.00),   # red (wrapped)
#             (0.04, 0.09),   # orange-red
#             (0.90, 0.96),   # magenta-red
#         ]

#         # Brightness spread: avoid too-dark maroons and too-light pinks
#         v = np.linspace(0.45, 0.95, n)

#         # Interleave extremes for max adjacent separation (legend and assignment)
#         order = []
#         i, j = 0, n - 1
#         while i <= j:
#             order.append(i)
#             if i != j:
#                 order.append(j)
#             i += 1
#             j -= 1
#         v = v[np.array(order)]

#         # Alternate saturation strongly
#         s = np.empty(n)
#         s[0::2] = np.linspace(0.98, 0.80, (n + 1) // 2)
#         s[1::2] = np.linspace(0.90, 0.65, n // 2)

#         # Assign hues by cycling bands, with jitter inside each band
#         h = np.empty(n)
#         for i in range(n):
#             a, b = hue_bands[i % len(hue_bands)]
#             h[i] = rng.uniform(a, b) % 1.0

#         rgb = mcolors.hsv_to_rgb(np.stack([h, s, v], axis=1))
#         return [(*rgb[i], 1.0) for i in range(n)]
#     def red_palette_separated(n, seed=42):
#         """
#         Red family with high separation.
#         Uses HSV with small hue jitter around red + spread saturation/value.
#         """
#         if n <= 0:
#             return []
#         rng = np.random.default_rng(seed)

#         # Value (brightness) spread, avoid too-dark and too-light
#         v = np.linspace(0.40, 0.95, n)

#         # Interleave extremes to maximize separation in legend order
#         order = []
#         i, j = 0, n - 1
#         while i <= j:
#             order.append(i)
#             if i != j:
#                 order.append(j)
#             i += 1
#             j -= 1
#         v = v[np.array(order)]

#         # Saturation alternation (neighbors differ more)
#         s = np.empty(n)
#         s[0::2] = np.linspace(0.98, 0.78, (n + 1) // 2)
#         s[1::2] = np.linspace(0.92, 0.65, n // 2)

#         # Hue jitter around red (0.0) but keep it red-looking
#         # Slightly wider jitter than before for more separation
#         h = (rng.uniform(-0.05, 0.05, size=n)) % 1.0

#         rgb = mcolors.hsv_to_rgb(np.stack([h, s, v], axis=1))
#         return [(*rgb[i], 1.0) for i in range(n)]

#     def qualitative_palette(n):
#         import matplotlib as mpl
#         tab20 = mpl.colormaps["tab20"].colors
#         set3  = mpl.colormaps["Set3"].colors
#         full_palette = list(tab20) + list(set3)
#         if n > len(full_palette):
#             extra = n - len(full_palette)
#             hsv_colors = [mcolors.hsv_to_rgb((i/float(extra), 0.7, 0.9)) for i in range(extra)]
#             palette = full_palette + hsv_colors
#         else:
#             palette = full_palette
#         idxs = np.linspace(0, len(palette)-1, n).astype(int)
#         palette = np.array(palette)[idxs]
#         return list(palette)

#     def qualitative_no_grey(n, seed=123):
#         """
#         Qualitative palette that avoids grey/white/pale colors.
#         Filters by saturation and brightness.
#         """
#         if n <= 0:
#             return []
#         rng = np.random.default_rng(seed)

#         colors = []
#         # We'll oversample candidates then filter
#         # (stable and fast for typical n <= ~50)
#         attempts = 0
#         while len(colors) < n and attempts < 20000:
#             attempts += 1

#             # Evenly spread hue, with some jitter
#             h = (len(colors) / max(n, 1) + rng.uniform(-0.03, 0.03)) % 1.0

#             # Force colorfulness and avoid pale/grey
#             s = rng.uniform(0.70, 1.00)   # high saturation
#             v = rng.uniform(0.55, 0.90)   # avoid near-white

#             rgb = mcolors.hsv_to_rgb((h, s, v))
#             rgba = (*rgb, 1.0)

#             colors.append(rgba)

#         return colors[:n]

#     # Assign colors according to subtask type, with MMLU = reds
#     mmlu_subtasks = sorted(mmlu_subtasks, key=str)
#     rng = np.random.default_rng(123)
#     rng.shuffle(mmlu_subtasks)  # stable shuffle
#     mmlu_colors = red_palette_ultra_separated(len(mmlu_subtasks), seed=42)
#     chinese_colors = qualitative_no_grey(len(chinese_subtasks))
#     other_colors = qualitative_no_grey(len(other_tasks))

#     all_subtasks = mmlu_subtasks + chinese_subtasks + other_tasks
#     subtask_colors = mmlu_colors + chinese_colors + other_colors
#     key_to_color = {k: subtask_colors[i] for i, k in enumerate(all_subtasks)}

#     # Build custom legend: MMLU subtasks, Chinese subtasks, other tasks — show ALL subtasks
#     handles_for_legend = []
#     for i, k in enumerate(mmlu_subtasks):
#         handles_for_legend.append(mpatches.Patch(color=key_to_color[k], label=k))
#     for i, k in enumerate(chinese_subtasks):
#         handles_for_legend.append(mpatches.Patch(color=key_to_color[k], label=k))
#     for i, k in enumerate(other_tasks):
#         handles_for_legend.append(mpatches.Patch(color=key_to_color[k], label=k))

#     idx_map = {int(idx): j for j, idx in enumerate(tsne_indices.tolist())}
#     rng = np.random.default_rng(42)

#     max_pts_per_client = 1000
#     fig, axes = plt.subplots(4, 3, figsize=(12, 12), sharex=True, sharey=True)
#     axes = axes.flatten()

#     #----- Plot all points panel -----
#     ax0 = axes[0]
#     for k in unique_keys:
#         mask = (task_keys == k)
#         if not np.any(mask):
#             continue
#         # Subsample 1/5th
#         X_k = X_tsne[mask]
#         n_pts = X_k.shape[0]
#         if n_pts > 0:
#             take = max(1, int(np.ceil(n_pts / 5)))
#             sel = rng.choice(n_pts, size=take, replace=False)
#             X_plot = X_k[sel]
#             color_this = key_to_color.get(k, (0.5,0.5,0.5,1))
#             ax0.scatter(
#                 X_plot[:, 0],
#                 X_plot[:, 1],
#                 s=8,
#                 color=color_this,
#                 alpha=0.8,
#                 label=k,
#                 linewidths=0,
#             )

#     ax0.set_title("All clients (used for t-SNE)", fontsize=FONT_TICK)
#     ax0.set_xticks([]); ax0.set_yticks([])

#     #----- Plot for each client -----
#     for subplot_idx, spec in enumerate(clients, start=1):
#         if subplot_idx >= len(axes):
#             break
#         ax = axes[subplot_idx]
#         cid = spec.get("cid", subplot_idx - 1)
#         client_train_idx = np.asarray(spec.get("train_idx", []), dtype=int)
#         client_train_idx = np.intersect1d(tsne_indices, client_train_idx, assume_unique=False)

#         if client_train_idx.size == 0:
#             ax.set_title(f"Client {subplot_idx} (no TRAIN in t-SNE subset)", fontsize=FONT_TICK)
#             ax.set_xticks([]); ax.set_yticks([])
#             continue

#         # Subsample 1/3rd for each client:
#         n_pts = client_train_idx.size
#         take = max(1, int(np.ceil(n_pts / 3)))
#         client_train_idx = rng.choice(client_train_idx, size=take, replace=False)

#         plot_client_pos = np.array([idx_map[int(i)] for i in client_train_idx], dtype=int)
#         for k in unique_keys:
#             mask = (task_keys[plot_client_pos] == k)
#             if not np.any(mask):
#                 continue
#             color_this = key_to_color.get(k, (0.5,0.5,0.5,1))
#             ax.scatter(
#                 X_tsne[plot_client_pos[mask], 0],
#                 X_tsne[plot_client_pos[mask], 1],
#                 s=8,
#                 color=color_this,
#                 alpha=0.8,
#                 linewidths=0,
#             )

#         # Set title with client numbers 1-10 (not 0-based)
#         ax.set_title(f"Client {subplot_idx}", fontsize=FONT_TICK)
#         ax.set_xticks([]); ax.set_yticks([])

#     for j in range(1 + len(clients), len(axes)):
#         fig.delaxes(axes[j])

#     fig.suptitle(
#         "t-SNE Visualization of (Subsampled) Data Points\nColors: MMLU=reds, others distinct",
#         fontsize=FONT_SUPTITLE, y=0.93
#     )
#     fig.legend(
#         handles_for_legend,
#         [h.get_label() for h in handles_for_legend],
#         loc="upper center",
#         ncol=5 if len(handles_for_legend) <= 10 else 8,
#         fontsize=max(FONT_LEGEND, 11),
#         frameon=False,
#         bbox_to_anchor=(0.5, 0.9),
#         columnspacing=0.5,
#         handletextpad=0.07
#     )
#     plt.tight_layout(rect=(0, 0, 1, 0.90))
#     plt.savefig(os.path.join(result_out_folder, "tsne_clients_grid.png"), dpi=200, bbox_inches="tight")
#     plt.savefig(os.path.join(result_out_folder, "tsne_clients_grid.pdf"), bbox_inches="tight")
#     plt.close()

########################################################
## Plot: Model expansion (MLP and K-Means) in one figure
########################################################
MODEL_EXPANSION_DIR = os.path.join(BASE_DIR, "model_expansion_out")
MLP_EXPANSION_PATH = os.path.join(MODEL_EXPANSION_DIR, "mlp_model_expansion_experiment_results.json")
KMEANS_EXPANSION_PATH = os.path.join(MODEL_EXPANSION_DIR, "kmeans_model_expansion_experiment_results.json")
model_expansion_out_folder = os.path.join(result_out_folder, "./model_expansion")
if not os.path.exists(model_expansion_out_folder):
    os.makedirs(model_expansion_out_folder)

def lighten_color2(color, amount=0.6):
    """Lighten the given color by multiplying (1-luminosity) by the given amount."""
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(
        c[0],
        1 - amount * (1 - c[1]),  # lighter by pulling toward 1.0
        c[2]
    )

mlp_expansion = None
if os.path.exists(MLP_EXPANSION_PATH):
    with open(MLP_EXPANSION_PATH, "r", encoding="utf-8") as f:
        mlp_expansion = json.load(f)
else:
    print(f"Missing model expansion results: {MLP_EXPANSION_PATH}")

kmeans_expansion = None
if os.path.exists(KMEANS_EXPANSION_PATH):
    with open(KMEANS_EXPANSION_PATH, "r", encoding="utf-8") as f:
        kmeans_expansion = json.load(f)
else:
    print(f"Missing model expansion results: {KMEANS_EXPANSION_PATH}")

if (mlp_expansion is not None) and (kmeans_expansion is not None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    def _plot_base_vs_adapted(ax, evals, title, color, marker, base_linestyle, base_marker_size, adapt_linestyle, adapt_marker_size):
        base = evals["base_holdout"]
        adapted = evals["adapted"]
        lighter_color = lighten_color2(color, amount=0.6)

        # Before New Clients Join (lighter color, base_linestyle)
        ax.plot(
            [p[0] for p in base["curve"]],
            [p[1] for p in base["curve"]],
            marker=marker,
            markersize=base_marker_size,
            linewidth=2.2,
            linestyle=base_linestyle,
            color=lighter_color,
            label=f"Before New Models Join ({base['auc']:.3f})",
        )
        # Expanded (full/original color, adapt_linestyle)
        ax.plot(
            [p[0] for p in adapted["curve"]],
            [p[1] for p in adapted["curve"]],
            marker=marker,
            markersize=adapt_marker_size,
            linewidth=2.2,
            linestyle=adapt_linestyle,
            color=color,
            label=f"Expanded ({adapted['auc']:.3f})",
        )
        ax.set_title(title, fontsize=FONT_TITLE+2)
        ax.set_xlabel("Avg Cost (in $)", fontsize=FONT_AXISLABEL+2)
        ax.tick_params(axis='both', labelsize=FONT_TICK+1)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=FONT_LEGEND+2)

        # NEW: consistent scientific formatting for cost axis
        format_cost_axis(ax, show_offset_text=True)

    mlp_evals = mlp_expansion["evaluations"]["global_test"]
    km_evals = kmeans_expansion["evaluations"]["global_test"]

    _plot_base_vs_adapted(
        axes[0],
        mlp_evals,
        "MLP Model Expansion",
        result_colors.get("mlp", "tab:blue"),
        marker="o",
        base_linestyle="solid",  # solid for base MLP
        base_marker_size=3,
        adapt_linestyle="solid", # solid for expanded MLP
        adapt_marker_size=3,
    )
    axes[0].set_ylabel("Accuracy", fontsize=FONT_AXISLABEL+2)

    _plot_base_vs_adapted(
        axes[1],
        km_evals,
        "K-Means Model Expansion",
        result_colors.get("kmeans", "tab:orange"),
        marker="s",
        base_linestyle="dashed", # dashed for base K-Means
        base_marker_size=3,
        adapt_linestyle="dashed",# dashed for expanded K-Means
        adapt_marker_size=3,
    )

    fig.suptitle("Base vs Expanded Router (Global TEST)", fontsize=FONT_SUPTITLE+2)
    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig(os.path.join(model_expansion_out_folder, "model_expansion_base_vs_adapted_mlp_kmeans.pdf"))
    plt.close()
else:
    print("Skip model expansion plot: at least one result not loaded.")

# Combine the base vs expanded curves for MLP and K-Means in a single figure (no subplots)
if (mlp_expansion is not None) and (kmeans_expansion is not None):
    plt.figure(figsize=(6, 3.8))
    mlp_evals = mlp_expansion["evaluations"]["global_test"]
    km_evals = kmeans_expansion["evaluations"]["global_test"]

    # Prepare colors
    mlp_color = result_colors.get("mlp", "tab:blue")
    kmeans_color = result_colors.get("kmeans", "tab:orange")
    mlp_color_light = lighten_color2(mlp_color, amount=0.6)
    kmeans_color_light = lighten_color2(kmeans_color, amount=0.6)

    # MLP base - solid, lighter, smaller dots
    plt.plot(
        [p[0] for p in mlp_evals["base_holdout"]["curve"]],
        [p[1] for p in mlp_evals["base_holdout"]["curve"]],
        marker="o",
        markersize=3,
        linewidth=2.2,
        linestyle="solid",
        color=mlp_color_light,
        label=f"MLP Before ({mlp_evals['base_holdout']['auc']:.3f})"
    )

    # MLP expanded - solid, main color, smaller dots
    plt.plot(
        [p[0] for p in mlp_evals["adapted"]["curve"]],
        [p[1] for p in mlp_evals["adapted"]["curve"]],
        marker="o",
        markersize=3,
        linewidth=2.2,
        linestyle="solid",
        color=mlp_color,
        label=f"MLP After ({mlp_evals['adapted']['auc']:.3f})"
    )

    # KMeans base - dashed, lighter, smaller squares
    plt.plot(
        [p[0] for p in km_evals["base_holdout"]["curve"]],
        [p[1] for p in km_evals["base_holdout"]["curve"]],
        marker="s",
        markersize=3,
        linewidth=2.2,
        linestyle="dashed",
        color=kmeans_color_light,
        label=f"K-Means Before ({km_evals['base_holdout']['auc']:.3f})"
    )

    # KMeans expanded - dashed, full color, smaller squares
    plt.plot(
        [p[0] for p in km_evals["adapted"]["curve"]],
        [p[1] for p in km_evals["adapted"]["curve"]],
        marker="s",
        markersize=3,
        linewidth=2.2,
        linestyle="dashed",
        color=kmeans_color,
        label=f"K-Means After ({km_evals['adapted']['auc']:.3f})"
    )

    plt.ylim(0.52, 0.81)

    plt.title("Global Test Performances Before and After New Models Join", fontsize=FONT_TITLE+2)
    plt.xlabel("Avg Cost (in $)", fontsize=FONT_AXISLABEL+2)
    plt.ylabel("Accuracy", fontsize=FONT_AXISLABEL+2)
    plt.tick_params(axis='both', labelsize=FONT_TICK+1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=FONT_LEGEND+2)

    # NEW: consistent scientific formatting for cost axis
    ax = plt.gca()
    format_cost_axis(ax, show_offset_text=True)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(os.path.join(model_expansion_out_folder, "model_expansion_base_vs_adapted_combined.pdf"), bbox_inches="tight")
    plt.close()


########################################################
## Plot: Client expansion (MLP and K-Means) in one figure
########################################################
CLIENT_EXPANSION_DIR = os.path.join(BASE_DIR, "client_expansion_out")
MLP_CLIENT_EXPANSION_PATH = os.path.join(CLIENT_EXPANSION_DIR, "mlp_client_expansion_results.json")
KMEANS_CLIENT_EXPANSION_PATH = os.path.join(CLIENT_EXPANSION_DIR, "kmeans_client_expansion_results.json")
client_expansion_out_folder = os.path.join(result_out_folder, "./client_expansion")
if not os.path.exists(client_expansion_out_folder):
    os.makedirs(client_expansion_out_folder)

mlp_client_expansion = None
if os.path.exists(MLP_CLIENT_EXPANSION_PATH):
    with open(MLP_CLIENT_EXPANSION_PATH, "r", encoding="utf-8") as f:
        mlp_client_expansion = json.load(f)
else:
    print(f"Missing client expansion results: {MLP_CLIENT_EXPANSION_PATH}")

kmeans_client_expansion = None
if os.path.exists(KMEANS_CLIENT_EXPANSION_PATH):
    with open(KMEANS_CLIENT_EXPANSION_PATH, "r", encoding="utf-8") as f:
        kmeans_client_expansion = json.load(f)
else:
    print(f"Missing client expansion results: {KMEANS_CLIENT_EXPANSION_PATH}")

if (mlp_client_expansion is not None) and (kmeans_client_expansion is not None):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    def _plot_client_expansion(ax, evals, title, color, marker, base_linestyle, adapt_linestyle, base_key, adapt_key):
        base = evals[base_key]
        adapted = evals[adapt_key]
        lighter_color = lighten_color(color, amount=0.6)

        ax.plot(
            [p[0] for p in base["curve"]],
            [p[1] for p in base["curve"]],
            marker=marker,
            markevery=5,
            markersize=3,
            linewidth=2.2,
            linestyle=base_linestyle,
            color=lighter_color,
            label=f"Before New Clients Join ({base['auc']:.2f})",
        )
        ax.plot(
            [p[0] for p in adapted["curve"]],
            [p[1] for p in adapted["curve"]],
            marker=marker,
            markevery=5,
            markersize=3,
            linewidth=2.2,
            linestyle=adapt_linestyle,
            color=color,
            label=f"After New Clients Join ({adapted['auc']:.2f})",
        )
        ax.set_ylim(bottom=0.4)
        ax.set_title(title, fontsize=FONT_TITLE+1)
        ax.set_xlabel("Avg Cost (in $)", fontsize=FONT_AXISLABEL+1)
        ax.tick_params(axis='both', labelsize=FONT_TICK)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=FONT_LEGEND+2)

        # NEW: consistent scientific formatting for cost axis
        format_cost_axis(ax, show_offset_text=True)

    mlp_client_evals = mlp_client_expansion["global_test"]
    kmeans_client_evals = kmeans_client_expansion["global_test"]

    _plot_client_expansion(
        axes[0],
        mlp_client_evals,
        "MLP Router",
        result_colors.get("mlp", "tab:blue"),
        marker="o",
        base_linestyle="solid",
        adapt_linestyle="solid",
        base_key="base_model",
        adapt_key="adapted_model",
    )
    axes[0].set_ylabel("Accuracy", fontsize=FONT_AXISLABEL+1)

    _plot_client_expansion(
        axes[1],
        kmeans_client_evals,
        "K-Means Router",
        result_colors.get("kmeans", "tab:orange"),
        marker="s",
        base_linestyle="dashed",
        adapt_linestyle="dashed",
        base_key="base_router",
        adapt_key="adapted_router",
    )

    fig.suptitle("Global Test Performance Before and After New Clients Join", fontsize=FONT_SUPTITLE+1)
    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig(os.path.join(client_expansion_out_folder, "client_expansion_base_vs_adapted_mlp_kmeans.pdf"), bbox_inches="tight")
    plt.close()
else:
    print("Skip client expansion plot: at least one result not loaded.")


########################################################
## Plot: High-heterogeneity personalization on LOCAL TEST splits
########################################################
HIGH_HET_DIR_MLP = os.path.join(BASE_DIR, "mlp_out_high_het")
HIGH_HET_DIR_KMEANS = os.path.join(BASE_DIR, "kmeans_out_high_het")
MLP_HIGH_HET_RESULTS_PATH = os.path.join(HIGH_HET_DIR_MLP, "mlp_experiment_results.json")
KMEANS_HIGH_HET_RESULTS_PATH = os.path.join(HIGH_HET_DIR_KMEANS, "kmeans_experiment_results.json")
high_het_out_folder = os.path.join(result_out_folder, "./high_het_results")
if not os.path.exists(high_het_out_folder):
    os.makedirs(high_het_out_folder)

mlp_high_het_results = None
if os.path.exists(MLP_HIGH_HET_RESULTS_PATH):
    with open(MLP_HIGH_HET_RESULTS_PATH, "r", encoding="utf-8") as f:
        mlp_high_het_results = json.load(f)
else:
    print(f"Missing high-het MLP results: {MLP_HIGH_HET_RESULTS_PATH}")

kmeans_high_het_results = None
if os.path.exists(KMEANS_HIGH_HET_RESULTS_PATH):
    with open(KMEANS_HIGH_HET_RESULTS_PATH, "r", encoding="utf-8") as f:
        kmeans_high_het_results = json.load(f)
else:
    print(f"Missing high-het K-Means results: {KMEANS_HIGH_HET_RESULTS_PATH}")

########################################################
## Plot: Client-Local (No-FL) vs Global vs Adaptive on LOCAL TEST splits (Grid, MLP)
########################################################
if mlp_high_het_results is None:
    print("Skip high-het MLP local/global/adaptive grid: results not loaded.")
else:
    per_client = mlp_high_het_results["evaluations"]["per_client_local_test"]
    client_ids = sorted(per_client.keys(), key=lambda x: int(x))

    nrows, ncols = 4, 3
    cap = nrows * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8), sharex=False, sharey=False)
    axes = axes.flatten()

    local_aucs = []
    global_aucs = []
    adaptive_aucs = []
    used = 0
    for idx, cid in enumerate(client_ids[:cap]):
        ax = axes[idx]
        used += 1
        entry = per_client[cid]

        loc = entry["local_only"]
        glob = entry["global"]
        adapt = entry["adaptive_ensemble"]
        local_aucs.append(loc["auc"])
        global_aucs.append(glob["auc"])
        adaptive_aucs.append(adapt["auc"])

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
        ax.plot([p[0] for p in adapt["curve"]],
                [p[1] for p in adapt["curve"]],
                marker="o", markersize=3.5, markevery=5, linestyle="--", linewidth=2.2,
                color='blue',#result_colors.get("adaptive-ensemble", "tab:blue"),
                label=f"Personalized ({adapt['auc']:.2f})")
        ax.set_title(f"Client {int(cid) + 1}", fontsize=FONT_TICK)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0.46)
        legend = ax.legend(
            fontsize=max(FONT_LEGEND - 2, 8) + 0.4,
            frameon=False,
            loc="lower right",
            handletextpad=0.10,      # squeeze text closer to symbol
            borderaxespad=0.10,      # reduce distance to axes border
            columnspacing=0.15,      # closer columns (in case of multiple)
        )
        legend.get_frame().set_alpha(0.0)
        legend.get_frame().set_facecolor('none')

        if (idx % ncols) == 0:
            ax.set_ylabel("Accuracy", fontsize=FONT_AXISLABEL+1)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Avg Cost (in $)", fontsize=FONT_AXISLABEL+1)

        # NEW: consistent scientific formatting for cost axis
        show_offset = (idx >= (nrows - 1) * ncols)
        format_cost_axis(ax, show_offset_text=show_offset)

    for j in range(used, cap):
        fig.delaxes(axes[j])

    avg_local = float(np.mean(local_aucs)) if local_aucs else float("nan")
    avg_global = float(np.mean(global_aucs)) if global_aucs else float("nan")
    avg_adapt = float(np.mean(adaptive_aucs)) if adaptive_aucs else float("nan")
    fig.suptitle(
        "MLP Router: Local Test Performance under High Heterogeneity\n"
        f"Mean AUC: Client-Local={avg_local:.2f}, Federated={avg_global:.2f}, Personalized={avg_adapt:.2f}",
        fontsize=FONT_SUPTITLE+1, y=0.97
    )
    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig(
        os.path.join(high_het_out_folder, "high_het_local_global_adaptive_local_test_grid_mlp.pdf"),
        bbox_inches="tight"
    )
    plt.close()

########################################################
## Plot: Client-Local (No-FL) vs Global vs Adaptive on LOCAL TEST splits (Grid, K-Means)
########################################################
if kmeans_high_het_results is None:
    print("Skip high-het K-Means local/global/adaptive grid: results not loaded.")
else:
    per_client = kmeans_high_het_results["evaluations"]["per_client_local_test"]
    client_ids = sorted(per_client.keys(), key=lambda x: int(x))

    nrows, ncols = 4, 3
    cap = nrows * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 8), sharex=False, sharey=False)
    axes = axes.flatten()

    local_aucs = []
    global_aucs = []
    adaptive_aucs = []
    used = 0
    for idx, cid in enumerate(client_ids[:cap]):
        ax = axes[idx]
        used += 1
        entry = per_client[cid]

        loc = entry["local_only"]
        glob = entry["global"]
        adapt = entry["adaptive_ensemble"]
        local_aucs.append(loc["auc"])
        global_aucs.append(glob["auc"])
        adaptive_aucs.append(adapt["auc"])

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
        ax.plot([p[0] for p in adapt["curve"]],
                [p[1] for p in adapt["curve"]],
                marker="o", markersize=3.5, markevery=5, linestyle="--", linewidth=2.2,
                color='blue',#result_colors.get("adaptive-ensemble", "tab:blue"),
                label=f"Personalized ({adapt['auc']:.2f})")
        ax.set_title(f"Client {int(cid) + 1}", fontsize=FONT_TICK)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(bottom=0.46)
        legend = ax.legend(
            fontsize=max(FONT_LEGEND - 2, 8) + 0.4,
            frameon=False,
            loc="lower right",
            handletextpad=0.10,      # squeeze text closer to symbol
            borderaxespad=0.10,      # reduce distance to axes border
            columnspacing=0.15,      # closer columns (in case of multiple)
        )
        legend.get_frame().set_alpha(0.0)
        legend.get_frame().set_facecolor('none')

        if (idx % ncols) == 0:
            ax.set_ylabel("Accuracy", fontsize=FONT_AXISLABEL+1)
        if idx >= (nrows - 1) * ncols:
            ax.set_xlabel("Avg Cost (in $)", fontsize=FONT_AXISLABEL+1)

        # NEW: consistent scientific formatting for cost axis
        show_offset = (idx >= (nrows - 1) * ncols)
        format_cost_axis(ax, show_offset_text=show_offset)

    for j in range(used, cap):
        fig.delaxes(axes[j])

    avg_local = float(np.mean(local_aucs)) if local_aucs else float("nan")
    avg_global = float(np.mean(global_aucs)) if global_aucs else float("nan")
    avg_adapt = float(np.mean(adaptive_aucs)) if adaptive_aucs else float("nan")
    fig.suptitle(
        "K-Means Router: Local Test Performance of Client-Local (No-FL) vs Federated vs Adaptive\n"
        f"Mean AUC: Client-Local={avg_local:.2f}, Federated={avg_global:.2f}, Personalized={avg_adapt:.2f}",
        fontsize=FONT_SUPTITLE+1, y=0.97
    )
    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig(
        os.path.join(high_het_out_folder, "high_het_local_global_adaptive_local_test_grid_kmeans.pdf"),
        bbox_inches="tight"
    )
    plt.close()

########################################################
## Plot: Representative Clients (4, 6, 8) on LOCAL TEST (MLP top, K-Means bottom)
########################################################
if (mlp_high_het_results is None) and (kmeans_high_het_results is None):
    print("Skip high-het selected client plot: no results loaded.")
else:
    selected_display = (4, 6, 8)
    nrows, ncols = 2, 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 6), sharex=False, sharey=False)
    axes = axes.flatten()

    # Top row: MLP
    if mlp_high_het_results is None:
        for col, display_id in enumerate(selected_display):
            ax = axes[col]
            ax.set_title(f"MLP Client {display_id}", fontsize=FONT_TICK)
            ax.text(0.5, 0.5, "Missing MLP results", ha="center", va="center", fontsize=FONT_TICK)
            ax.set_axis_off()
    else:
        per_client = mlp_high_het_results["evaluations"]["per_client_local_test"]
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
            adapt = entry["adaptive_ensemble"]
            ax.plot([p[0] for p in loc["curve"]],
                    [p[1] for p in loc["curve"]],
                    marker="o", markersize=3.5, linestyle="-", linewidth=3,
                    color=result_colors.get("client-local", "tab:brown"),
                    label=f"Client-Local ({loc['auc']:.2f})")
            ax.plot([p[0] for p in glob["curve"]],
                    [p[1] for p in glob["curve"]],
                    marker="o", markersize=3.5, linestyle="-", linewidth=3,
                    color=result_colors.get("global", "tab:red"),
                    label=f"Federated ({glob['auc']:.2f})")
            ax.plot([p[0] for p in adapt["curve"]],
                    [p[1] for p in adapt["curve"]],
                    marker="o", markersize=3.5, linestyle="--", linewidth=3,
                    color='blue',#result_colors.get("adaptive-ensemble", "tab:blue"),
                    label=f"Personalized ({adapt['auc']:.2f})")
            ax.set_title(f"MLP Client {display_id}", fontsize=FONT_TICK+2.5)
            ax.tick_params(axis="both", labelsize=FONT_TICK+1.5)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=max(FONT_LEGEND - 2, 8)+4, handletextpad=0.10, borderaxespad=0.10, columnspacing=0.15)
            if col == 0:
                ax.set_ylabel("Accuracy", fontsize=FONT_AXISLABEL+2)
            
            ax.set_ylim(bottom=0.46)

            # NEW: consistent scientific formatting for cost axis (hide offset in top row)
            format_cost_axis(ax, show_offset_text=False)

    # Bottom row: K-Means
    if kmeans_high_het_results is None:
        for col, display_id in enumerate(selected_display):
            ax = axes[col + 3]
            ax.set_title(f"K-Means Client {display_id}", fontsize=FONT_TICK)
            ax.text(0.5, 0.5, "Missing K-Means results", ha="center", va="center", fontsize=FONT_TICK)
            ax.set_axis_off()
    else:
        per_client = kmeans_high_het_results["evaluations"]["per_client_local_test"]
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
            adapt = entry["adaptive_ensemble"]
            ax.plot([p[0] for p in loc["curve"]],
                    [p[1] for p in loc["curve"]],
                    marker="o", markersize=4, linestyle="-", linewidth=3,
                    color=result_colors.get("client-local", "tab:brown"),
                    label=f"Client-Local ({loc['auc']:.2f})")
            ax.plot([p[0] for p in glob["curve"]],
                    [p[1] for p in glob["curve"]],
                    marker="o", markersize=4, linestyle="-", linewidth=3,
                    color=result_colors.get("global", "tab:red"),
                    label=f"Federated ({glob['auc']:.2f})")
            ax.plot([p[0] for p in adapt["curve"]],
                    [p[1] for p in adapt["curve"]],
                    marker="o", markersize=4, linestyle="--", linewidth=3,
                    color='blue',#result_colors.get("adaptive-ensemble", "tab:blue"),
                    label=f"Personalized ({adapt['auc']:.2f})")
            ax.set_title(f"K-Means Client {display_id}", fontsize=FONT_TICK+2.5)
            ax.tick_params(axis="both", labelsize=FONT_TICK+1.5)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.legend(fontsize=max(FONT_LEGEND - 2, 8)+4, handletextpad=0.10, borderaxespad=0.10, columnspacing=0.15)
            if col == 0:
                ax.set_ylabel("Accuracy", fontsize=FONT_AXISLABEL+3)
            ax.set_xlabel("Avg Cost (in $)", fontsize=FONT_AXISLABEL+3)
            ax.set_ylim(bottom=0.46)
            # NEW: consistent scientific formatting for cost axis (show offset in bottom row)
            format_cost_axis(ax, show_offset_text=True)

    fig.suptitle(
        f"Local Test Performance "
        "of Client-Local (No-FL) vs Federated vs Personalized",
        fontsize=FONT_SUPTITLE+2.5, y=0.97
    )
    plt.tight_layout(rect=(0, 0, 1, 1))
    plt.savefig(
        os.path.join(high_het_out_folder, "high_het_local_global_adaptive_selected_clients_mlp_kmeans.pdf"),
        bbox_inches="tight"
    )
    plt.close()


########################################################
## Plot: Model heterogeneity of train sets
########################################################
import json
import numpy as np
import matplotlib.pyplot as plt

# Flag: Normalize within every client or across all clients+models
NORMALIZE_WITHIN_CLIENT = True  # Set to False for global normalization, True for per-client normalization

# Set file path for kmeans experiment results
with open(KMEANS_RESULTS_PATH, "r", encoding="utf-8") as f:
    kmeans_results = json.load(f)

per_client_model_active_samples = kmeans_results.get("per_client_model_active_samples", {})

# Fetch model names directly from results (matches the kmeans_out JSON)
model_names = kmeans_results.get("model_names", [])
if not model_names:
    raise RuntimeError("No 'model_names' found in kmeans_results. Please check the results JSON.")


# --- Helper: map model name to readable legend name ---
def pretty_model_name(raw):
    # Manual list; add as needed
    replacements = {
        "WizardLM/WizardLM-13B-V1.2": "WizardLM 13B V1.2",
        "claude-instant-v1": "Claude Instant v1",
        "claude-v1": "Claude v1",
        "claude-v2": "Claude v2",
        "gpt-3.5-turbo-1106": "GPT-3.5 Turbo 1106",
        "gpt-4-1106-preview": "GPT-4 1106 Preview",
        "meta/code-llama-instruct-34b-chat": "CodeLlama Instruct 34B Chat",
        "meta/llama-2-70b-chat": "Llama 2 70B Chat",
        "mistralai/mistral-7b-chat": "Mistral 7B Chat",
        "mistralai/mixtral-8x7b-chat": "Mixtral 8x7B Chat",
        "zero-one-ai/Yi-34B-Chat": "Yi 34B Chat",
    }
    if raw in replacements:
        return replacements[raw]
    # Otherwise generic prettifying: take final part after /, replace - and _, capitalize
    pretty = raw.split("/")[-1]
    pretty = pretty.replace("-", " ").replace("_", " ")
    pretty = " ".join([x.capitalize() if not x.isupper() else x for x in pretty.split()])
    return pretty

pretty_model_names = [pretty_model_name(m) for m in model_names]


# Prepare data
client_ids = sorted(per_client_model_active_samples.keys(), key=lambda x: int(x))
num_clients = len(client_ids)
num_models = max(len(per_client_model_active_samples[cid]) for cid in client_ids)

# Create matrix of sample counts (clients x models)
counts_matrix = np.zeros((num_clients, num_models), dtype=int)
for i, cid in enumerate(client_ids):
    row = per_client_model_active_samples[cid]
    counts_matrix[i, :len(row)] = row

if NORMALIZE_WITHIN_CLIENT:
    # Normalize counts within every client (row)
    norm_counts = np.zeros_like(counts_matrix, dtype=float)
    for i in range(num_clients):
        row = counts_matrix[i, :]
        min_count = row.min() if row.size > 0 else 1
        max_count = row.max() if row.size > 0 else 1
        # Avoid division by zero if all values are equal
        if max_count - min_count == 0:
            norm_counts[i, :] = 0.0
        else:
            norm_counts[i, :] = (row - min_count) / (max_count - min_count + 1e-8)
else:
    # Normalize counts across all clients+models (as before)
    min_count = counts_matrix.min() if counts_matrix.size > 0 else 1
    max_count = counts_matrix.max() if counts_matrix.size > 0 else 1
    norm_counts = (counts_matrix - min_count) / (max_count - min_count + 1e-8)  # avoid division by 0

print("DEBUG: norm_counts[0]:", norm_counts[0])

# For better visualization, use a minimum size
min_marker_size = 5
max_marker_size = 400

# Generate a color for each model
import matplotlib.cm
from matplotlib.colors import to_hex

def get_distinct_colors(n):
    """
    Generate n visually distinct colors, trying to avoid near-duplicates for adjacent indices.
    Uses tab10, tab20, then hsv if needed.
    """
    # Try tab20 if n > 10
    if n <= 10:
        cmap = plt.get_cmap("tab10", n)
        tab_colors = [to_hex(cmap(i)) for i in range(n)]
    elif n <= 20:
        cmap = plt.get_cmap("tab20", n)
        tab_colors = [to_hex(cmap(i)) for i in range(n)]
    else:
        # fallback: evenly spaced HSV
        tab_colors = [to_hex(plt.cm.hsv(i / n)) for i in range(n)]
    # Fix for neighboring same colors:
    # If tab10 or tab20 repeat at the end (for >10 or >20), rotate last color
    for i in range(1, n):
        if tab_colors[i] == tab_colors[i-1]:
            # try simple color swap by shifting hsv
            from colorsys import hsv_to_rgb, rgb_to_hsv
            rgb = plt.colors.to_rgb(tab_colors[i])
            h, s, v = rgb_to_hsv(*rgb)
            h_new = (h + 0.1) % 1.0
            rgb_new = hsv_to_rgb(h_new, s, v)
            tab_colors[i] = to_hex(rgb_new)
    return tab_colors

colors = get_distinct_colors(num_models)


# Plot
fig, ax = plt.subplots(figsize=(max(6, num_models * 0.7), max(4, num_clients * 0.45)))

scatter_plots = []
for model_idx in range(num_models):
    xs = []
    ys = []
    sizes = []
    for client_idx in range(num_clients):
        xs.append(model_idx)
        ys.append(client_idx)
        s = norm_counts[client_idx, model_idx]
        size = min_marker_size + s * (max_marker_size - min_marker_size)
        sizes.append(size)
    dbg_label = pretty_model_names[model_idx] if model_idx < len(pretty_model_names) else f"Model {model_idx}"

    scatter = ax.scatter(
        xs, ys,
        s=sizes,
        c=[colors[model_idx]]*num_clients,
        alpha=0.75,
        label=dbg_label
    )
    scatter_plots.append(scatter)

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Client ID", fontsize=12)
ax.set_xticks(np.arange(num_models))
# ax.set_xticklabels(pretty_model_names, rotation=45, ha="right")
ax.set_yticks(np.arange(num_clients))
ax.set_yticklabels([str(cid) for cid in client_ids])

if NORMALIZE_WITHIN_CLIENT:
    ax.set_title("Distribution of Models in Clients' Train Sets", fontsize=14)
else:
    ax.set_title("Distribution of Models in Clients' Train Sets", fontsize=14)
ax.grid(True, axis="both", linestyle="--", alpha=0.3)
legend = ax.legend(
    handles=scatter_plots,
    labels=pretty_model_names if len(pretty_model_names) == num_models else [f"Model {i}" for i in range(num_models)],
    title="Models",
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    fontsize=10,
    title_fontsize=11
)

plt.tight_layout()
plt.savefig(os.path.join(result_out_folder, "client_model_activesamples_bubble.pdf"), bbox_inches="tight")
plt.close()




########################################################
## Plot: Representatives side by side 
########################################################
########################################################
## Plot: Representative Clients (4, 6, 8) on LOCAL TEST (MLP top, K-Means bottom)
########################################################

# These two codes plot two different figures. I want to merge them in the same figure, left will be the top one, right will be the bottom one. in total 2x6 subplots but each part will have its own sup title, also only single y label per row (just on the lefts). give brief separation between groups as well.


# if (mlp_results is None) and (kmeans_results is None):
#     print("Skip selected client plot: no results loaded.")
# else:
#     selected_display = (4, 6, 8)
#     nrows, ncols = 2, 3
#     fig, axes = plt.subplots(nrows, ncols, figsize=(10, 6), sharex=False, sharey=False)
#     axes = axes.flatten()

#     # Top row: MLP
#     if mlp_results is None:
#         for col, display_id in enumerate(selected_display):
#             ax = axes[col]
#             ax.set_title(f"MLP Client {display_id}", fontsize=FONT_TICK)
#             ax.text(0.5, 0.5, "Missing MLP results", ha="center", va="center", fontsize=FONT_TICK)
#             ax.set_axis_off()
#     else:
#         per_client = mlp_results["evaluations"]["per_client_local_test"]
#         for col, display_id in enumerate(selected_display):
#             ax = axes[col]
#             cid = display_id - 1
#             key = str(cid)
#             if key not in per_client:
#                 ax.set_title(f"MLP Client {display_id}", fontsize=FONT_TICK)
#                 ax.text(0.5, 0.5, "Missing client", ha="center", va="center", fontsize=FONT_TICK)
#                 ax.set_axis_off()
#                 continue
#             entry = per_client[key]
#             loc = entry["local_only"]
#             glob = entry["global"]
#             ax.plot([p[0] for p in loc["curve"]],
#                     [p[1] for p in loc["curve"]],
#                     marker="o", markersize=6, markevery=5, linestyle="-", linewidth=3,
#                     color=result_colors.get("client-local", "tab:brown"),
#                     label=f"Client-Local ({loc['auc']:.2f})")
#             ax.plot([p[0] for p in glob["curve"]],
#                     [p[1] for p in glob["curve"]],
#                     marker="o", markersize=6, markevery=5, linestyle="-", linewidth=3,
#                     color=result_colors.get("global", "tab:red"),
#                     label=f"Federated ({glob['auc']:.2f})")
#             ax.set_ylim(bottom=0.45)
#             ax.set_title(f"MLP Client {display_id}", fontsize=FONT_TICK+2.5)
#             ax.tick_params(axis="both", labelsize=FONT_TICK+1.5)
#             ax.grid(True, linestyle="-", alpha=0.4)
#             ax.legend(fontsize=max(FONT_LEGEND - 2, 8)+4, handletextpad=0.10, borderaxespad=0.10, columnspacing=0.15)
#             if col == 0:
#                 ax.set_ylabel("Accuracy", fontsize=FONT_AXISLABEL+2)

#             # NEW: consistent scientific formatting for cost axis
#             # Top row has no x-label; hide the offset text to reduce clutter.
#             format_cost_axis(ax, show_offset_text=False)
#             # Xlabel only on bottom row, done below

#     # Bottom row: K-Means
#     if kmeans_results is None:
#         for col, display_id in enumerate(selected_display):
#             ax = axes[col + 3]
#             ax.set_title(f"K-Means Client {display_id}", fontsize=FONT_TICK)
#             ax.text(0.5, 0.5, "Missing K-Means results", ha="center", va="center", fontsize=FONT_TICK)
#             ax.set_axis_off()
#     else:
#         per_client = kmeans_results["evaluations"]["per_client_local_test"]
#         for col, display_id in enumerate(selected_display):
#             ax = axes[col + 3]
#             cid = display_id - 1
#             key = str(cid)
#             if key not in per_client:
#                 ax.set_title(f"K-Means Client {display_id}", fontsize=FONT_TICK)
#                 ax.text(0.5, 0.5, "Missing client", ha="center", va="center", fontsize=FONT_TICK)
#                 ax.set_axis_off()
#                 continue
#             entry = per_client[key]
#             loc = entry["local_only"]
#             glob = entry["global"]
#             ax.plot([p[0] for p in loc["curve"]],
#                     [p[1] for p in loc["curve"]],
#                     marker="o", markersize= 6, markevery=5, linestyle="-", linewidth=3,
#                     color=result_colors.get("client-local", "tab:brown"),
#                     label=f"Client-Local ({loc['auc']:.2f})")
#             ax.plot([p[0] for p in glob["curve"]],
#                     [p[1] for p in glob["curve"]],
#                     marker="o", markersize=6, markevery=5, linestyle="-", linewidth=3,
#                     color=result_colors.get("global", "tab:red"),
#                     label=f"Federated ({glob['auc']:.2f})")
#             ax.set_ylim(bottom=0.45)
#             ax.set_title(f"K-Means Client {display_id}", fontsize=FONT_TICK+2.5)
#             ax.tick_params(axis="both", labelsize=FONT_TICK+1.5)
#             ax.grid(True, linestyle="-", alpha=0.4)
#             ax.legend(fontsize=max(FONT_LEGEND - 2, 8)+4, handletextpad=0.10, borderaxespad=0.10, columnspacing=0.15)
#             if col == 0:
#                 ax.set_ylabel("Accuracy", fontsize=FONT_AXISLABEL+3)
#             ax.set_xlabel("Avg Cost (in $)", fontsize=FONT_AXISLABEL+3)

#             # NEW: consistent scientific formatting for cost axis
#             # Bottom row has x-label; show offset text.
#             format_cost_axis(ax, show_offset_text=True)

#     fig.suptitle(
#         f"Representative Clients {tuple(selected_display)}: Local Test Performance of Federated vs Client-Local", fontsize=FONT_SUPTITLE+2.5, y=0.97
#     )
#     plt.tight_layout(rect=(0, 0, 1, 1)) # remove unnecessary whitespaces
#     plt.savefig(os.path.join(result_out_folder, "local_vs_global_local_test_selected_clients_mlp_kmeans.pdf"), bbox_inches="tight")
#     plt.close()


# ########################################################
# ## Plot: Representative Clients (4, 6, 8) on LOCAL TEST (MLP top, K-Means bottom)
# ########################################################
# if (mlp_high_het_results is None) and (kmeans_high_het_results is None):
#     print("Skip high-het selected client plot: no results loaded.")
# else:
#     selected_display = (4, 6, 8)
#     nrows, ncols = 2, 3
#     fig, axes = plt.subplots(nrows, ncols, figsize=(10, 6), sharex=False, sharey=False)
#     axes = axes.flatten()

#     # Top row: MLP
#     if mlp_high_het_results is None:
#         for col, display_id in enumerate(selected_display):
#             ax = axes[col]
#             ax.set_title(f"MLP Client {display_id}", fontsize=FONT_TICK)
#             ax.text(0.5, 0.5, "Missing MLP results", ha="center", va="center", fontsize=FONT_TICK)
#             ax.set_axis_off()
#     else:
#         per_client = mlp_high_het_results["evaluations"]["per_client_local_test"]
#         for col, display_id in enumerate(selected_display):
#             ax = axes[col]
#             cid = display_id - 1
#             key = str(cid)
#             if key not in per_client:
#                 ax.set_title(f"MLP Client {display_id}", fontsize=FONT_TICK)
#                 ax.text(0.5, 0.5, "Missing client", ha="center", va="center", fontsize=FONT_TICK)
#                 ax.set_axis_off()
#                 continue
#             entry = per_client[key]
#             loc = entry["local_only"]
#             glob = entry["global"]
#             adapt = entry["adaptive_ensemble"]
#             ax.plot([p[0] for p in loc["curve"]],
#                     [p[1] for p in loc["curve"]],
#                     marker="o", markersize=3.5, linestyle="-", linewidth=3,
#                     color=result_colors.get("client-local", "tab:brown"),
#                     label=f"Client-Local ({loc['auc']:.2f})")
#             ax.plot([p[0] for p in glob["curve"]],
#                     [p[1] for p in glob["curve"]],
#                     marker="o", markersize=3.5, linestyle="-", linewidth=3,
#                     color=result_colors.get("global", "tab:red"),
#                     label=f"Federated ({glob['auc']:.2f})")
#             ax.plot([p[0] for p in adapt["curve"]],
#                     [p[1] for p in adapt["curve"]],
#                     marker="o", markersize=3.5, linestyle="--", linewidth=3,
#                     color='blue',#result_colors.get("adaptive-ensemble", "tab:blue"),
#                     label=f"Personalized ({adapt['auc']:.2f})")
#             ax.set_title(f"MLP Client {display_id}", fontsize=FONT_TICK+2.5)
#             ax.tick_params(axis="both", labelsize=FONT_TICK+1.5)
#             ax.grid(True, linestyle="--", alpha=0.4)
#             ax.legend(fontsize=max(FONT_LEGEND - 2, 8)+4, handletextpad=0.10, borderaxespad=0.10, columnspacing=0.15)
#             if col == 0:
#                 ax.set_ylabel("Accuracy", fontsize=FONT_AXISLABEL+2)
            
#             ax.set_ylim(bottom=0.46)

#             # NEW: consistent scientific formatting for cost axis (hide offset in top row)
#             format_cost_axis(ax, show_offset_text=False)

#     # Bottom row: K-Means
#     if kmeans_high_het_results is None:
#         for col, display_id in enumerate(selected_display):
#             ax = axes[col + 3]
#             ax.set_title(f"K-Means Client {display_id}", fontsize=FONT_TICK)
#             ax.text(0.5, 0.5, "Missing K-Means results", ha="center", va="center", fontsize=FONT_TICK)
#             ax.set_axis_off()
#     else:
#         per_client = kmeans_high_het_results["evaluations"]["per_client_local_test"]
#         for col, display_id in enumerate(selected_display):
#             ax = axes[col + 3]
#             cid = display_id - 1
#             key = str(cid)
#             if key not in per_client:
#                 ax.set_title(f"K-Means Client {display_id}", fontsize=FONT_TICK)
#                 ax.text(0.5, 0.5, "Missing client", ha="center", va="center", fontsize=FONT_TICK)
#                 ax.set_axis_off()
#                 continue
#             entry = per_client[key]
#             loc = entry["local_only"]
#             glob = entry["global"]
#             adapt = entry["adaptive_ensemble"]
#             ax.plot([p[0] for p in loc["curve"]],
#                     [p[1] for p in loc["curve"]],
#                     marker="o", markersize=4, linestyle="-", linewidth=3,
#                     color=result_colors.get("client-local", "tab:brown"),
#                     label=f"Client-Local ({loc['auc']:.2f})")
#             ax.plot([p[0] for p in glob["curve"]],
#                     [p[1] for p in glob["curve"]],
#                     marker="o", markersize=4, linestyle="-", linewidth=3,
#                     color=result_colors.get("global", "tab:red"),
#                     label=f"Federated ({glob['auc']:.2f})")
#             ax.plot([p[0] for p in adapt["curve"]],
#                     [p[1] for p in adapt["curve"]],
#                     marker="o", markersize=4, linestyle="--", linewidth=3,
#                     color='blue',#result_colors.get("adaptive-ensemble", "tab:blue"),
#                     label=f"Personalized ({adapt['auc']:.2f})")
#             ax.set_title(f"K-Means Client {display_id}", fontsize=FONT_TICK+2.5)
#             ax.tick_params(axis="both", labelsize=FONT_TICK+1.5)
#             ax.grid(True, linestyle="--", alpha=0.4)
#             ax.legend(fontsize=max(FONT_LEGEND - 2, 8)+4, handletextpad=0.10, borderaxespad=0.10, columnspacing=0.15)
#             if col == 0:
#                 ax.set_ylabel("Accuracy", fontsize=FONT_AXISLABEL+3)
#             ax.set_xlabel("Avg Cost (in $)", fontsize=FONT_AXISLABEL+3)
#             ax.set_ylim(bottom=0.46)
#             # NEW: consistent scientific formatting for cost axis (show offset in bottom row)
#             format_cost_axis(ax, show_offset_text=True)

#     fig.suptitle(
#         f"Local Test Performance "
#         "of Client-Local (No-FL) vs Federated vs Personalized",
#         fontsize=FONT_SUPTITLE+2.5, y=0.97
#     )
#     plt.tight_layout(rect=(0, 0, 1, 1))
#     plt.savefig(
#         os.path.join(high_het_out_folder, "high_het_local_global_adaptive_selected_clients_mlp_kmeans.pdf"),
#         bbox_inches="tight"
#     )
#     plt.close()


























































