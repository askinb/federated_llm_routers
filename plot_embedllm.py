"""
Federated EmbedLLM comparison figures (Appendix H, Figures 27 and 28).

Reads the results written by routerbench/ and proxrouter/ (fl_mlp_router.py, fl_kmeans_router.py,
fl_mf_router.py), so run those first. Figures are saved to ./paper_results/.
"""
import os, json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Computer Modern Roman", "Times New Roman"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FONT_AX   = 12
FONT_T    = 12
FONT_TICK = 11
FONT_LEG  = 9
FONT_SUP  = 14

OUT = "./paper_results/"
os.makedirs(OUT, exist_ok=True)

def load(path):
    if not os.path.exists(path):
        print(f"  [SKIP] {path}")
        return None
    with open(path) as f:
        return json.load(f)

def _c(curve): return [p[0] for p in curve]
def _a(curve): return [p[1] for p in curve]

MLP_RB   = load("routerbench/mlp_out/mlp_experiment_results.json")
MLP_PR   = load("proxrouter/mlp_out/mlp_experiment_results.json")
KM_RB    = load("routerbench/kmeans_out/kmeans_experiment_results.json")
KM_PR    = load("proxrouter/kmeans_out/kmeans_experiment_results.json")
EMBED_RB = load("routerbench/mf_out/mf_experiment_results.json")
EMBED_PR = load("proxrouter/mf_out/mf_experiment_results.json")


# Figure 27: federated MLP vs K-Means vs EmbedLLM on the global test set
def plot_three_methods_global():
    datasets = [
        ("RouterBench", MLP_RB, KM_RB, EMBED_RB, -3, 0.4),
        ("ProxRouter",  MLP_PR, KM_PR, EMBED_PR, -5, 0.4),
    ]
    if not any(all([mlp, km, emb]) for _, mlp, km, emb, _, _ in datasets):
        print("  [SKIP] Not enough data for 3-method comparison (NeurIPS)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    axes = axes[0]

    for col, (tag, mlp, km, emb, sci_pow, ylim_bot) in enumerate(datasets):
        ax = axes[col]

        if mlp:
            c_mlp = mlp["evaluations"]["global_test"]["global_model"]
            ax.plot(_c(c_mlp["curve"]), _a(c_mlp["curve"]),
                    color="tab:blue", ls="-", lw=2.2,
                    marker="o", ms=6, markevery=5,
                    label=f'Federated MLP ({c_mlp["auc"]:.2f})')
        if km:
            c_km = km["evaluations"]["global_test"]["global_model"]
            ax.plot(_c(c_km["curve"]), _a(c_km["curve"]),
                    color="tab:orange", ls="-", lw=2.2,
                    marker="s", ms=6, markevery=5,
                    label=f'Federated K-Means ({c_km["auc"]:.2f})')
        if emb:
            c_emb = emb["evaluations"]["global_test"]["global_model"]
            ax.plot(_c(c_emb["curve"]), _a(c_emb["curve"]),
                    color="tab:green", ls="-", lw=2.2,
                    marker="^", ms=6, markevery=5,
                    label=f'Federated EmbedLLM ({c_emb["auc"]:.2f})')

        fmt = mticker.ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((sci_pow, sci_pow))
        ax.xaxis.set_major_formatter(fmt)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(sci_pow, sci_pow), useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(FONT_TICK)

        ax.set_xlabel("Average Cost per Query", fontsize=FONT_AX)
        if col == 0:
            ax.set_ylabel("Average Accuracy", fontsize=FONT_AX)
        ax.set_title(tag, fontsize=FONT_T + 1)
        ax.legend(fontsize=FONT_LEG, loc="lower right")
        ax.tick_params(labelsize=FONT_TICK)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=ylim_bot)

    fig.suptitle("Federated Routers on Global Test: MLP vs K-Means vs EmbedLLM",
                 fontsize=FONT_SUP, y=1.02)
    fig.tight_layout()
    path = f"{OUT}embedllm_vs_mlp_kmeans_global_test.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# Figure 28: federated EmbedLLM vs client-local EmbedLLM on the global test set
def plot_embedllm_global_test():
    datasets = [
        ("RouterBench", EMBED_RB, -3),
        ("ProxRouter",  EMBED_PR, -5),
    ]
    if not any(d for _, d, _ in datasets):
        print("  [SKIP] No EmbedLLM data for plot 10 (NeurIPS)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), squeeze=False)
    axes = axes[0]
    cmap = plt.get_cmap("tab20")

    for col, (tag, emb, sci_pow) in enumerate(datasets):
        ax = axes[col]
        if not emb:
            ax.set_title(tag, fontsize=FONT_T + 1)
            ax.text(0.5, 0.5, "Missing", ha="center", va="center")
            continue

        gt = emb["evaluations"]["global_test"]["global_model"]
        pcogt = emb["evaluations"].get("per_client_on_global_test", {})

        ax.plot(_c(gt["curve"]), _a(gt["curve"]),
                color="tab:blue", lw=2.5, marker="o", markevery=5, ms=5,
                label=f'Federated ({gt["auc"]:.2f})')

        for i, cid in enumerate(sorted(pcogt.keys(), key=int)):
            lo = pcogt[cid].get("local_only", None)
            if lo:
                ax.plot(_c(lo["curve"]), _a(lo["curve"]),
                        color=cmap(i % 20), lw=1.2, marker="o", markevery=5, ms=3,
                        label=f'Client {int(cid)+1} ({lo["auc"]:.2f})')

        fmt = mticker.ScalarFormatter(useMathText=True)
        fmt.set_scientific(True)
        fmt.set_powerlimits((sci_pow, sci_pow))
        ax.xaxis.set_major_formatter(fmt)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(sci_pow, sci_pow), useMathText=True)
        ax.xaxis.get_offset_text().set_fontsize(FONT_TICK)

        if col == 0:
            ax.set_ylabel("Accuracy (EmbedLLM)", fontsize=FONT_AX)
        ax.set_xlabel("Avg Cost (in $)", fontsize=FONT_AX)
        ax.set_title(tag, fontsize=FONT_T + 1)
        ax.tick_params(labelsize=FONT_TICK)
        ax.grid(True, ls="--", alpha=0.5)
        ax.set_ylim(bottom=0.35)

        handles, labels = ax.get_legend_handles_labels()
        leg = ax.legend(handles, labels,
                        fontsize=FONT_LEG - 0.5, ncol=3, loc="lower center",
                        bbox_to_anchor=(0.56, 0),
                        columnspacing=0.08, handletextpad=0.13, borderaxespad=0.14)
        leg.get_frame().set_edgecolor("none")

    fig.suptitle("Global Test: Federated EmbedLLM vs Client-Local (No-FL)",
                 fontsize=FONT_SUP, y=1.02)
    fig.tight_layout()
    path = f"{OUT}embedllm_fed_vs_local_global_test.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


if __name__ == "__main__":
    plot_three_methods_global()
    plot_embedllm_global_test()
