# Federate the Router: Learning Language Model Routers with Sparse and Decentralized Evaluations

**Baris Askin\*, Shivam Patel\*, Anupam Nayak\*, Andrea Vigano, Jiin Woo, Gauri Joshi, Carlee Joe-Wong**
<br>NeurIPS 2026 · [Paper Link](https://arxiv.org/abs/2601.22318)
<br><sub>\*Equal contribution</sub>

## Abstract

Large language models (LLMs) are increasingly accessed as remotely hosted services by edge and enterprise
clients that cannot run frontier models locally. Since models vary widely in capability and price, routing
queries to models that balance quality and inference cost is essential. Existing router approaches assume
access to centralized query–model evaluation data. However, these data are often fragmented across clients,
such as end users and organizations, and are privacy-sensitive, which makes centralizing data infeasible.
Additionally, per-client router training is ineffective since local evaluation data is limited and covers
only a restricted query distribution and a biased subset of model evaluations. We introduce the first
federated framework for LLM routing, enabling clients to learn a shared routing policy from local offline
query–model evaluation data. Our framework supports both parametric multilayer perceptron router and
nonparametric K-means router under heterogeneous client query distributions and non-uniform model coverage.
Across three benchmarks, federated collaboration improves the accuracy–cost frontier over client-local
routers, both via increased effective model coverage and better query generalization. Our theoretical
results also validate that federated training reduces routing suboptimality.

## Repository structure

```
├── data/              datasets (RouterBench, ProxRouter, SPROUT) with query embeddings
├── routerbench/       RouterBench experiments
├── proxrouter/        ProxRouter experiments
├── sprout/            SPROUT experiments
└── plot_embedllm.py   EmbedLLM comparison figures
```

Each dataset folder contains:

- `fl_mlp_router.py`, `fl_kmeans_router.py`: federated vs. client-local and centralized routers
- `*_model_expansion.py`: new models join the pool
- `*_new_clients.py`: new clients join the system
- `*_high_het.py`: high heterogeneity and adaptive personalization
- `fl_mf_router.py`: federated EmbedLLM baseline
- `plot_router_results.py`: paper figures from the saved results

`sprout/` contains the main experiments (`fl_mlp_router.py`, `fl_kmeans_router.py`) and plotting.

## Usage

```bash
pip install -r requirements.txt

python routerbench/fl_mlp_router.py          # run an experiment
python routerbench/plot_router_results.py    # plot its results
```

The datasets are included in `data/` and prepared automatically on the first run (see
[`data/README.md`](data/README.md)).

## Citation

```bibtex
@inproceedings{askin2026federate,
  title     = {Federate the Router: Learning Language Model Routers with Sparse and Decentralized Evaluations},
  author    = {Askin, Baris and Patel, Shivam and Nayak, Anupam and Vigano, Andrea and Woo, Jiin and Joshi, Gauri and Joe-Wong, Carlee},
  booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
  year      = {2026}
}
```

## Contact

Please reach out to [Baris Askin](https://askinb.github.io) for questions and correspondence.
