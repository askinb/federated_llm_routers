# Datasets

Three LLM-routing datasets with precomputed query embeddings, one row per query.

| File | Dataset | Queries | Models | Source datasets | Task labels (`eval_name`) | Size |
|---|---|---|---|---|---|---|
| `routerbench_0shot_w4emb.parquet` | RouterBench [1] | 36,497 | 11 | 8 | 86 | 126 MB |
| `proxrouter_train.parquet` | ProxRouter-Data [2] | 18,825 | 14 | 12 | 12 | 57 MB |
| `sprout_w4emb.parquet` | SPROUT [3] | 44,241 | 13 | 6 | 20 | 140 MB |

Task labels are finer than source datasets (e.g., RouterBench splits MMLU into its 57 subjects); clients are
formed by a Dirichlet partition over the task labels.

## Getting the files

The parquet files are stored in `chunks/` as pieces of at most 22 MiB, so they fit in a Git repository
without Git LFS. You do not need to do anything: every experiment script calls
`reassemble.ensure_dataset()`, which joins the chunks into `data/<name>.parquet` the first time a dataset is
used and verifies it against `SHA256SUMS`. To rebuild all three up front:

```bash
python data/reassemble.py
```

The rebuilt `.parquet` files are git-ignored; `chunks/` is the copy under version control.

## Columns

| Column | Content |
|---|---|
| `prompt` | Query text |
| `eval_name` | Task / subtask label; clients are formed by a Dirichlet partition over these labels |
| `<model>` | Quality of `<model>`'s response to the query in [0, 1]: binary correctness for ProxRouter-Data; mostly binary for RouterBench (exact match, plus graded scores on its LLM-judged tasks); graded scores for SPROUT |
| `<model>\|total_cost` | Inference cost of that response in USD |
| `<model>\|model_response` | Response text (RouterBench only; not used by the experiments) |
| `all_mpnet_base_v2_embedding` | 768-d query embedding (float32) |
| `sample_id`, `oracle_model_to_route_to` | Carried over from the source dataset; not used by the experiments |
| `split` | SPROUT's original train/validation/test label; not used (all rows are pooled and re-partitioned across clients) |

## How the embeddings were computed

Each `prompt` was encoded with [`sentence-transformers/all-mpnet-base-v2`](https://huggingface.co/sentence-transformers/all-mpnet-base-v2)
(`SentenceTransformer.encode`, batch size 64), L2-normalized, and stored as float32. 


## Sources

Please cite the original datasets if you use this data.

1. Q. J. Hu, J. Bieker, X. Li, N. Jiang, B. Keigwin, G. Ranganath, K. Keutzer, S. K. Upadhyay.
   *RouterBench: A Benchmark for Multi-LLM Routing System.* Agentic Markets Workshop at ICML 2024.
   [ICML 2024 Workshop Agentic Markets](https://openreview.net/forum?id=IVXmV8Uxwh)
2. S. Patel, N. Jali, A. Mallick, G. Joshi.
   *ProxRouter: Proximity-Weighted LLM Query Routing for Improved Robustness to Outliers.* 2026.
   [AISTATS 2026](https://openreview.net/forum?id=GYWls3tyqM)
3. S. Somerstep, F. M. Polo, A. F. M. de Oliveira, P. Mangal, M. Silva, O. Bhardwaj, M. Yurochkin, S. Maity.
   *CARROT: A Cost Aware Rate Optimal Router.* ICLR 2025 Workshop on Foundation Models in the Wild.
   [ICLR 2025 Workshop FM-Wild](https://openreview.net/forum?id=xEBOy2ze1U)
