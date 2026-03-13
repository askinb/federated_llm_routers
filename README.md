# Federated LLM Router Code

This repository contains the implementation code for [Federate the Router: Learning LM Routers with Sparse and Decentralized Evaluations](https://arxiv.org/abs/2601.22318).


## Requirements

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Dataset

The experiments require the dataset file `routerbench_0shot_w4emb.parquet`, and `proxrouter_train.parquet`. 

**Download the dataset from this link:** https://drive.google.com/drive/folders/1zNW5z5Hau-j0JfFFvGC5uZudtBxXvwBV?usp=share_link

Place the `.parquet` file in the same directory as the Python scripts.

## Usage

All files can be run with the following command:

```bash
python {file_name}.py
```

## Files Description

- `fl_kmeans_router*.py` - K-means based federated router implementations
- `fl_mlp_router*.py` - MLP-based federated router implementations
- `*_high_het.py` - High heterogeneity & personalization experiments
- `*_model_expansion.py` - Model expansion experiments  
- `*_new_clients.py` - New client joining experiments
- `proxrouter_experiments/` - Additional experimental variants with proxrouter dataset

## Output

Each script will create its own output directory with results and visualizations.

For questions/comments, please send an email to [Baris Askin](https://askinb.github.io).