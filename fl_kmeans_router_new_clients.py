import os
import math
import json
import ast
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

OUTPUT_DIR = "./client_expansion_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 100
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
DEVICE = torch.device(DEVICE)
print("Using device:", DEVICE)

P = "./routerbench_0shot_w4emb.parquet"
EMB_COL = "all_mpnet_base_v2_embedding"
CHAR_WORD_LEN_FLAG = False

N_CLIENTS = 10

N_HOLD = 3  # 3 number of clients joining after base training
HOLDOUT_TASK_FRAC = 0.3  # 0.4 for testing

DIRICHLET_ALPHA = 0.6
MODEL_COV_MINMAX = (1, 1)
LABEL_KEEP_FRAC = 0.1
CLIENT_TRAIN_FRAC = 0.75

K_LOCAL = 15
K_GLOBAL = 20

KMEANS_LOCAL_N_INIT = 3
KMEANS_GLOBAL_N_INIT = 3
KMEANS_LOCAL_MAX_ITER = 30
KMEANS_GLOBAL_MAX_ITER = 30

BATCH_SIZE = 256

LAMBDA_GRID = np.geomspace(1e-2, 1e7, 100)

FALLBACK_ACC = 0.0      # Zero accuracy when no samples available
FALLBACK_COST = 1e6     # Large cost when no samples available

MODEL_DIRICHLET_ALPHA = 0.45

# DATA LOADING
df = pd.read_parquet(P)
print("Loaded shape:", df.shape)
print("First 20 columns:", df.columns.tolist()[:20])

drop_cols = [c for c in df.columns if c.endswith("|model_response")]
if drop_cols:
    df = df.drop(columns=drop_cols)
    print(f"Dropped {len(drop_cols)} `|model_response` columns; new shape:", df.shape)

# PARSERS
# Detect model names via presence of '|total_cost'
cost_cols = [c for c in df.columns if c.endswith("|total_cost")]
candidate_models = [c[:-len("|total_cost")] for c in cost_cols]
MODEL_NAMES = sorted([m for m in candidate_models if m in df.columns])
K_MODELS = len(MODEL_NAMES)
print(f"Detected {K_MODELS} models:", MODEL_NAMES)


def _to_float_or_nan(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return np.nan
        try:
            return float(s)
        except Exception:
            try:
                obj = json.loads(s)
            except Exception:
                try:
                    obj = ast.literal_eval(s)
                except Exception:
                    obj = None
            if isinstance(obj, dict):
                for key in ["total_cost", "cost", "price"]:
                    if key in obj:
                        try:
                            return float(obj[key])
                        except Exception:
                            pass
            return np.nan
    if isinstance(x, dict):
        for key in ["total_cost", "cost", "price"]:
            if key in x:
                try:
                    return float(x[key])
                except Exception:
                    pass
    return np.nan


def _to_acc_float_01_or_nan(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        if isinstance(x, float) and math.isnan(x):
            return np.nan
        return float(min(1.0, max(0.0, x)))
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"nan", "none", "null"}:
            return np.nan
        try:
            v = float(s)
            return float(min(1.0, max(0.0, v)))
        except Exception:
            try:
                obj = json.loads(s)
            except Exception:
                try:
                    obj = ast.literal_eval(s)
                except Exception:
                    obj = None
            if isinstance(obj, dict):
                for key in ["is_correct", "correct", "accuracy", "acc", "score", "p"]:
                    if key in obj:
                        return _to_acc_float_01_or_nan(obj[key])
            return np.nan
    if isinstance(x, dict):
        for key in ["is_correct", "correct", "accuracy", "acc", "score", "p"]:
            if key in x:
                return _to_acc_float_01_or_nan(x[key])
    return np.nan


def prompt_to_text(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (list, tuple)):
        return " ".join([str(x) for x in v])
    return str(v)


def parse_embedding_cell(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return np.asarray(v, dtype=np.float32)
    if isinstance(v, str):
        try:
            obj = json.loads(v)
        except Exception:
            try:
                obj = ast.literal_eval(v)
            except Exception:
                obj = None
        if isinstance(obj, (list, tuple, np.ndarray)):
            return np.asarray(obj, dtype=np.float32)
    return None


def zscore(a):
    m, s = a.mean(), a.std()
    s = s if s > 1e-6 else 1.0
    return (a - m) / s

# FEATURES & LABELS
if CHAR_WORD_LEN_FLAG:
    prompt_text = df["prompt"].apply(prompt_to_text).astype(str)
    char_len = prompt_text.map(len).to_numpy(dtype=np.float32)
    word_len = prompt_text.map(lambda s: len(s.split())).to_numpy(dtype=np.float32)
    char_len_z = zscore(char_len)
    word_len_z = zscore(word_len)
else:
    char_len_z = None
    word_len_z = None

eval_names = df["eval_name"].fillna("unknown").astype(str)
eval_names_array = eval_names.to_numpy()

emb_list = [parse_embedding_cell(v) for v in df[EMB_COL].tolist()]
D_emb = next((len(e) for e in emb_list if e is not None), None)
assert D_emb is not None, f"No valid vectors in {EMB_COL}"
print("Embedding dim:", D_emb)

X_list = []
for i, e in enumerate(emb_list):
    if e is None:
        e = np.zeros(D_emb, dtype=np.float32)
    parts = [e.astype(np.float32)]
    if CHAR_WORD_LEN_FLAG:
        parts.append(np.array([char_len_z[i], word_len_z[i]], dtype=np.float32))
    x = np.concatenate(parts, axis=0)
    X_list.append(x)

X = np.stack(X_list).astype(np.float32)
N = len(df)
print("X shape:", X.shape)

expected_dim = D_emb + (2 if CHAR_WORD_LEN_FLAG else 0)
assert X.shape[1] == expected_dim, f"Expected feature dim {expected_dim}, got {X.shape[1]}"

Y_acc = np.full((N, K_MODELS), np.nan, dtype=np.float32)
M_acc = np.zeros((N, K_MODELS), dtype=np.float32)
for m_idx, m in enumerate(MODEL_NAMES):
    arr = np.array([_to_acc_float_01_or_nan(v) for v in df[m].tolist()], dtype=np.float32)
    mask = ~np.isnan(arr)
    Y_acc[:, m_idx] = np.where(mask, arr, np.nan)
    M_acc[:, m_idx] = mask.astype(np.float32)

Y_cost = np.full((N, K_MODELS), np.nan, dtype=np.float32)
M_cost = np.zeros((N, K_MODELS), dtype=np.float32)
for m_idx, m in enumerate(MODEL_NAMES):
    ccol = m + "|total_cost"
    arr = np.array([_to_float_or_nan(v) for v in df[ccol].tolist()], dtype=np.float32)
    arr = np.where(np.isnan(arr), np.nan, np.maximum(0.0, arr))
    mask = ~np.isnan(arr)
    Y_cost[:, m_idx] = np.where(mask, arr, np.nan)
    M_cost[:, m_idx] = mask.astype(np.float32)

print("Label coverage (acc, cost):", M_acc.mean(), M_cost.mean())

# CLIENT PARTITIONING

def partition_by_evalname_indices_new(eval_names: np.ndarray, n_clients: int, alpha: float = 0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    cats, inv = np.unique(eval_names, return_inverse=True)
    G = len(cats)
    client_idxs = [[] for _ in range(n_clients)]
    for g in range(G):
        cat_indices = np.where(inv == g)[0]
        cat_size = len(cat_indices)
        if cat_size == 0:
            continue
        rng.shuffle(cat_indices)
        client_props = rng.dirichlet(alpha * np.ones(n_clients))
        client_counts = np.round(client_props * cat_size).astype(int)
        diff = cat_size - client_counts.sum()
        if diff > 0:
            for _ in range(diff):
                client_counts[rng.integers(0, n_clients)] += 1
        elif diff < 0:
            for _ in range(-diff):
                candidates = np.where(client_counts > 0)[0]
                if len(candidates) > 0:
                    client_counts[rng.choice(candidates)] -= 1
        start_idx = 0
        for c in range(n_clients):
            count = client_counts[c]
            if count > 0:
                end_idx = start_idx + count
                client_idxs[c].extend(cat_indices[start_idx:end_idx])
                start_idx = end_idx
    for c in range(n_clients):
        rng.shuffle(client_idxs[c])
    return [np.array(idx, dtype=int) for idx in client_idxs]


def partition_rows_by_evalname(rows: np.ndarray, eval_names_all: np.ndarray, n_clients: int, alpha: float, rng):
    if rows.size == 0 or n_clients == 0:
        return [np.array([], dtype=int) for _ in range(n_clients)]
    eval_sub = eval_names_all[rows]
    parts = partition_by_evalname_indices_new(eval_sub, n_clients, alpha=alpha, rng=rng)
    return [rows[idx] for idx in parts]


def sample_model_subset_per_client(K, min_frac=0.6, max_frac=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    frac = rng.uniform(min_frac, max_frac)
    k = max(1, int(round(frac * K)))
    chosen = rng.choice(K, size=k, replace=False)
    mask = np.zeros(K, dtype=bool)
    mask[chosen] = True
    return mask

def generate_client_model_probabilities(n_models: int, dirichlet_alpha: Optional[float] = None, rng=None):
    """
    Generate probabilities for model selection per client using Dirichlet distribution.
    If dirichlet_alpha is None, returns uniform probabilities.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    
    if dirichlet_alpha is None:
        # Uniform probabilities
        return np.ones(n_models, dtype=np.float32) / n_models
    else:
        # Dirichlet probabilities
        return rng.dirichlet(dirichlet_alpha * np.ones(n_models)).astype(np.float32)


def apply_client_masks(Yacc, Macc, Yc, Mc, model_mask: np.ndarray, label_keep_frac: float, 
                      client_model_probs: Optional[np.ndarray] = None, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    Yacc_c, Macc_c = Yacc.copy(), Macc.copy()
    Yc_c, Mc_c = Yc.copy(), Mc.copy()
    not_kept = ~model_mask
    Yacc_c[:, not_kept] = np.nan
    Macc_c[:, not_kept] = 0.0
    Yc_c[:, not_kept] = np.nan
    Mc_c[:, not_kept] = 0.0
    if label_keep_frac < 1.0:
        n_rows, n_models = Macc_c.shape
        if n_models > 0:
            k_keep = max(1, int(math.floor(label_keep_frac * n_models)))
            keep_mask = np.zeros((n_rows, n_models), dtype=bool)
            for i in range(n_rows):
                if client_model_probs is not None:
                    # Use Dirichlet probabilities for sampling
                    chosen = rng.choice(n_models, size=k_keep, replace=False, p=client_model_probs)
                else:
                    # Uniform sampling (original behavior)
                    chosen = rng.choice(n_models, size=k_keep, replace=False)
                keep_mask[i, chosen] = True
            drop_mask = ~keep_mask
            Yacc_c[drop_mask] = np.nan
            Macc_c[drop_mask] = 0.0
            Yc_c[drop_mask] = np.nan
            Mc_c[drop_mask] = 0.0
    return Yacc_c, Macc_c, Yc_c, Mc_c


def select_holdout_tasks(eval_names_all: np.ndarray, frac: float, rng):
    unique_tasks = np.unique(eval_names_all)
    if unique_tasks.size == 0 or frac <= 0:
        return set()
    n_hold = max(1, int(round(frac * unique_tasks.size)))
    n_hold = min(n_hold, unique_tasks.size)
    chosen = rng.choice(unique_tasks, size=n_hold, replace=False)
    return set(chosen.tolist())


class CentralDataset(Dataset):
    def __init__(self, X, Yacc, Macc, Yc, Mc):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Yacc = torch.tensor(np.nan_to_num(Yacc, nan=-1.0), dtype=torch.float32)
        self.Macc = torch.tensor(Macc, dtype=torch.float32)
        self.Yc = torch.tensor(np.nan_to_num(Yc, nan=-1.0), dtype=torch.float32)
        self.Mc = torch.tensor(Mc, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        x = self.X[i]
        y_acc = self.Yacc[i]
        m_acc = self.Macc[i]
        y_cost = self.Yc[i]
        m_cost = self.Mc[i]
        return {
            "x": x,
            "y_acc": torch.where(m_acc > 0.5, y_acc, torch.zeros_like(y_acc)),
            "m_acc": m_acc,
            "y_cost": torch.where(m_cost > 0.5, y_cost, torch.zeros_like(y_cost)),
            "m_cost": m_cost,
        }


def build_clients_from_partitions(
    client_ids: List[int],
    client_rows_list: List[np.ndarray],
    X_all, Yacc_all, Macc_all, Yc_all, Mc_all,
    client_train_frac: float,
    model_cov_minmax,
    label_keep_frac: float,
    rng,
):
    clients_spec = []
    collected_train_idxs = []
    collected_test_idxs = []

    gX_parts, gYacc_parts, gMacc_parts, gYc_parts, gMc_parts = [], [], [], [], []

    for cid, rows in zip(client_ids, client_rows_list):
        if rows.size == 0:
            continue
        rows = np.array(rows, dtype=int)
        rng.shuffle(rows)

        n_local_train = int(client_train_frac * len(rows))
        local_train_rows = rows[:n_local_train]
        local_test_rows = rows[n_local_train:]

        if len(local_train_rows) == 0:
            if len(local_test_rows) > 0:
                collected_test_idxs.extend(local_test_rows.tolist())
            continue

        collected_train_idxs.extend(local_train_rows.tolist())
        if len(local_test_rows) > 0:
            collected_test_idxs.extend(local_test_rows.tolist())

        model_mask = sample_model_subset_per_client(
            K_MODELS, min_frac=model_cov_minmax[0], max_frac=model_cov_minmax[1], rng=rng
        )

        # Generate client-specific model probabilities for Dirichlet sampling
        client_probs = generate_client_model_probabilities(K_MODELS, MODEL_DIRICHLET_ALPHA, rng=rng)

        print(f"Client {cid} model probabilities:", client_probs)
        
        Yacc_train_c, Macc_train_c, Yc_train_c, Mc_train_c = apply_client_masks(
            Yacc_all[local_train_rows], Macc_all[local_train_rows],
            Yc_all[local_train_rows], Mc_all[local_train_rows],
            model_mask=model_mask, label_keep_frac=label_keep_frac, 
            client_model_probs=client_probs, rng=rng
        )

        Yacc_test_c = Yacc_all[local_test_rows].copy()
        Macc_test_c = Macc_all[local_test_rows].copy()
        Yc_test_c = Yc_all[local_test_rows].copy()
        Mc_test_c = Mc_all[local_test_rows].copy()

        train_ds = CentralDataset(
            X_all[local_train_rows], Yacc_train_c, Macc_train_c, Yc_train_c, Mc_train_c
        )
        test_ds = CentralDataset(
            X_all[local_test_rows], Yacc_test_c, Macc_test_c, Yc_test_c, Mc_test_c
        )

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            num_workers=0, pin_memory=(DEVICE.type == "cuda")
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False,
            num_workers=0, pin_memory=(DEVICE.type == "cuda")
        )

        clients_spec.append({
            "cid": cid,
            "train_idx": local_train_rows,
            "test_idx": local_test_rows,
            "model_mask": model_mask,
            "train_ds": train_ds,
            "test_ds": test_ds,
            "train_loader": train_loader,
            "test_loader": test_loader,
        })

        gX_parts.append(X_all[local_train_rows])
        gYacc_parts.append(Yacc_train_c)
        gMacc_parts.append(Macc_train_c)
        gYc_parts.append(Yc_train_c)
        gMc_parts.append(Mc_train_c)

    collected_train_idxs = np.array(collected_train_idxs, dtype=int)
    collected_test_idxs = np.array(collected_test_idxs, dtype=int)

    if len(gX_parts) > 0:
        gX = np.concatenate(gX_parts, axis=0)
        gYacc = np.concatenate(gYacc_parts, axis=0)
        gMacc = np.concatenate(gMacc_parts, axis=0)
        gYc = np.concatenate(gYc_parts, axis=0)
        gMc = np.concatenate(gMc_parts, axis=0)
    else:
        gX, gYacc, gMacc, gYc, gMc = (
            X_all[:0], Yacc_all[:0], Macc_all[:0], Yc_all[:0], Mc_all[:0]
        )

    global_train_ds = CentralDataset(gX, gYacc, gMacc, gYc, gMc)
    global_train_loader = DataLoader(
        global_train_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == "cuda")
    )

    return clients_spec, {
        "train_loader": global_train_loader,
        "train_ds": global_train_ds,
        "train_indices": collected_train_idxs,
        "test_indices": collected_test_idxs,
    }


rng = np.random.default_rng(SEED)

# Choose holdout tasks and client IDs
holdout_tasks = select_holdout_tasks(eval_names_array, HOLDOUT_TASK_FRAC, rng)
holdout_task_mask = np.isin(eval_names_array, list(holdout_tasks))

all_client_ids = np.arange(N_CLIENTS)
if N_HOLD > N_CLIENTS:
    raise ValueError("N_HOLD must be <= N_CLIENTS")

holdout_client_ids = rng.choice(all_client_ids, size=N_HOLD, replace=False)
base_client_ids = np.array([cid for cid in all_client_ids if cid not in set(holdout_client_ids)], dtype=int)

print("Holdout client IDs:", holdout_client_ids.tolist())
print("Base client IDs:", base_client_ids.tolist())
print(f"Holdout tasks: {len(holdout_tasks)} / {len(np.unique(eval_names_array))}")

rows_all = np.arange(N, dtype=int)
rows_holdout_tasks = rows_all[holdout_task_mask]
rows_base_tasks = rows_all[~holdout_task_mask]

base_client_rows = partition_rows_by_evalname(
    rows_base_tasks, eval_names_array, n_clients=len(base_client_ids), alpha=DIRICHLET_ALPHA, rng=rng
)
holdout_client_rows = partition_rows_by_evalname(
    rows_holdout_tasks, eval_names_array, n_clients=len(holdout_client_ids), alpha=DIRICHLET_ALPHA, rng=rng
)

base_clients_spec, base_global_sets = build_clients_from_partitions(
    client_ids=base_client_ids.tolist(),
    client_rows_list=base_client_rows,
    X_all=X,
    Yacc_all=Y_acc,
    Macc_all=M_acc,
    Yc_all=Y_cost,
    Mc_all=M_cost,
    client_train_frac=CLIENT_TRAIN_FRAC,
    model_cov_minmax=MODEL_COV_MINMAX,
    label_keep_frac=LABEL_KEEP_FRAC,
    rng=rng,
)

holdout_clients_spec, holdout_global_sets = build_clients_from_partitions(
    client_ids=holdout_client_ids.tolist(),
    client_rows_list=holdout_client_rows,
    X_all=X,
    Yacc_all=Y_acc,
    Macc_all=M_acc,
    Yc_all=Y_cost,
    Mc_all=M_cost,
    client_train_frac=CLIENT_TRAIN_FRAC,
    model_cov_minmax=MODEL_COV_MINMAX,
    label_keep_frac=LABEL_KEEP_FRAC,
    rng=rng,
)

all_test_indices = np.concatenate([
    base_global_sets["test_indices"],
    holdout_global_sets["test_indices"],
])

global_test_ds = CentralDataset(
    X[all_test_indices],
    Y_acc[all_test_indices], M_acc[all_test_indices],
    Y_cost[all_test_indices], M_cost[all_test_indices],
)

global_test_loader = DataLoader(
    global_test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=(DEVICE.type == "cuda")
)

print(f"Base clients: {len(base_clients_spec)}")
print(f"Holdout clients: {len(holdout_clients_spec)}")
print(f"Global TEST size: {len(all_test_indices)}")

# K-MEANS UTILS

def _kmeans_assign_labels_torch(
    X_t: torch.Tensor,
    centers_t: torch.Tensor,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    N = X_t.shape[0]
    if N == 0:
        return torch.empty((0,), device=X_t.device, dtype=torch.long)

    x_norm = (X_t * X_t).sum(dim=1)
    c_norm = (centers_t * centers_t).sum(dim=1)

    if chunk_size is None or chunk_size >= N:
        dots = X_t @ centers_t.t()
        dists = x_norm[:, None] + c_norm[None, :] - 2.0 * dots
        return torch.argmin(dists, dim=1)

    labels = torch.empty((N,), device=X_t.device, dtype=torch.long)
    for s in range(0, N, chunk_size):
        e = min(N, s + chunk_size)
        dots = X_t[s:e] @ centers_t.t()
        dists = x_norm[s:e, None] + c_norm[None, :] - 2.0 * dots
        labels[s:e] = torch.argmin(dists, dim=1)
    return labels


def run_kmeans(
    X: np.ndarray,
    n_clusters: int,
    n_init: int = 2,
    max_iter: int = 50,
    tol: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
    device: Optional[torch.device] = None,
    chunk_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(SEED)
    if device is None:
        device = DEVICE

    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,D); got shape={X.shape}")
    N, D = X.shape
    if N == 0:
        return np.zeros((0, D), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    n_clusters = min(int(n_clusters), max(1, N))

    with torch.no_grad():
        X_t = torch.as_tensor(X, device=device, dtype=torch.float32)
        ones = torch.ones((N,), device=device, dtype=torch.float32)

        best_inertia: Optional[float] = None
        best_centers_t: Optional[torch.Tensor] = None
        best_labels_t: Optional[torch.Tensor] = None

        for _ in range(n_init):
            init_idx = np.arange(N) if N <= n_clusters else rng.choice(N, size=n_clusters, replace=False)
            centers_t = X_t.index_select(0, torch.as_tensor(init_idx, device=device, dtype=torch.long)).clone()
            labels_t = torch.zeros((N,), device=device, dtype=torch.long)

            for _it in range(max_iter):
                labels_t = _kmeans_assign_labels_torch(X_t, centers_t, chunk_size=chunk_size)

                sums = torch.zeros_like(centers_t)
                counts = torch.zeros((n_clusters,), device=device, dtype=torch.float32)
                sums.index_add_(0, labels_t, X_t)
                counts.index_add_(0, labels_t, ones)

                new_centers_t = centers_t.clone()
                nonempty = counts > 0
                if nonempty.any():
                    new_centers_t[nonempty] = sums[nonempty] / counts[nonempty].unsqueeze(1)

                empty = ~nonempty
                if empty.any():
                    empty_idx = torch.nonzero(empty, as_tuple=False).squeeze(1).cpu().numpy()
                    reinit = rng.integers(0, N, size=int(empty_idx.shape[0]))
                    new_centers_t[torch.as_tensor(empty_idx, device=device)] = X_t.index_select(
                        0, torch.as_tensor(reinit, device=device, dtype=torch.long)
                    )

                shift = torch.norm(new_centers_t - centers_t) / (torch.norm(centers_t) + 1e-8)
                centers_t = new_centers_t
                if float(shift.cpu()) < tol:
                    break

            inertia = float(((X_t - centers_t[labels_t]) ** 2).sum().cpu())
            if (best_inertia is None) or (inertia < best_inertia):
                best_inertia = inertia
                best_centers_t = centers_t.clone()
                best_labels_t = labels_t.clone()

        return best_centers_t.cpu().numpy(), best_labels_t.cpu().numpy()


def run_weighted_kmeans(
    X: np.ndarray,
    weights: np.ndarray,
    n_clusters: int,
    n_init: int = 2,
    max_iter: int = 50,
    tol: float = 1e-4,
    rng: Optional[np.random.Generator] = None,
    device: Optional[torch.device] = None,
    chunk_size: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(SEED)
    if device is None:
        device = DEVICE

    X = np.asarray(X)
    weights = np.asarray(weights)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (N,D); got shape={X.shape}")
    N, D = X.shape
    if N == 0:
        return np.zeros((0, D), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    n_clusters = min(int(n_clusters), max(1, N))

    w = weights.astype(np.float64, copy=False)
    w_total = float(w.sum())
    if not np.isfinite(w_total) or w_total <= 0:
        w = np.ones_like(w, dtype=np.float64)
        w_total = float(w.sum())

    with torch.no_grad():
        X_t = torch.as_tensor(X, device=device, dtype=torch.float32)
        w_t = torch.as_tensor(w, device=device, dtype=torch.float32)

        best_inertia: Optional[float] = None
        best_centers_t: Optional[torch.Tensor] = None
        best_labels_t: Optional[torch.Tensor] = None

        for _ in range(n_init):
            if N <= n_clusters:
                init_idx = np.arange(N)
            else:
                init_idx = rng.choice(N, size=n_clusters, replace=False, p=w / w_total)
            centers_t = X_t.index_select(0, torch.as_tensor(init_idx, device=device, dtype=torch.long)).clone()
            labels_t = torch.zeros((N,), device=device, dtype=torch.long)

            for _it in range(max_iter):
                labels_t = _kmeans_assign_labels_torch(X_t, centers_t, chunk_size=chunk_size)

                sums_wx = torch.zeros_like(centers_t)
                sums_w = torch.zeros((n_clusters,), device=device)

                sums_wx.index_add_(0, labels_t, X_t * w_t.unsqueeze(1))
                sums_w.index_add_(0, labels_t, w_t)

                new_centers_t = centers_t.clone()
                nonempty = sums_w > 0
                if nonempty.any():
                    new_centers_t[nonempty] = sums_wx[nonempty] / sums_w[nonempty].unsqueeze(1)

                empty = ~nonempty
                if empty.any():
                    empty_idx = torch.nonzero(empty, as_tuple=False).squeeze(1).cpu().numpy()
                    reinit = rng.integers(0, N, size=int(empty_idx.shape[0]))
                    new_centers_t[torch.as_tensor(empty_idx, device=device)] = X_t.index_select(
                        0, torch.as_tensor(reinit, device=device, dtype=torch.long)
                    )

                shift = torch.norm(new_centers_t - centers_t) / (torch.norm(centers_t) + 1e-8)
                centers_t = new_centers_t
                if float(shift.cpu()) < tol:
                    break

            d2 = ((X_t - centers_t[labels_t]) ** 2).sum(dim=1)
            inertia = float((w_t * d2).sum().cpu())
            if (best_inertia is None) or (inertia < best_inertia):
                best_inertia = inertia
                best_centers_t = centers_t.clone()
                best_labels_t = labels_t.clone()

        return best_centers_t.cpu().numpy(), best_labels_t.cpu().numpy()

# ROUTER
class KMeansRouter(nn.Module):
    def __init__(self, centers: np.ndarray, cluster_acc: np.ndarray, cluster_cost: np.ndarray):
        super().__init__()
        centers = torch.tensor(centers, dtype=torch.float32)
        acc = torch.tensor(cluster_acc, dtype=torch.float32)
        cost = torch.tensor(cluster_cost, dtype=torch.float32)
        self.register_buffer("centers", centers)
        self.register_buffer("cluster_acc", acc)
        self.register_buffer("cluster_cost", cost)

    def forward(self, x: torch.Tensor):
        x_exp = x.unsqueeze(1)
        centers = self.centers.unsqueeze(0)
        dists = ((x_exp - centers) ** 2).sum(dim=2)
        idx = torch.argmin(dists, dim=1)
        p_acc = self.cluster_acc[idx]
        c_raw = self.cluster_cost[idx]
        return {"p_acc": p_acc, "c_raw": c_raw}

# BASELINES

def compute_model_baselines(train_ds: CentralDataset) -> Tuple[np.ndarray, np.ndarray]:
    Yacc = train_ds.Yacc.numpy()
    Macc = train_ds.Macc.numpy()
    Yc = train_ds.Yc.numpy()
    Mc = train_ds.Mc.numpy()

    acc_sum = (Yacc * Macc).sum(axis=0)
    acc_count = Macc.sum(axis=0).clip(min=1e-8)
    mean_acc = (acc_sum / acc_count).astype(np.float32)

    cost_sum = (Yc * Mc).sum(axis=0)
    cost_count = Mc.sum(axis=0).clip(min=1e-8)
    mean_cost = (cost_sum / cost_count).astype(np.float32)

    return mean_acc, mean_cost

# FEDERATED K-MEANS WITH STATS

def compute_cluster_sums_counts(
    centers: np.ndarray,
    clients: List[Dict],
    n_models: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_clusters = centers.shape[0]
    acc_sum = np.zeros((n_clusters, n_models), dtype=np.float64)
    acc_count = np.zeros((n_clusters, n_models), dtype=np.float64)
    cost_sum = np.zeros((n_clusters, n_models), dtype=np.float64)
    cost_count = np.zeros((n_clusters, n_models), dtype=np.float64)

    for c in clients:
        X_local = c["train_ds"].X.numpy()
        if X_local.shape[0] == 0:
            continue
        with torch.no_grad():
            X_t = torch.as_tensor(X_local, device=DEVICE, dtype=torch.float32)
            C_t = torch.as_tensor(centers, device=DEVICE, dtype=torch.float32)
            assign = _kmeans_assign_labels_torch(X_t, C_t, chunk_size=None).cpu().numpy()

        Yacc = c["train_ds"].Yacc.numpy()
        Macc = c["train_ds"].Macc.numpy()
        Yc = c["train_ds"].Yc.numpy()
        Mc = c["train_ds"].Mc.numpy()

        for k in range(n_clusters):
            idx = np.where(assign == k)[0]
            if idx.size == 0:
                continue
            Ma = Macc[idx] > 0.5
            Mc_mask = Mc[idx] > 0.5

            if Ma.any():
                vals_acc = np.where(Ma, Yacc[idx], 0.0)
                acc_sum[k] += vals_acc.sum(axis=0)
                acc_count[k] += Ma.sum(axis=0)

            if Mc_mask.any():
                vals_cost = np.where(Mc_mask, Yc[idx], 0.0)
                cost_sum[k] += vals_cost.sum(axis=0)
                cost_count[k] += Mc_mask.sum(axis=0)

    return acc_sum, acc_count, cost_sum, cost_count


def build_federated_kmeans_router_with_stats(
    clients: List[Dict],
    K_local: int,
    K_global: int,
    baseline_acc: np.ndarray,
    baseline_cost: np.ndarray,
) -> Tuple[KMeansRouter, np.ndarray, Dict[str, np.ndarray]]:
    rng = np.random.default_rng(SEED)

    all_centers = []
    all_weights = []
    for c in clients:
        X_local = c["train_ds"].X.numpy()
        if X_local.shape[0] == 0:
            continue
        n_clusters = min(K_local, X_local.shape[0])
        centers_c, labels_c = run_kmeans(
            X_local,
            n_clusters=n_clusters,
            n_init=KMEANS_LOCAL_N_INIT,
            max_iter=KMEANS_LOCAL_MAX_ITER,
            rng=rng,
        )
        counts_c = np.bincount(labels_c, minlength=n_clusters)
        for j in range(n_clusters):
            if counts_c[j] > 0:
                all_centers.append(centers_c[j])
                all_weights.append(counts_c[j])

    if len(all_centers) == 0:
        raise ValueError("No local centers available; check client partitions.")

    all_centers = np.stack(all_centers, axis=0)
    all_weights = np.asarray(all_weights, dtype=np.float64)
    print("Total local centroids fed to server:", all_centers.shape[0])

    K_global_eff = min(K_global, all_centers.shape[0])
    global_centers, _ = run_weighted_kmeans(
        all_centers,
        all_weights,
        n_clusters=K_global_eff,
        n_init=KMEANS_GLOBAL_N_INIT,
        max_iter=KMEANS_GLOBAL_MAX_ITER,
        rng=rng,
    )
    print("Federated global centers shape:", global_centers.shape)

    acc_sum, acc_count, cost_sum, cost_count = compute_cluster_sums_counts(
        global_centers, clients, n_models=baseline_acc.shape[0]
    )

    n_clusters = global_centers.shape[0]
    n_models = baseline_acc.shape[0]

    cluster_acc = np.zeros_like(acc_sum, dtype=np.float32)
    cluster_cost = np.zeros_like(cost_sum, dtype=np.float32)
    for k in range(n_clusters):
        for m in range(n_models):
            if acc_count[k, m] > 0:
                cluster_acc[k, m] = acc_sum[k, m] / acc_count[k, m]
            else:
                cluster_acc[k, m] = FALLBACK_ACC
            if cost_count[k, m] > 0:
                cluster_cost[k, m] = cost_sum[k, m] / cost_count[k, m]
            else:
                cluster_cost[k, m] = FALLBACK_COST

    router = KMeansRouter(global_centers, cluster_acc, cluster_cost).to(DEVICE)
    router.eval()

    stats = {
        "acc_sum": acc_sum,
        "acc_count": acc_count,
        "cost_sum": cost_sum,
        "cost_count": cost_count,
    }
    return router, global_centers, stats


def update_router_with_new_clients(
    centers: np.ndarray,
    base_stats: Dict[str, np.ndarray],
    new_clients: List[Dict],
    baseline_acc: np.ndarray,
    baseline_cost: np.ndarray,
) -> KMeansRouter:
    new_acc_sum, new_acc_count, new_cost_sum, new_cost_count = compute_cluster_sums_counts(
        centers, new_clients, n_models=baseline_acc.shape[0]
    )

    acc_sum = base_stats["acc_sum"] + new_acc_sum
    acc_count = base_stats["acc_count"] + new_acc_count
    cost_sum = base_stats["cost_sum"] + new_cost_sum
    cost_count = base_stats["cost_count"] + new_cost_count

    n_clusters = centers.shape[0]
    n_models = baseline_acc.shape[0]

    cluster_acc = np.zeros_like(acc_sum, dtype=np.float32)
    cluster_cost = np.zeros_like(cost_sum, dtype=np.float32)
    for k in range(n_clusters):
        for m in range(n_models):
            if acc_count[k, m] > 0:
                cluster_acc[k, m] = acc_sum[k, m] / acc_count[k, m]
            else:
                cluster_acc[k, m] = FALLBACK_ACC
            if cost_count[k, m] > 0:
                cluster_cost[k, m] = cost_sum[k, m] / cost_count[k, m]
            else:
                cluster_cost[k, m] = FALLBACK_COST

    router = KMeansRouter(centers, cluster_acc, cluster_cost).to(DEVICE)
    router.eval()
    return router

# EVALUATION
@torch.no_grad()
def predict_heads(model: nn.Module, loader: DataLoader):
    model.eval()
    P_list, Cpred_list = [], []
    Macc_list, Mcost_list = [], []
    Yacc_list, Ycost_list = [], []
    for batch in loader:
        x = batch["x"].to(DEVICE)
        out = model(x)
        P_list.append(out["p_acc"].cpu())
        Cpred_list.append(out["c_raw"].cpu())
        Macc_list.append(batch["m_acc"])
        Mcost_list.append(batch["m_cost"])
        Yacc_list.append(batch["y_acc"])
        Ycost_list.append(batch["y_cost"])
    P = torch.cat(P_list, dim=0)
    Cpred = torch.cat(Cpred_list, dim=0)
    Macc = torch.cat(Macc_list, dim=0)
    Mcost = torch.cat(Mcost_list, dim=0)
    Yacc = torch.cat(Yacc_list, dim=0)
    Ycost = torch.cat(Ycost_list, dim=0)
    return P, Cpred, Macc, Mcost, Yacc, Ycost


def evaluate_curve(model: nn.Module, loader: DataLoader, lambda_grid: np.ndarray):
    P, Cpred, Macc, Mcost, Yacc, Ycost = predict_heads(model, loader)

    points = []
    rows = torch.arange(P.size(0))
    for lam in lambda_grid:
        U = P - float(lam) * Cpred
        choice = torch.argmax(U, dim=1)
        obs = (Macc[rows, choice] > 0.5) & (Mcost[rows, choice] > 0.5)
        if obs.sum() == 0:
            points.append((np.nan, np.nan, 0))
            continue
        yacc = Yacc[rows[obs], choice[obs]]
        ycost = Ycost[rows[obs], choice[obs]]
        points.append((float(ycost.mean()), float(yacc.mean()), int(obs.sum())))

    pts = [(c, a, cv) for (c, a, cv) in points if (c == c) and (a == a)]
    if len(pts) < 2:
        return pts, 0.0
    pts_sorted = sorted(pts, key=lambda t: t[0])
    costs = np.array([t[0] for t in pts_sorted], dtype=np.float64)
    accs = np.array([t[1] for t in pts_sorted], dtype=np.float64)
    area = np.trapz(accs, costs)
    crng = float(costs.max() - costs.min()) if len(costs) > 1 else 0.0
    auc = float(area / (crng + 1e-12)) if crng > 0 else 0.0
    return pts_sorted, auc

# BUILD ROUTERS
base_train_ds = base_global_sets["train_ds"]
if len(base_train_ds) == 0:
    raise ValueError("Base training dataset is empty. Adjust HOLDOUT_TASK_FRAC.")

global_mean_acc, global_mean_cost = compute_model_baselines(base_train_ds)
print("Global baseline acc (first 3 models):", global_mean_acc[:3])
print("Global baseline cost (first 3 models):", global_mean_cost[:3])

fed_router, global_centers, base_stats = build_federated_kmeans_router_with_stats(
    base_clients_spec,
    K_local=K_LOCAL,
    K_global=K_GLOBAL,
    baseline_acc=global_mean_acc,
    baseline_cost=global_mean_cost,
)

base_curve, base_auc = evaluate_curve(fed_router, global_test_loader, LAMBDA_GRID)
print(f"Base router AUC (TEST): {base_auc:.4f}")

adapted_router = update_router_with_new_clients(
    global_centers,
    base_stats=base_stats,
    new_clients=holdout_clients_spec,
    baseline_acc=global_mean_acc,
    baseline_cost=global_mean_cost,
)

adapt_curve, adapt_auc = evaluate_curve(adapted_router, global_test_loader, LAMBDA_GRID)
print(f"Adapted router AUC (TEST): {adapt_auc:.4f}")

# PLOT

def plot_base_vs_adapted(base_curve, base_auc, adapt_curve, adapt_auc, outpath):
    def _clean(pts):
        pts = [(c, a, cv) for (c, a, cv) in pts if (c == c) and (a == a)]
        return sorted(pts, key=lambda t: t[0])

    base_pts = _clean(base_curve)
    adapt_pts = _clean(adapt_curve)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [p[0] for p in base_pts], [p[1] for p in base_pts],
        marker="o", linewidth=2.5,
        label=f"Base (AUC={base_auc:.3f})"
    )
    ax.plot(
        [p[0] for p in adapt_pts], [p[1] for p in adapt_pts],
        marker="o", linestyle="--", linewidth=2.5,
        label=f"Adapted (AUC={adapt_auc:.3f})"
    )
    ax.set_xlabel("Cost")
    ax.set_ylabel("Accuracy")
    ax.set_title("Base vs Holdout-Adapted K-means Router (Global TEST)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)


plot_base_vs_adapted(
    base_curve, base_auc,
    adapt_curve, adapt_auc,
    os.path.join(OUTPUT_DIR, "kmeans_base_vs_adapted_global_test.png"),
)

# SAVE RESULTS

def _curve_to_list(points):
    return [(float(c), float(a), int(cv)) for (c, a, cv) in points]


def _to_serializable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return obj

results = {
    "method": "kmeans_client_expansion",
    "dataset": {
        "num_samples": int(N),
        "num_models": int(K_MODELS),
    },
    "holdout": {
        "n_hold_clients": int(N_HOLD),
        "holdout_client_ids": _to_serializable(holdout_client_ids),
        "holdout_task_frac": float(HOLDOUT_TASK_FRAC),
        "holdout_tasks": sorted(list(holdout_tasks)),
    },
    "global_test": {
        "lambda_grid": _to_serializable(LAMBDA_GRID),
        "base_router": {"curve": _curve_to_list(base_curve), "auc": float(base_auc)},
        "adapted_router": {"curve": _curve_to_list(adapt_curve), "auc": float(adapt_auc)},
    },
}

out_json = os.path.join(OUTPUT_DIR, "kmeans_client_expansion_results.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Saved experiment results to", out_json)
