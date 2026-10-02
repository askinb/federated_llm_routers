import os
if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
import math
import json
import ast
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
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

import sys
sys.path.insert(0, "../data")
from reassemble import ensure_dataset
P = ensure_dataset("proxrouter_train.parquet")
df = pd.read_parquet(P)
print("Loaded shape:", df.shape)
print("First 20 columns:", df.columns.tolist()[:20])

EMB_COL = "all_mpnet_base_v2_embedding"
CHAR_WORD_LEN_FLAG = False

N_CLIENTS = 10

DIRICHLET_ALPHA = 0.6
MODEL_COV_MINMAX = (1, 1)
LABEL_KEEP_FRAC = 0.1
CLIENT_TRAIN_FRAC = 0.75

HIDDEN = (512, 512)
DROPOUT = 0.10
BATCH_SIZE = 128
LR_LOCAL = 3e-4
WEIGHT_DECAY = 1e-4
LOCAL_EPOCHS = 1
MAX_ROUNDS = 7
PARTICIPATION_FRAC = 0.6

N_HOLD = 3  # number of clients joining after base training
HOLDOUT_TASK_FRAC = 0.65  # fraction of unique eval_name tasks reserved for holdout clients
# Adaptation (holdout clients)
MODEL_ADAPT_MAX_ROUNDS = 2
ADAPT_LR = 5e-4
ADAPT_WEIGHT_DECAY = 3e-4
ADAPT_LOCAL_EPOCHS = 1
ADAPT_MSE_WEIGHT = 1.0

# Loss weights
W_ACC = 1.0
W_COST = 1.0

# Lambda sweep for acc-cost curve
LAMBDA_GRID = np.geomspace(1e-2, 1e7, 100)

EPS = 1e-8

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
K = len(MODEL_NAMES)
print(f"Detected {K} models:", MODEL_NAMES)


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
    feat_parts = [e.astype(np.float32)]
    if CHAR_WORD_LEN_FLAG:
        feat_parts.append(np.array([char_len_z[i], word_len_z[i]], dtype=np.float32))
    x = np.concatenate(feat_parts, axis=0)
    X_list.append(x)

X = np.stack(X_list).astype(np.float32)
N = len(df)
print("X shape:", X.shape)

expected_dim = D_emb + (2 if CHAR_WORD_LEN_FLAG else 0)
assert X.shape[1] == expected_dim, f"Expected {expected_dim}, got {X.shape[1]}"

Y_acc = np.full((N, K), np.nan, dtype=np.float32)
M_acc = np.zeros((N, K), dtype=np.float32)
for m_idx, m in enumerate(MODEL_NAMES):
    arr = np.array([_to_acc_float_01_or_nan(v) for v in df[m].tolist()], dtype=np.float32)
    mask = ~np.isnan(arr)
    Y_acc[:, m_idx] = np.where(mask, arr, np.nan)
    M_acc[:, m_idx] = mask.astype(np.float32)

Y_cost = np.full((N, K), np.nan, dtype=np.float32)
M_cost = np.zeros((N, K), dtype=np.float32)
for m_idx, m in enumerate(MODEL_NAMES):
    ccol = m + "|total_cost"
    arr = np.array([_to_float_or_nan(v) for v in df[ccol].tolist()], dtype=np.float32)
    arr = np.where(np.isnan(arr), np.nan, np.maximum(0.0, arr))
    mask = ~np.isnan(arr)
    Y_cost[:, m_idx] = np.where(mask, arr, np.nan)
    M_cost[:, m_idx] = mask.astype(np.float32)

print("Label coverage (acc, cost):", M_acc.mean(), M_cost.mean())


def fit_logz_scaler(y_cost_raw: np.ndarray, m_cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y = y_cost_raw.copy()
    y[np.isnan(y)] = 0.0
    mask = m_cost > 0.5
    y_log = np.zeros_like(y, dtype=np.float64)
    y_log[mask] = np.log1p(y[mask].astype(np.float64))
    mu = np.zeros((K,), dtype=np.float64)
    sigma = np.ones((K,), dtype=np.float64)
    for k in range(K):
        vals = y_log[mask[:, k], k]
        if vals.size == 0:
            mu[k] = 0.0
            sigma[k] = 1.0
        else:
            mu[k] = float(vals.mean())
            s = float(vals.std())
            sigma[k] = s if s > 1e-6 else 1.0
    return mu.astype(np.float32), sigma.astype(np.float32)

# Compute cost normalization scalers from full dataset
mu_log, sigma_log = fit_logz_scaler(Y_cost, M_cost)
print("Cost normalization scalers computed from full dataset")

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


def apply_client_masks(Yacc, Macc, Yc, Mc, model_mask: np.ndarray, label_keep_frac: float, rng=None):
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
    def __init__(self, X, Yacc, Macc, Yc, Mc, mu_log, sigma_log):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Yacc = torch.tensor(np.nan_to_num(Yacc, nan=-1.0), dtype=torch.float32)
        self.Macc = torch.tensor(Macc, dtype=torch.float32)
        self.Yc = torch.tensor(np.nan_to_num(Yc, nan=-1.0), dtype=torch.float32)
        self.Mc = torch.tensor(Mc, dtype=torch.float32)
        self.mu = torch.tensor(mu_log, dtype=torch.float32)
        self.sigma = torch.tensor(sigma_log, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def _normalize_cost_targets(self, y_raw):
        mask = (y_raw >= 0.0)
        y_log = torch.zeros_like(y_raw)
        y_log[mask] = torch.log1p(y_raw[mask])
        y_tilde = torch.zeros_like(y_raw)
        y_tilde[mask] = (y_log[mask] - self.mu[mask]) / (self.sigma[mask] + 1e-8)
        y_tilde = torch.clamp(y_tilde, -3.0, 3.0)
        return y_tilde

    def __getitem__(self, i):
        x = self.X[i]
        y_acc = self.Yacc[i]
        m_acc = self.Macc[i]
        y_cost = self.Yc[i]
        m_cost = self.Mc[i]
        y_cost_tilde = self._normalize_cost_targets(y_cost)
        return {
            "x": x,
            "y_acc": torch.where(m_acc > 0.5, y_acc, torch.zeros_like(y_acc)),
            "m_acc": m_acc,
            "y_cost_tilde": torch.where(m_cost > 0.5, y_cost_tilde, torch.zeros_like(y_cost_tilde)),
            "m_cost": m_cost,
        }


def build_clients_from_partitions(
    client_ids: List[int],
    client_rows_list: List[np.ndarray],
    X_all, Yacc_all, Macc_all, Yc_all, Mc_all,
    mu_log, sigma_log,
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
            K, min_frac=model_cov_minmax[0], max_frac=model_cov_minmax[1], rng=rng
        )

        Yacc_train_c, Macc_train_c, Yc_train_c, Mc_train_c = apply_client_masks(
            Yacc_all[local_train_rows], Macc_all[local_train_rows],
            Yc_all[local_train_rows], Mc_all[local_train_rows],
            model_mask=model_mask, label_keep_frac=label_keep_frac, rng=rng
        )

        # No masking on test
        Yacc_test_c = Yacc_all[local_test_rows].copy()
        Macc_test_c = Macc_all[local_test_rows].copy()
        Yc_test_c = Yc_all[local_test_rows].copy()
        Mc_test_c = Mc_all[local_test_rows].copy()

        train_ds = CentralDataset(
            X_all[local_train_rows], Yacc_train_c, Macc_train_c, Yc_train_c, Mc_train_c,
            mu_log, sigma_log
        )
        test_ds = CentralDataset(
            X_all[local_test_rows], Yacc_test_c, Macc_test_c, Yc_test_c, Mc_test_c,
            mu_log, sigma_log
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

    global_train_ds = CentralDataset(gX, gYacc, gMacc, gYc, gMc, mu_log, sigma_log)
    global_train_loader = DataLoader(
        global_train_ds, batch_size=BATCH_SIZE, shuffle=True,
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

# Split rows into base vs holdout tasks
rows_all = np.arange(N, dtype=int)
rows_holdout_tasks = rows_all[holdout_task_mask]
rows_base_tasks = rows_all[~holdout_task_mask]

# Partition rows to clients (Dirichlet by eval_name) within each group
base_client_rows = partition_rows_by_evalname(
    rows_base_tasks, eval_names_array, n_clients=len(base_client_ids), alpha=DIRICHLET_ALPHA, rng=rng
)
holdout_client_rows = partition_rows_by_evalname(
    rows_holdout_tasks, eval_names_array, n_clients=len(holdout_client_ids), alpha=DIRICHLET_ALPHA, rng=rng
)

# Build base clients
base_clients_spec, base_global_sets = build_clients_from_partitions(
    client_ids=base_client_ids.tolist(),
    client_rows_list=base_client_rows,
    X_all=X,
    Yacc_all=Y_acc,
    Macc_all=M_acc,
    Yc_all=Y_cost,
    Mc_all=M_cost,
    mu_log=mu_log,
    sigma_log=sigma_log,
    client_train_frac=CLIENT_TRAIN_FRAC,
    model_cov_minmax=MODEL_COV_MINMAX,
    label_keep_frac=LABEL_KEEP_FRAC,
    rng=rng,
)

# Build holdout clients
holdout_clients_spec, holdout_global_sets = build_clients_from_partitions(
    client_ids=holdout_client_ids.tolist(),
    client_rows_list=holdout_client_rows,
    X_all=X,
    Yacc_all=Y_acc,
    Macc_all=M_acc,
    Yc_all=Y_cost,
    Mc_all=M_cost,
    mu_log=mu_log,
    sigma_log=sigma_log,
    client_train_frac=CLIENT_TRAIN_FRAC,
    model_cov_minmax=MODEL_COV_MINMAX,
    label_keep_frac=LABEL_KEEP_FRAC,
    rng=rng,
)

# Global test set = union of base + holdout test indices
all_test_indices = np.concatenate([
    base_global_sets["test_indices"],
    holdout_global_sets["test_indices"],
])

global_test_ds = CentralDataset(
    X[all_test_indices],
    Y_acc[all_test_indices], M_acc[all_test_indices],
    Y_cost[all_test_indices], M_cost[all_test_indices],
    mu_log, sigma_log,
)

global_test_loader = DataLoader(
    global_test_ds, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=(DEVICE.type == "cuda")
)

print(f"Base clients: {len(base_clients_spec)}")
print(f"Holdout clients: {len(holdout_clients_spec)}")
print(f"Global TEST size: {len(all_test_indices)}")

# MODEL
class PerModelLogZScaler(nn.Module):
    def __init__(self, mu_log: torch.Tensor, sigma_log: torch.Tensor):
        super().__init__()
        assert mu_log.shape == sigma_log.shape
        self.register_buffer("mu", mu_log.clone().detach())
        self.register_buffer("sigma", sigma_log.clone().detach())

    def denormalize(self, y_tilde: torch.Tensor):
        y_log = y_tilde * (self.sigma + 1e-8) + self.mu
        return torch.expm1(y_log).clamp_min(0.0)


class PerModelHead(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.acc = nn.Linear(d_in, 1)
        self.cost = nn.Linear(d_in, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        acc_logit = self.acc(h)
        cost_tilde = self.cost(h)
        return acc_logit, cost_tilde


class MLPQueryRouter(nn.Module):
    def __init__(self, q_dim: int, n_models: int, hidden=(512, 512), dropout=0.1,
                 scaler: Optional[PerModelLogZScaler] = None):
        super().__init__()
        layers = []
        d = q_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.q_dim = q_dim
        self.n_models = n_models
        self.scaler = scaler

        self.model_heads = nn.ModuleList([
            PerModelHead(d_in=d) for _ in range(n_models)
        ])

    def forward(self, x: torch.Tensor):
        h = self.trunk(x)
        acc_logits_list = []
        cost_tilde_list = []
        for head in self.model_heads:
            acc_logit, cost_tilde_single = head(h)
            acc_logits_list.append(acc_logit)
            cost_tilde_list.append(cost_tilde_single)

        acc_logits = torch.cat(acc_logits_list, dim=1)
        c_tilde = torch.cat(cost_tilde_list, dim=1)
        p_acc = torch.sigmoid(acc_logits)

        out = {"p_acc": p_acc, "c_tilde": c_tilde}
        if self.scaler is not None:
            out["c_raw"] = self.scaler.denormalize(c_tilde)
        return out


# LOSS & EVAL

def compute_masked_losses(out, batch, w_acc=W_ACC, w_cost=W_COST):
    p = out["p_acc"]
    c_til = out["c_tilde"]

    y_acc = batch["y_acc"].to(DEVICE)
    m_acc = batch["m_acc"].to(DEVICE)
    y_ctil = batch["y_cost_tilde"].to(DEVICE)
    m_cost = batch["m_cost"].to(DEVICE)

    L_acc = ((p - y_acc).pow(2) * m_acc).sum() / (m_acc.sum() + 1e-8)
    L_cost = (F.smooth_l1_loss(c_til, y_ctil, reduction="none") * m_cost).sum() / (m_cost.sum() + 1e-8)
    return {
        "L_total": w_acc * L_acc + w_cost * L_cost,
        "L_acc": L_acc,
        "L_cost": L_cost,
    }


@torch.no_grad()
def predict_heads(model: MLPQueryRouter, loader: DataLoader):
    model.eval()
    P_list, Craw_list, Macc_list, Mcost_list, Yacc_list, Ycost_tilde_list = [], [], [], [], [], []
    for batch in loader:
        x = batch["x"].to(DEVICE)
        out = model.forward(x)
        assert "c_raw" in out, "Cost must be raw for evaluation."
        P_list.append(out["p_acc"].cpu())
        Craw_list.append(out["c_raw"].cpu())
        Macc_list.append(batch["m_acc"])
        Mcost_list.append(batch["m_cost"])
        Yacc_list.append(batch["y_acc"])
        Ycost_tilde_list.append(batch["y_cost_tilde"])
    P = torch.cat(P_list, dim=0)
    C = torch.cat(Craw_list, dim=0)
    Macc = torch.cat(Macc_list, dim=0)
    Mcost = torch.cat(Mcost_list, dim=0)
    Yacc = torch.cat(Yacc_list, dim=0)
    Ycost_til = torch.cat(Ycost_tilde_list, dim=0)

    if hasattr(model, "scaler") and (model.scaler is not None):
        mu = model.scaler.mu.cpu()
        sigma = model.scaler.sigma.cpu()
        Y_log = Ycost_til * (sigma + 1e-8) + mu
        Ycost_raw = torch.expm1(Y_log).clamp_min(0.0)
    else:
        Ycost_raw = Ycost_til
    return P, C, Macc, Mcost, Yacc, Ycost_raw


def evaluate_curve(model: MLPQueryRouter, loader: DataLoader, lambda_grid: np.ndarray):
    P, Cpred, Macc, Mcost, Yacc, Ycost_raw = predict_heads(model, loader)
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
        ycost = Ycost_raw[rows[obs], choice[obs]]
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

# FEDERATED TRAINING
class Client:
    def __init__(self, cid: int, train_loader: DataLoader, test_loader: DataLoader):
        self.cid = cid
        self.train_loader = train_loader
        self.test_loader = test_loader

    def local_train_from(
        self,
        base_model: nn.Module,
        epochs=1,
        lr=2e-4,
        weight_decay=1e-4,
        ref_model: Optional[nn.Module] = None,
        distill_weight: float = 0.0,
    ):
        model = MLPQueryRouter(
            q_dim=base_model.trunk[0].in_features,
            n_models=base_model.n_models,
            hidden=HIDDEN,
            dropout=DROPOUT,
            scaler=base_model.scaler,
        ).to(DEVICE)
        model.load_state_dict(base_model.state_dict(), strict=True)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()
        if ref_model is not None:
            ref_model.eval()

        for _ in range(epochs):
            for batch in self.train_loader:
                x = batch["x"].to(DEVICE)
                out = model.forward(x)
                losses = compute_masked_losses(out, batch)
                total_loss = losses["L_total"]

                if ref_model is not None and distill_weight > 0.0:
                    with torch.no_grad():
                        ref_out = ref_model.forward(x)
                    mse = F.mse_loss(out["p_acc"], ref_out["p_acc"]) + \
                          F.mse_loss(out["c_tilde"], ref_out["c_tilde"])
                    total_loss = total_loss + distill_weight * mse

                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        delta = {k: (model.state_dict()[k] - base_model.state_dict()[k]).detach().cpu()
                 for k in model.state_dict()}
        weight = len(self.train_loader.dataset)
        return model, delta, weight


class FedAvgServer:
    def __init__(self, global_model: MLPQueryRouter):
        self.model = global_model
        self.history = {
            "round": [],
            "global_val_auc": [],
        }

    def aggregate(self, deltas: List[Dict[str, torch.Tensor]], weights: List[int]):
        if not deltas:
            return
        total = float(sum(weights))
        avg_delta = {}
        for k in deltas[0].keys():
            avg_delta[k] = sum(deltas[i][k] * (weights[i] / total) for i in range(len(deltas)))
        with torch.no_grad():
            state = self.model.state_dict()
            for k in state:
                state[k].add_(avg_delta[k].to(state[k].device))
            self.model.load_state_dict(state, strict=True)


def federated_train(
    server: FedAvgServer,
    clients: List[Client],
    global_test_loader: DataLoader,
    rounds: int,
    participate_frac: float,
    local_epochs: int,
    lr_local: float,
    weight_decay: float,
    lambda_grid: np.ndarray,
    ref_model: Optional[nn.Module] = None,
    distill_weight: float = 0.0,
):
    rng = np.random.default_rng(SEED)

    for r in range(1, rounds + 1):
        m = max(1, int(round(participate_frac * len(clients))))
        chosen = rng.choice(len(clients), size=m, replace=False)
        deltas, weights = [], []

        for idx in chosen:
            _, d, w = clients[idx].local_train_from(
                server.model,
                epochs=local_epochs,
                lr=lr_local,
                weight_decay=weight_decay,
                ref_model=ref_model,
                distill_weight=distill_weight,
            )
            deltas.append(d)
            weights.append(w)

        server.aggregate(deltas, weights)

        _, g_auc = evaluate_curve(server.model, global_test_loader, lambda_grid)
        server.history["round"].append(r)
        server.history["global_val_auc"].append(g_auc)
        print(f"[Round {r:02d}] Global AUC (TEST): {g_auc:.4f}")

    return server

# BUILD CLIENT OBJECTS
base_clients = [
    Client(cid=spec["cid"], train_loader=spec["train_loader"], test_loader=spec["test_loader"])
    for spec in base_clients_spec
]

holdout_clients = [
    Client(cid=spec["cid"], train_loader=spec["train_loader"], test_loader=spec["test_loader"])
    for spec in holdout_clients_spec
]

# BASE FEDERATED TRAINING
q_dim = X.shape[1]
scaler = PerModelLogZScaler(torch.tensor(mu_log), torch.tensor(sigma_log))
base_model = MLPQueryRouter(
    q_dim=q_dim,
    n_models=K,
    hidden=HIDDEN,
    dropout=DROPOUT,
    scaler=scaler,
).to(DEVICE)

server = FedAvgServer(base_model)
server = federated_train(
    server,
    base_clients,
    global_test_loader,
    rounds=MAX_ROUNDS,
    participate_frac=PARTICIPATION_FRAC,
    local_epochs=LOCAL_EPOCHS,
    lr_local=LR_LOCAL,
    weight_decay=WEIGHT_DECAY,
    lambda_grid=LAMBDA_GRID,
)

# Base evaluation
base_curve, base_auc = evaluate_curve(server.model, global_test_loader, LAMBDA_GRID)
print(f"Base model AUC (TEST): {base_auc:.4f}")

# Snapshot base history before adaptation
base_history = json.loads(json.dumps(server.history))

# Freeze base snapshot for adaptation regularization
base_ref_model = MLPQueryRouter(
    q_dim=q_dim,
    n_models=K,
    hidden=HIDDEN,
    dropout=DROPOUT,
    scaler=scaler,
).to(DEVICE)
base_ref_model.load_state_dict(server.model.state_dict(), strict=True)

# ADAPTATION WITH HOLDOUT CLIENTS
if holdout_clients:
    # Reset history for adaptation phase
    server.history = {"round": [], "global_val_auc": []}
    server = federated_train(
        server,
        holdout_clients,
        global_test_loader,
        rounds=MODEL_ADAPT_MAX_ROUNDS,
        participate_frac=1.0,
        local_epochs=ADAPT_LOCAL_EPOCHS,
        lr_local=ADAPT_LR,
        weight_decay=ADAPT_WEIGHT_DECAY,
        lambda_grid=LAMBDA_GRID,
        ref_model=base_ref_model,
        distill_weight=ADAPT_MSE_WEIGHT,
    )

adapt_curve, adapt_auc = evaluate_curve(server.model, global_test_loader, LAMBDA_GRID)
print(f"Adapted model AUC (TEST): {adapt_auc:.4f}")

adapt_history = json.loads(json.dumps(server.history))

# PLOTS

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
    ax.set_title("Base vs Holdout-Adapted Router (Global TEST)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)


plot_base_vs_adapted(
    base_curve, base_auc,
    adapt_curve, adapt_auc,
    os.path.join(OUTPUT_DIR, "mlp_base_vs_adapted_global_test.png"),
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
    "method": "mlp_client_expansion",
    "dataset": {
        "num_samples": int(N),
        "num_models": int(K),
    },
    "holdout": {
        "n_hold_clients": int(N_HOLD),
        "holdout_client_ids": _to_serializable(holdout_client_ids),
        "holdout_task_frac": float(HOLDOUT_TASK_FRAC),
        "holdout_tasks": sorted(list(holdout_tasks)),
    },
    "base_training": {
        "rounds": int(MAX_ROUNDS),
        "history": _to_serializable(base_history),
    },
    "adaptation": {
        "rounds": int(MODEL_ADAPT_MAX_ROUNDS),
        "distill_weight": float(ADAPT_MSE_WEIGHT),
        "history": _to_serializable(adapt_history),
    },
    "global_test": {
        "lambda_grid": _to_serializable(LAMBDA_GRID),
        "base_model": {"curve": _curve_to_list(base_curve), "auc": float(base_auc)},
        "adapted_model": {"curve": _curve_to_list(adapt_curve), "auc": float(adapt_auc)},
    },
}

out_json = os.path.join(OUTPUT_DIR, "mlp_client_expansion_results.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Saved experiment results to", out_json)
