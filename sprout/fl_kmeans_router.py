import os
import sys

# Change working directory to script location so relative paths work
if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

OUTPUT_DIR = "./kmeans_out"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================
# Load & prune
# =========================================
import pandas as pd

import sys
sys.path.insert(0, "../data")
from reassemble import ensure_dataset
P = ensure_dataset("sprout_w4emb.parquet")
df = pd.read_parquet(P)
print("Loaded shape:", df.shape)
print("First 20 columns:", df.columns.tolist()[:20])

# Drop model_response columns (not used)
drop_cols = [c for c in df.columns if c.endswith("|model_response")]
if drop_cols:
    df = df.drop(columns=drop_cols)
    print(f"Dropped {len(drop_cols)} `|model_response` columns; new shape:", df.shape)

# =========================================
# Config & imports
# =========================================
import math, json, ast, random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Reproducibility
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

# Embedding column (fixed globally for all clients)
EMB_COL = "all_mpnet_base_v2_embedding"

CHAR_WORD_LEN_FLAG = False   # embedding-only
TSNE_FLAG = False

# Client/global splits
CLIENT_TRAIN_FRAC, CLIENT_TEST_FRAC = 0.75, 0.25

# Router hyperparams
K_LOCAL = 15        # number of local clusters per client
K_GLOBAL = 20       # number of global clusters at server
CENTRALIZED_K = 30  # number of clusters for centralized training

# K-means configuration
KMEANS_LOCAL_N_INIT = 3
KMEANS_GLOBAL_N_INIT = 3
KMEANS_LOCAL_MAX_ITER = 30
KMEANS_GLOBAL_MAX_ITER = 30

# Batch size for evaluation loaders (no training)
BATCH_SIZE = 256

# Lambda sweep for acc-cost curves
LAMBDA_GRID = np.geomspace(1e-2, 1e7, 100)

# Client generation knobs
N_CLIENTS = 10
DIRICHLET_ALPHA = 0.6          # task non-IID
MODEL_COV_MINMAX = (1, 1)      # fraction of models available per client
LABEL_KEEP_FRAC = 0.1          # per-entry label keep probability

EPS = 1e-8

# Fallback values for cluster-model pairs with no samples
FALLBACK_ACC = 0.0      # Zero accuracy when no samples available
FALLBACK_COST = 1e6     # Large cost when no samples available

PLOT_CLIENT_IDS = [i for i in range(N_CLIENTS)]

MODEL_DIRICHLET_ALPHA = 0.45

# =========================================
# Detect models & parsers
# =========================================
# Detect model names via presence of '|total_cost'
cost_cols = [c for c in df.columns if c.endswith("|total_cost")]
candidate_models = [c[:-len("|total_cost")] for c in cost_cols]
MODEL_NAMES = sorted([m for m in candidate_models if m in df.columns])
K_MODELS = len(MODEL_NAMES)
print(f"Detected {K_MODELS} models:", MODEL_NAMES)

def _to_float_or_nan(x):
    if x is None: return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)): return float(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"nan","none","null"}: return np.nan
        try: return float(s)
        except:
            try: obj = json.loads(s)
            except:
                try: obj = ast.literal_eval(s)
                except: obj = None
            if isinstance(obj, dict):
                for key in ["total_cost","cost","price"]:
                    if key in obj:
                        try: return float(obj[key])
                        except: pass
            return np.nan
    if isinstance(x, dict):
        for key in ["total_cost","cost","price"]:
            if key in x:
                try: return float(x[key])
                except: pass
    return np.nan

def _to_acc_float_01_or_nan(x):
    if x is None: return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        if isinstance(x, float) and math.isnan(x): return np.nan
        return float(min(1.0, max(0.0, x)))
    if isinstance(x, str):
        s = x.strip()
        if s == "" or s.lower() in {"nan","none","null"}: return np.nan
        try:
            v = float(s); return float(min(1.0, max(0.0, v)))
        except:
            try: obj = json.loads(s)
            except:
                try: obj = ast.literal_eval(s)
                except: obj = None
            if isinstance(obj, dict):
                for key in ["is_correct","correct","accuracy","acc","score","p"]:
                    if key in obj:
                        return _to_acc_float_01_or_nan(obj[key])
            return np.nan
    if isinstance(x, dict):
        for key in ["is_correct","correct","accuracy","acc","score","p"]:
            if key in x:
                return _to_acc_float_01_or_nan(x[key])
    return np.nan

# =========================================
# Build features & labels/masks
# =========================================
def prompt_to_text(v):
    if v is None: return ""
    if isinstance(v, str): return v
    if isinstance(v, (list, tuple)): return " ".join([str(x) for x in v])
    return str(v)

def parse_embedding_cell(v):
    if v is None: return None
    if isinstance(v, (list, tuple, np.ndarray)): return np.asarray(v, dtype=np.float32)
    if isinstance(v, str):
        try: obj = json.loads(v)
        except:
            try: obj = ast.literal_eval(v)
            except: obj = None
        if isinstance(obj, (list, tuple, np.ndarray)):
            return np.asarray(obj, dtype=np.float32)
    return None

def zscore(a):
    m, s = a.mean(), a.std()
    s = s if s > 1e-6 else 1.0
    return (a - m) / s

char_len_z = None
word_len_z = None

eval_names = df["eval_name"].fillna("unknown").astype(str)

emb_list = [parse_embedding_cell(v) for v in df[EMB_COL].tolist()]
D_emb = next((len(e) for e in emb_list if e is not None), None)
assert D_emb is not None, f"No valid vectors in {EMB_COL}"
print("Embedding dim:", D_emb)

X_list = []
for i, e in enumerate(emb_list):
    if e is None:
        e = np.zeros(D_emb, dtype=np.float32)
    parts = [e.astype(np.float32)]
    x = np.concatenate(parts, axis=0)
    X_list.append(x)

X = np.stack(X_list).astype(np.float32)
N = len(df)
print("X shape:", X.shape)

assert X.shape[1] == D_emb, f"Expected feature dim {D_emb}, got {X.shape[1]}"

# Acc labels / masks
Y_acc = np.full((N, K_MODELS), np.nan, dtype=np.float32)
M_acc = np.zeros((N, K_MODELS), dtype=np.float32)
for m_idx, m in enumerate(MODEL_NAMES):
    arr = np.array([_to_acc_float_01_or_nan(v) for v in df[m].tolist()], dtype=np.float32)
    mask = ~np.isnan(arr)
    Y_acc[:, m_idx] = np.where(mask, arr, np.nan)
    M_acc[:, m_idx] = mask.astype(np.float32)

# Cost labels / masks (raw cost, no log-normalization)
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

eval_names_array = eval_names.to_numpy()
print("Full dataset ready for client partitioning:", X.shape)

# =========================================
# Dataset class & global loaders
# =========================================
class CentralDataset(Dataset):
    """
    Stores features + (possibly sparse) labels for accuracy and cost.
    K-Means version: stores raw cost (y_cost), no log-z normalization.
    We never train here; only use labels for evaluating routers.
    """
    def __init__(self, X, Yacc, Macc, Yc, Mc):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Yacc = torch.tensor(np.nan_to_num(Yacc, nan=-1.0), dtype=torch.float32)
        self.Macc = torch.tensor(Macc, dtype=torch.float32)
        self.Yc   = torch.tensor(np.nan_to_num(Yc,   nan=-1.0), dtype=torch.float32)
        self.Mc   = torch.tensor(Mc, dtype=torch.float32)

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

# =========================================
# Client partitioning helpers
# =========================================
def partition_by_evalname_indices_new(eval_names: np.ndarray, n_clients: int, alpha: float = 0.5, rng=None):
    """
    For each category, sample a Dirichlet distribution over clients.
    This ensures each category is fully distributed across clients according to the sampled proportions.
    """
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

def sample_model_subset_per_client(K, min_frac=0.6, max_frac=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    frac = rng.uniform(min_frac, max_frac)
    k = max(1, int(round(frac * K)))
    chosen = rng.choice(K, size=k, replace=False)
    mask = np.zeros(K, dtype=bool); mask[chosen] = True
    return mask

def generate_client_model_probabilities(n_models: int, dirichlet_alpha: Optional[float] = None, rng=None):
    """
    Generate probabilities for model selection per client using Dirichlet distribution.
    If dirichlet_alpha is None, returns uniform probabilities.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    if dirichlet_alpha is None:
        return np.ones(n_models, dtype=np.float32) / n_models
    else:
        return rng.dirichlet(dirichlet_alpha * np.ones(n_models)).astype(np.float32)

def apply_client_masks(Yacc, Macc, Yc, Mc, model_mask: np.ndarray, label_keep_frac: float,
                      client_model_probs: Optional[np.ndarray] = None, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    Yacc_c, Macc_c = Yacc.copy(), Macc.copy()
    Yc_c,   Mc_c   = Yc.copy(),   Mc.copy()
    not_kept = ~model_mask
    Yacc_c[:, not_kept] = np.nan; Macc_c[:, not_kept] = 0.0
    Yc_c[:,   not_kept] = np.nan; Mc_c[:,   not_kept] = 0.0
    if label_keep_frac < 1.0:
        n_rows, n_models = Macc_c.shape
        if n_models > 0:
            k_keep = max(1, int(math.floor(label_keep_frac * n_models)))
            keep_mask = np.zeros((n_rows, n_models), dtype=bool)
            for i in range(n_rows):
                if client_model_probs is not None:
                    chosen = rng.choice(n_models, size=k_keep, replace=False, p=client_model_probs)
                else:
                    chosen = rng.choice(n_models, size=k_keep, replace=False)
                keep_mask[i, chosen] = True
            drop_mask = ~keep_mask
            Yacc_c[drop_mask] = np.nan; Macc_c[drop_mask] = 0.0
            Yc_c[drop_mask]   = np.nan; Mc_c[drop_mask]   = 0.0
    return Yacc_c, Macc_c, Yc_c, Mc_c

def make_clients_with_local_splits_single_pass(
    X_all, Yacc_all, Macc_all, Yc_all, Mc_all,
    eval_names_all: np.ndarray,
    n_clients: int = 10,
    dirichlet_alpha: float = 0.5,
    model_cov_minmax=(0.6, 1.0),
    label_keep_frac: float = 0.7,
    client_train_frac: float = 0.75,
):
    """
    Single-pass client creation:

    1. Partition full dataset across clients using per-category Dirichlet logic
    2. Within each client, split into train/test
    3. Apply model availability mask + label sparsification (TRAIN ONLY)
    4. Build per-client datasets/loaders
    5. Build global train dataset/loaders as union of *client-train views* (masked/sparsified)
       Build global test dataset/loaders as union of client test splits (full label coverage)
    """
    rng = np.random.default_rng(SEED)

    client_row_idxs = partition_by_evalname_indices_new(
        eval_names_all, n_clients, dirichlet_alpha, rng=rng
    )

    clients_spec = []
    collected_train_idxs = []
    collected_test_idxs = []

    # collect the *masked* client-train arrays for centralized training
    gX_parts, gYacc_parts, gMacc_parts, gYc_parts, gMc_parts = [], [], [], [], []

    for c, rows in enumerate(client_row_idxs):
        if len(rows) == 0:
            continue

        rows = np.array(rows, dtype=int)
        rng.shuffle(rows)

        n_local_train = int(client_train_frac * len(rows))
        local_train_rows = rows[:n_local_train]
        local_test_rows  = rows[n_local_train:]

        if len(local_train_rows) == 0:
            collected_test_idxs.extend(local_test_rows.tolist())
            continue

        collected_train_idxs.extend(local_train_rows.tolist())
        collected_test_idxs.extend(local_test_rows.tolist())

        model_mask = sample_model_subset_per_client(
            K_MODELS, min_frac=model_cov_minmax[0], max_frac=model_cov_minmax[1], rng=rng
        )

        # Generate client-specific model probabilities for Dirichlet sampling
        client_probs = generate_client_model_probabilities(K_MODELS, MODEL_DIRICHLET_ALPHA, rng=rng)

        print(f"Client {c} model probabilities:", client_probs)

        # apply masks + label sparsification on training part
        Yacc_train_c, Macc_train_c, Yc_train_c, Mc_train_c = apply_client_masks(
            Yacc_all[local_train_rows], Macc_all[local_train_rows],
            Yc_all[local_train_rows],   Mc_all[local_train_rows],
            model_mask=model_mask, label_keep_frac=label_keep_frac,
            client_model_probs=client_probs, rng=rng
        )

        # NO masking on test: keep full label coverage for evaluation
        Yacc_test_c = Yacc_all[local_test_rows].copy()
        Macc_test_c = Macc_all[local_test_rows].copy()
        Yc_test_c   = Yc_all[local_test_rows].copy()
        Mc_test_c   = Mc_all[local_test_rows].copy()

        train_ds = CentralDataset(
            X_all[local_train_rows], Yacc_train_c, Macc_train_c, Yc_train_c, Mc_train_c
        )
        test_ds = CentralDataset(
            X_all[local_test_rows],  Yacc_test_c,  Macc_test_c,  Yc_test_c,  Mc_test_c
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
            "cid": c,
            "train_idx": local_train_rows,
            "test_idx": local_test_rows,
            "model_mask": model_mask,
            "train_ds": train_ds,
            "test_ds": test_ds,
            "train_loader": train_loader,
            "test_loader": test_loader,
        })

        # add this client's *masked train view* into centralized/global train
        gX_parts.append(X_all[local_train_rows])
        gYacc_parts.append(Yacc_train_c)
        gMacc_parts.append(Macc_train_c)
        gYc_parts.append(Yc_train_c)
        gMc_parts.append(Mc_train_c)

    collected_train_idxs = np.array(collected_train_idxs, dtype=int)
    collected_test_idxs  = np.array(collected_test_idxs, dtype=int)

    # global train == union of client-train views (masked/sparsified)
    if len(gX_parts) > 0:
        gX    = np.concatenate(gX_parts, axis=0)
        gYacc = np.concatenate(gYacc_parts, axis=0)
        gMacc = np.concatenate(gMacc_parts, axis=0)
        gYc   = np.concatenate(gYc_parts, axis=0)
        gMc   = np.concatenate(gMc_parts, axis=0)
    else:
        gX, gYacc, gMacc, gYc, gMc = (
            X_all[:0], Yacc_all[:0], Macc_all[:0], Yc_all[:0], Mc_all[:0]
        )

    global_train_ds = CentralDataset(gX, gYacc, gMacc, gYc, gMc)

    # global test stays full coverage (all models available at test)
    global_test_ds = CentralDataset(
        X_all[collected_test_idxs],
        Yacc_all[collected_test_idxs], Macc_all[collected_test_idxs],
        Yc_all[collected_test_idxs],   Mc_all[collected_test_idxs]
    )

    global_train_loader = DataLoader(
        global_train_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == "cuda")
    )
    global_test_loader = DataLoader(
        global_test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == "cuda")
    )

    return clients_spec, {
        "train_loader": global_train_loader,
        "test_loader": global_test_loader,
        "train_ds": global_train_ds,
        "test_ds": global_test_ds,
        "train_indices": collected_train_idxs,
        "test_indices": collected_test_idxs,
    }


# =========================================
# Create clients and global loaders
# =========================================
clients_spec, global_sets = make_clients_with_local_splits_single_pass(
    X, Y_acc, M_acc, Y_cost, M_cost,
    eval_names_all=eval_names_array,
    n_clients=N_CLIENTS,
    dirichlet_alpha=DIRICHLET_ALPHA,
    model_cov_minmax=MODEL_COV_MINMAX,
    label_keep_frac=LABEL_KEEP_FRAC,
    client_train_frac=CLIENT_TRAIN_FRAC
)

print(f"Created {len(clients_spec)} clients")
print(f"Global training set size: {len(global_sets['train_indices'])}")
print(f"Global test set size: {len(global_sets['test_indices'])}")

global_train_loader = global_sets["train_loader"]
global_test_loader = global_sets["test_loader"]
global_train_ds = global_sets["train_ds"]
global_test_ds = global_sets["test_ds"]

# Client wrapper
class Client:
    def __init__(self, cid: int, train_ds: CentralDataset, test_ds: CentralDataset,
                 train_loader: DataLoader, test_loader: DataLoader):
        self.cid = cid
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.train_loader = train_loader
        self.test_loader = test_loader

clients: List[Client] = []
for spec in clients_spec:
    clients.append(Client(
        cid=spec["cid"],
        train_ds=spec["train_ds"],
        test_ds=spec["test_ds"],
        train_loader=spec["train_loader"],
        test_loader=spec["test_loader"],
    ))

# =========================================
# Baseline model-wise stats (for cluster fallbacks)
# =========================================
def compute_model_baselines(train_ds: CentralDataset) -> Tuple[np.ndarray, np.ndarray]:
    Yacc = train_ds.Yacc.numpy()
    Macc = train_ds.Macc.numpy()
    Yc   = train_ds.Yc.numpy()
    Mc   = train_ds.Mc.numpy()

    acc_sum = (Yacc * Macc).sum(axis=0)
    acc_count = Macc.sum(axis=0).clip(min=1e-8)
    mean_acc = (acc_sum / acc_count).astype(np.float32)

    cost_sum = (Yc * Mc).sum(axis=0)
    cost_count = Mc.sum(axis=0).clip(min=1e-8)
    mean_cost = (cost_sum / cost_count).astype(np.float32)

    return mean_acc, mean_cost

global_mean_acc, global_mean_cost = compute_model_baselines(global_train_ds)
print("Global baseline acc (first 3 models):", global_mean_acc[:3])
print("Global baseline cost (first 3 models):", global_mean_cost[:3])

# =========================================
# K-means utilities (Torch backend)
# =========================================
def _kmeans_assign_labels_torch(
    X_t: torch.Tensor,
    centers_t: torch.Tensor,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Assign each row in X_t to its nearest center (squared L2). Returns (N,) long tensor."""
    N = X_t.shape[0]
    if N == 0:
        return torch.empty((0,), device=X_t.device, dtype=torch.long)

    # d(x,c)^2 = ||x||^2 + ||c||^2 - 2 x*c
    x_norm = (X_t * X_t).sum(dim=1)          # (N,)
    c_norm = (centers_t * centers_t).sum(dim=1)  # (K,)

    if chunk_size is None or chunk_size >= N:
        dots = X_t @ centers_t.t()           # (N,K)
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
    use_torch: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Torch-backed k-means (CUDA/MPS/CPU). Returns NumPy centers (K,D) and labels (N,).
    """
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

    if not use_torch:
        best_inertia = None
        best_centers = None
        best_labels = None
        for _ in range(n_init):
            init_idx = np.arange(N) if N <= n_clusters else rng.choice(N, size=n_clusters, replace=False)
            centers = X[init_idx].copy()
            labels = np.zeros(N, dtype=np.int64)
            for _it in range(max_iter):
                dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                labels = dists.argmin(axis=1)
                new_centers = np.zeros_like(centers)
                for k in range(n_clusters):
                    mask = (labels == k)
                    new_centers[k] = X[mask].mean(axis=0) if mask.any() else X[rng.integers(0, N)]
                shift = np.linalg.norm(new_centers - centers) / (np.linalg.norm(centers) + 1e-8)
                centers = new_centers
                if shift < tol:
                    break
            inertia = float(((X - centers[labels]) ** 2).sum())
            if (best_inertia is None) or (inertia < best_inertia):
                best_inertia, best_centers, best_labels = inertia, centers.copy(), labels.copy()
        return best_centers, best_labels

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

                sums = torch.zeros_like(centers_t)  # (K,D)
                counts = torch.zeros((n_clusters,), device=device, dtype=torch.float32)  # (K,)
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
    use_torch: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Torch-backed weighted k-means. Returns NumPy centers (K,D) and labels (N,).
    """
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

    if not use_torch:
        best_inertia = None
        best_centers = None
        best_labels = None
        for _ in range(n_init):
            if N <= n_clusters:
                init_idx = np.arange(N)
            else:
                init_idx = rng.choice(N, size=n_clusters, replace=False, p=w / w_total)
            centers = X[init_idx].copy()
            labels = np.zeros(N, dtype=np.int64)
            for _it in range(max_iter):
                dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
                labels = dists.argmin(axis=1)
                new_centers = np.zeros_like(centers)
                for k in range(n_clusters):
                    mask = (labels == k)
                    if mask.any():
                        ww = w[mask][:, None]
                        new_centers[k] = (X[mask] * ww).sum(axis=0) / ww.sum()
                    else:
                        new_centers[k] = X[rng.integers(0, N)]
                shift = np.linalg.norm(new_centers - centers) / (np.linalg.norm(centers) + 1e-8)
                centers = new_centers
                if shift < tol:
                    break
            d2 = ((X - centers[labels]) ** 2).sum(axis=1)
            inertia = float((w * d2).sum())
            if (best_inertia is None) or (inertia < best_inertia):
                best_inertia, best_centers, best_labels = inertia, centers.copy(), labels.copy()
        return best_centers, best_labels

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

                sums_wx = torch.zeros_like(centers_t)               # (K,D)
                sums_w = torch.zeros((n_clusters,), device=device)  # (K,)

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


# =========================================
# Router definitions
# =========================================
class KMeansRouter(nn.Module):
    """
    Router based on K-means clusters:
      - cluster_centers: (Kc, D)
      - cluster_acc: (Kc, K_MODELS)  average accuracy per model
      - cluster_cost: (Kc, K_MODELS) average cost per model
    For a query x, we route to nearest center and use its stats.
    """
    def __init__(self, centers: np.ndarray, cluster_acc: np.ndarray, cluster_cost: np.ndarray):
        super().__init__()
        centers = torch.tensor(centers, dtype=torch.float32)
        acc = torch.tensor(cluster_acc, dtype=torch.float32)
        cost = torch.tensor(cluster_cost, dtype=torch.float32)
        self.register_buffer("centers", centers)
        self.register_buffer("cluster_acc", acc)
        self.register_buffer("cluster_cost", cost)

    def forward(self, x: torch.Tensor):
        # x: (B, D)
        x_exp = x.unsqueeze(1)              # (B, 1, D)
        centers = self.centers.unsqueeze(0) # (1, Kc, D)
        dists = ((x_exp - centers) ** 2).sum(dim=2)  # (B, Kc)
        idx = torch.argmin(dists, dim=1)    # (B,)
        p_acc = self.cluster_acc[idx]       # (B, K_MODELS)
        c_raw = self.cluster_cost[idx]      # (B, K_MODELS)
        return {"p_acc": p_acc, "c_raw": c_raw}


# =========================================
# Build routers: federated, centralized, local-only
# =========================================
def build_federated_kmeans_router(
    clients: List[Client],
    K_local: int,
    K_global: int,
    baseline_acc: np.ndarray,
    baseline_cost: np.ndarray,
) -> KMeansRouter:
    """
    Federated K-means:

    1. Each client runs local K-means (K_local) on its local TRAIN features.
       It returns local centroids and cluster sizes.
    2. Server runs weighted K-means on these centroids (weights = cluster sizes)
       to obtain K_global global centroids.
    3. Server sends global centroids to clients. Each client assigns its local
       training examples to global centroids and computes per-cluster per-model
       acc and cost averages. These statistics are sent back.
    4. Server aggregates (weighted by counts) to obtain global cluster stats.
    """
    rng = np.random.default_rng(SEED)

    # Step 1: collect local centroids + weights
    all_centers = []
    all_weights = []
    for c in clients:
        X_local = c.train_ds.X.numpy()
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

    all_centers = np.stack(all_centers, axis=0)
    all_weights = np.asarray(all_weights, dtype=np.float64)
    print("Total local centroids fed to server:", all_centers.shape[0])

    # Step 2: global weighted K-means
    K_global_eff = min(K_global, all_centers.shape[0])
    global_centers, _ = run_weighted_kmeans(
        all_centers, all_weights,
        n_clusters=K_global_eff,
        n_init=KMEANS_GLOBAL_N_INIT,
        max_iter=KMEANS_GLOBAL_MAX_ITER,
        rng=rng,
    )
    print("Federated global centers shape:", global_centers.shape)

    # Steps 3 & 4: compute global per-cluster per-model stats
    n_clusters = global_centers.shape[0]
    n_models = baseline_acc.shape[0]

    cluster_acc_sum = np.zeros((n_clusters, n_models), dtype=np.float64)
    cluster_acc_count = np.zeros((n_clusters, n_models), dtype=np.float64)
    cluster_cost_sum = np.zeros((n_clusters, n_models), dtype=np.float64)
    cluster_cost_count = np.zeros((n_clusters, n_models), dtype=np.float64)

    for c in clients:
        X_local = c.train_ds.X.numpy()
        if X_local.shape[0] == 0:
            continue

        # assign to global centers
        with torch.no_grad():
            X_t = torch.as_tensor(X_local, device=DEVICE, dtype=torch.float32)
            C_t = torch.as_tensor(global_centers, device=DEVICE, dtype=torch.float32)
            assign = _kmeans_assign_labels_torch(X_t, C_t, chunk_size=None).cpu().numpy()

        Yacc = c.train_ds.Yacc.numpy()
        Macc = c.train_ds.Macc.numpy()
        Yc   = c.train_ds.Yc.numpy()
        Mc   = c.train_ds.Mc.numpy()

        for k in range(n_clusters):
            idx = np.where(assign == k)[0]
            if idx.size == 0:
                continue

            Ma = Macc[idx] > 0.5
            Mc_mask = Mc[idx] > 0.5

            if Ma.any():
                vals_acc = np.where(Ma, Yacc[idx], 0.0)
                sum_acc = vals_acc.sum(axis=0)
                count_acc = Ma.sum(axis=0)
                cluster_acc_sum[k] += sum_acc
                cluster_acc_count[k] += count_acc

            if Mc_mask.any():
                vals_cost = np.where(Mc_mask, Yc[idx], 0.0)
                sum_cost = vals_cost.sum(axis=0)
                count_cost = Mc_mask.sum(axis=0)
                cluster_cost_sum[k] += sum_cost
                cluster_cost_count[k] += count_cost

    cluster_acc = np.zeros_like(cluster_acc_sum, dtype=np.float32)
    cluster_cost = np.zeros_like(cluster_cost_sum, dtype=np.float32)
    for k in range(n_clusters):
        for m in range(n_models):
            if cluster_acc_count[k, m] > 0:
                cluster_acc[k, m] = cluster_acc_sum[k, m] / cluster_acc_count[k, m]
            else:
                cluster_acc[k, m] = FALLBACK_ACC
            if cluster_cost_count[k, m] > 0:
                cluster_cost[k, m] = cluster_cost_sum[k, m] / cluster_cost_count[k, m]
            else:
                cluster_cost[k, m] = FALLBACK_COST

    fed_router = KMeansRouter(global_centers, cluster_acc, cluster_cost).to(DEVICE)
    fed_router.eval()
    return fed_router

def build_centralized_kmeans_router(
    train_ds: CentralDataset,
    K_clusters: int,
    baseline_acc: np.ndarray,
    baseline_cost: np.ndarray,
) -> KMeansRouter:
    """
    Centralized K-means router trained on union of all client TRAIN data.
    """
    X_train = train_ds.X.numpy()
    centers, labels = run_kmeans(
        X_train,
        n_clusters=min(K_clusters, X_train.shape[0]),
        n_init=KMEANS_GLOBAL_N_INIT,
        max_iter=KMEANS_GLOBAL_MAX_ITER,
    )
    n_clusters = centers.shape[0]
    n_models = baseline_acc.shape[0]

    cluster_acc_sum = np.zeros((n_clusters, n_models), dtype=np.float64)
    cluster_acc_count = np.zeros((n_clusters, n_models), dtype=np.float64)
    cluster_cost_sum = np.zeros((n_clusters, n_models), dtype=np.float64)
    cluster_cost_count = np.zeros((n_clusters, n_models), dtype=np.float64)

    Yacc = train_ds.Yacc.numpy()
    Macc = train_ds.Macc.numpy()
    Yc   = train_ds.Yc.numpy()
    Mc   = train_ds.Mc.numpy()

    for k in range(n_clusters):
        idx = np.where(labels == k)[0]
        if idx.size == 0:
            continue
        Ma = Macc[idx] > 0.5
        Mc_mask = Mc[idx] > 0.5

        if Ma.any():
            vals_acc = np.where(Ma, Yacc[idx], 0.0)
            sum_acc = vals_acc.sum(axis=0)
            count_acc = Ma.sum(axis=0)
            cluster_acc_sum[k] += sum_acc
            cluster_acc_count[k] += count_acc

        if Mc_mask.any():
            vals_cost = np.where(Mc_mask, Yc[idx], 0.0)
            sum_cost = vals_cost.sum(axis=0)
            count_cost = Mc_mask.sum(axis=0)
            cluster_cost_sum[k] += sum_cost
            cluster_cost_count[k] += count_cost

    cluster_acc = np.zeros_like(cluster_acc_sum, dtype=np.float32)
    cluster_cost = np.zeros_like(cluster_cost_sum, dtype=np.float32)
    for k in range(n_clusters):
        for m in range(n_models):
            if cluster_acc_count[k, m] > 0:
                cluster_acc[k, m] = cluster_acc_sum[k, m] / cluster_acc_count[k, m]
            else:
                cluster_acc[k, m] = FALLBACK_ACC
            if cluster_cost_count[k, m] > 0:
                cluster_cost[k, m] = cluster_cost_sum[k, m] / cluster_cost_count[k, m]
            else:
                cluster_cost[k, m] = FALLBACK_COST

    cent_router = KMeansRouter(centers, cluster_acc, cluster_cost).to(DEVICE)
    cent_router.eval()
    return cent_router

def build_local_only_kmeans_routers(
    clients: List[Client],
    K_local: int,
    baseline_acc: np.ndarray,
    baseline_cost: np.ndarray,
) -> Dict[int, KMeansRouter]:
    """
    Local-only routers: each client runs K-means on its own data only.
    """
    local_routers: Dict[int, KMeansRouter] = {}
    for c in clients:
        X_local = c.train_ds.X.numpy()
        if X_local.shape[0] == 0:
            continue
        centers, labels = run_kmeans(
            X_local,
            n_clusters=min(K_local, X_local.shape[0]),
            n_init=KMEANS_LOCAL_N_INIT,
            max_iter=KMEANS_LOCAL_MAX_ITER,
        )
        n_clusters = centers.shape[0]
        n_models = baseline_acc.shape[0]

        cluster_acc_sum = np.zeros((n_clusters, n_models), dtype=np.float64)
        cluster_acc_count = np.zeros((n_clusters, n_models), dtype=np.float64)
        cluster_cost_sum = np.zeros((n_clusters, n_models), dtype=np.float64)
        cluster_cost_count = np.zeros((n_clusters, n_models), dtype=np.float64)

        Yacc = c.train_ds.Yacc.numpy()
        Macc = c.train_ds.Macc.numpy()
        Yc   = c.train_ds.Yc.numpy()
        Mc   = c.train_ds.Mc.numpy()

        for k in range(n_clusters):
            idx = np.where(labels == k)[0]
            if idx.size == 0:
                continue
            Ma = Macc[idx] > 0.5
            Mc_mask = Mc[idx] > 0.5

            if Ma.any():
                vals_acc = np.where(Ma, Yacc[idx], 0.0)
                sum_acc = vals_acc.sum(axis=0)
                count_acc = Ma.sum(axis=0)
                cluster_acc_sum[k] += sum_acc
                cluster_acc_count[k] += count_acc

            if Mc_mask.any():
                vals_cost = np.where(Mc_mask, Yc[idx], 0.0)
                sum_cost = vals_cost.sum(axis=0)
                count_cost = Mc_mask.sum(axis=0)
                cluster_cost_sum[k] += sum_cost
                cluster_cost_count[k] += count_cost

        cluster_acc = np.zeros_like(cluster_acc_sum, dtype=np.float32)
        cluster_cost = np.zeros_like(cluster_cost_sum, dtype=np.float32)
        for k in range(n_clusters):
            for m in range(n_models):
                if cluster_acc_count[k, m] > 0:
                    cluster_acc[k, m] = cluster_acc_sum[k, m] / cluster_acc_count[k, m]
                else:
                    cluster_acc[k, m] = FALLBACK_ACC
                if cluster_cost_count[k, m] > 0:
                    cluster_cost[k, m] = cluster_cost_sum[k, m] / cluster_cost_count[k, m]
                else:
                    cluster_cost[k, m] = FALLBACK_COST

        router_c = KMeansRouter(centers, cluster_acc, cluster_cost).to(DEVICE)
        router_c.eval()
        local_routers[c.cid] = router_c

    return local_routers


# =========================================
# Evaluation utilities (curves & AUC)
# =========================================
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
    """
    For each lambda:
      choose model m* = argmax (pred_acc - lambda * pred_cost)
      realize (acc, cost) from ground-truth on chosen arms (where observed)
      average -> (cost, acc, coverage)
    Returns (sorted_points_by_cost, normalized_auc).
    """
    P, Cpred, Macc, Mcost, Yacc, Ycost = predict_heads(model, loader)

    points = []
    rows = torch.arange(P.size(0))
    for lam in lambda_grid:
        U = P - float(lam) * Cpred
        choice = torch.argmax(U, dim=1)
        obs = (Macc[rows, choice] > 0.5) & (Mcost[rows, choice] > 0.5)
        if obs.sum() == 0:
            points.append((np.nan, np.nan, 0)); continue
        yacc  = Yacc[rows[obs], choice[obs]]
        ycost = Ycost[rows[obs], choice[obs]]
        points.append((float(ycost.mean()), float(yacc.mean()), int(obs.sum())))

    pts = [(c,a,cv) for (c,a,cv) in points if (c==c) and (a==a)]
    if len(pts) < 2:
        return pts, 0.0
    pts_sorted = sorted(pts, key=lambda t: t[0])
    costs = np.array([t[0] for t in pts_sorted], dtype=np.float64)
    accs  = np.array([t[1] for t in pts_sorted], dtype=np.float64)
    area = np.trapz(accs, costs)
    crng = float(costs.max() - costs.min()) if len(costs) > 1 else 0.0
    auc = float(area / (crng + 1e-12)) if crng > 0 else 0.0
    return pts_sorted, auc


# =========================================
# Build routers
# =========================================
print("\n=== Building routers ===")
fed_router = build_federated_kmeans_router(
    clients, K_local=K_LOCAL, K_global=K_GLOBAL,
    baseline_acc=global_mean_acc, baseline_cost=global_mean_cost
)
cent_router = build_centralized_kmeans_router(
    global_train_ds, K_clusters=CENTRALIZED_K,
    baseline_acc=global_mean_acc, baseline_cost=global_mean_cost
)
local_only_routers = build_local_only_kmeans_routers(
    clients, K_local=8,
    baseline_acc=global_mean_acc, baseline_cost=global_mean_cost
)

print(f"Built federated router, centralized router, and "
      f"{len(local_only_routers)} local-only routers.")


# =========================================
# Build federated history (single-round, for JSON compatibility)
# =========================================
def build_single_round_history(
    global_router: KMeansRouter,
    clients: List[Client],
    global_test_loader: DataLoader,
) -> Dict[str, list]:
    """
    For compatibility with previous plotting code: create a
    'history' with a single 'round' (0).
    """
    _, g_auc = evaluate_curve(global_router, global_test_loader, LAMBDA_GRID)
    per_client_auc = {}
    per_client_curve = {}
    for c in clients:
        c_curve, c_auc = evaluate_curve(global_router, c.test_loader, LAMBDA_GRID)
        per_client_auc[c.cid] = c_auc
        per_client_curve[c.cid] = c_curve
    history = {
        "round": [0],
        "global_val_auc": [g_auc],
        "per_client_val_auc_on_global": [per_client_auc],
        "per_client_curve_on_global": [per_client_curve],
    }
    return history

router_history = build_single_round_history(fed_router, clients, global_test_loader)
print(f"Federated global AUC: {router_history['global_val_auc'][0]:.4f}")


# =========================================
# Save experimental results to JSON
# =========================================
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

def _eval_curve(model, loader):
    curve, auc = evaluate_curve(model, loader, LAMBDA_GRID)
    return {"curve": _curve_to_list(curve), "auc": float(auc)}

client_ids_all = sorted(local_only_routers.keys())

per_client_local = {}
for cid in client_ids_all:
    local_loader = clients[cid].test_loader
    per_client_local[str(cid)] = {
        "global": _eval_curve(fed_router, local_loader),
        "local_only": _eval_curve(local_only_routers[cid], local_loader),
    }

per_client_on_global = {}
for cid in client_ids_all:
    per_client_on_global[str(cid)] = {
        "local_only": _eval_curve(local_only_routers[cid], global_test_loader),
    }

# Compute model-wise active sample counts for every client (based on train set mask)
per_client_model_active_samples = {}
for cid in client_ids_all:
    train_ds = clients[cid].train_ds
    if hasattr(train_ds, "Macc"):
        mask = train_ds.Macc
    else:
        mask = train_ds[1]
    if isinstance(mask, torch.Tensor):
        sample_counts = mask.detach().cpu().numpy().sum(axis=0).astype(int).tolist()
    else:
        sample_counts = np.array(mask).sum(axis=0).astype(int).tolist()
    per_client_model_active_samples[str(cid)] = sample_counts

results = {
    "method": "kmeans",
    "lambda_grid": _to_serializable(LAMBDA_GRID),
    "dataset": {
        "num_samples": int(N),
        "num_models": int(K_MODELS),
    },
    "federated": {
        "history": _to_serializable(router_history),
    },
    "evaluations": {
        "global_test": {
            "global_model": _eval_curve(fed_router, global_test_loader),
            "centralized_model": _eval_curve(cent_router, global_test_loader),
        },
        "per_client_local_test": per_client_local,
        "per_client_on_global_test": per_client_on_global,
    },
    "per_client_model_active_samples": per_client_model_active_samples,
    "model_names": MODEL_NAMES,
}

out_json = os.path.join(OUTPUT_DIR, "kmeans_experiment_results.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Saved experiment results to", out_json)
print("\nAll K-means experiments finished. Results in:", OUTPUT_DIR)
