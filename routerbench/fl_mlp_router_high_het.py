
import os
if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = "./mlp_out_high_het"
os.makedirs(OUTPUT_DIR, exist_ok=True)

import pandas as pd


import sys
sys.path.insert(0, "../data")
from reassemble import ensure_dataset
P = ensure_dataset("routerbench_0shot_w4emb.parquet")
df = pd.read_parquet(P)
print("Loaded shape:", df.shape)
print("First 20 columns:", df.columns.tolist()[:20])

drop_cols = [c for c in df.columns if c.endswith("|model_response")]
if drop_cols:
    df = df.drop(columns=drop_cols)
    print(f"Dropped {len(drop_cols)} `|model_response` columns; new shape:", df.shape)




import os, math, json, ast, random
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

SEED = 100
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

TSNE_FLAG = False
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
DEVICE = torch.device(DEVICE)
print("Using device:", DEVICE)

EMB_COL = "all_mpnet_base_v2_embedding"

CHAR_WORD_LEN_FLAG = False

TRAIN_FRAC, TEST_FRAC = 0.70, 0.30

# Local client splits (within each client's allocated data)
CLIENT_TRAIN_FRAC, CLIENT_TEST_FRAC = 0.75, 0.25

HIDDEN = (512, 512)
DROPOUT = 0.10
BATCH_SIZE = 128
LR_LOCAL = 1e-3
WEIGHT_DECAY = 3e-4
LOCAL_EPOCHS = 1            # per client per FL round (local steps)
MAX_ROUNDS = 5             # FL communication rounds
PARTICIPATION_FRAC = 0.6

CENTRALIZED_LR = 1e-5
CENTRALIZED_WD = 1e-4
CENTRALIZED_EPOCHS = 1

# Loss weights
W_ACC = 1.0
W_COST = 1.0

LAMBDA_GRID = np.geomspace(1e-2, 1e7, 100)

N_CLIENTS = 10
DIRICHLET_ALPHA = 0.03         # task non-IID
MODEL_COV_MINMAX = (1,1)  # fraction of models available per client
LABEL_KEEP_FRAC = 0.1          # per-entry label keep probability

EPS = 1e-8

PLOT_CLIENT_IDS = [i for i in range(N_CLIENTS)]

MODEL_DIRICHLET_ALPHA = 1


# Detect model names via presence of '|total_cost'
cost_cols = [c for c in df.columns if c.endswith("|total_cost")]
candidate_models = [c[:-len("|total_cost")] for c in cost_cols]
MODEL_NAMES = sorted([m for m in candidate_models if m in df.columns])
K = len(MODEL_NAMES)
print(f"Detected {K} models:", MODEL_NAMES)

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
eval_names_array = eval_names.to_numpy()   # keep this for partitioning + t-SNE labeling


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
        vals = y_log[mask[:,k], k]
        if vals.size == 0:
            mu[k] = 0.0; sigma[k] = 1.0
        else:
            mu[k] = float(vals.mean())
            s = float(vals.std())
            sigma[k] = s if s > 1e-6 else 1.0
    return mu.astype(np.float32), sigma.astype(np.float32)

# Compute cost normalization scalers from full dataset
mu_log, sigma_log = fit_logz_scaler(Y_cost, M_cost)
print("Cost normalization scalers computed from full dataset")
print("Scaler sample (mu, sigma):", mu_log[:3], sigma_log[:3])


# No central split needed - will partition full dataset directly across clients
eval_names_array = eval_names.to_numpy()
print("Full dataset ready for client partitioning:", X.shape)


class CentralDataset(Dataset):
    def __init__(self, X, Yacc, Macc, Yc, Mc, mu_log, sigma_log):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Yacc = torch.tensor(np.nan_to_num(Yacc, nan=-1.0), dtype=torch.float32)
        self.Macc = torch.tensor(Macc, dtype=torch.float32)
        self.Yc   = torch.tensor(np.nan_to_num(Yc,   nan=-1.0), dtype=torch.float32)
        self.Mc   = torch.tensor(Mc, dtype=torch.float32)
        self.mu = torch.tensor(mu_log, dtype=torch.float32)
        self.sigma = torch.tensor(sigma_log, dtype=torch.float32)

    def __len__(self): return self.X.shape[0]

    def _normalize_cost_targets(self, y_raw):
        mask = (y_raw >= 0.0)
        y_log = torch.zeros_like(y_raw); y_log[mask] = torch.log1p(y_raw[mask])
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
            "m_cost": m_cost
        }

# Central datasets removed - using collected global sets from client creation


def partition_by_evalname_indices_new(eval_names: np.ndarray, n_clients: int, alpha: float = 0.5, rng=None):
    """
    New partitioning logic: For each category, sample a Dirichlet distribution over clients.
    This ensures each category is fully distributed across clients according to the sampled proportions.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    cats, inv = np.unique(eval_names, return_inverse=True)
    G = len(cats)
    
    # Initialize client index lists
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
                    # Use Dirichlet probabilities for sampling
                    chosen = rng.choice(n_models, size=k_keep, replace=False, p=client_model_probs)
                else:
                    # Uniform sampling (original behavior)
                    chosen = rng.choice(n_models, size=k_keep, replace=False)
                keep_mask[i, chosen] = True
            drop_mask = ~keep_mask
            Yacc_c[drop_mask] = np.nan; Macc_c[drop_mask] = 0.0
            Yc_c[drop_mask]   = np.nan; Mc_c[drop_mask]   = 0.0
    return Yacc_c, Macc_c, Yc_c, Mc_c

def make_clients_with_local_splits_single_pass(
    X_all, Yacc_all, Macc_all, Yc_all, Mc_all,
    eval_names_all: np.ndarray,
    mu_log: np.ndarray, sigma_log: np.ndarray,
    n_clients: int = 10,
    dirichlet_alpha: float = 0.5,
    model_cov_minmax=(0.6, 1.0),
    label_keep_frac: float = 0.7,
    client_train_frac: float = 0.75,
):
    """
    Single-pass client creation:

    1) Partition full dataset across clients using per-category Dirichlet logic
    2) Within each client, split into train/test
    3) Apply model availability mask + label sparsification (TRAIN ONLY)
    4) Build per-client datasets/loaders
    5) Build global train dataset/loaders as union of *client-train views* (masked/sparsified)
       Build global test dataset/loaders as union of client test splits (full label coverage)
    """
    rng = np.random.default_rng(SEED)

    client_row_idxs = partition_by_evalname_indices_new(
        eval_names_all, n_clients, dirichlet_alpha, rng=rng
    )

    clients_spec = []
    collected_train_idxs = []
    collected_test_idxs = []

    # collect the *masked* client-train arrays for global/centralized training
    gX_parts, gYacc_parts, gMacc_parts, gYc_parts, gMc_parts = [], [], [], [], []

    for c, rows in enumerate(client_row_idxs):
        if len(rows) == 0:
            continue

        rows = np.array(rows, dtype=int)
        rng.shuffle(rows)

        n_local_train = int(client_train_frac * len(rows))
        local_train_rows = rows[:n_local_train]
        local_test_rows  = rows[n_local_train:]

        # if no train data, keep test indices but skip train contributions
        if len(local_train_rows) == 0:
            if len(local_test_rows) > 0:
                collected_test_idxs.extend(local_test_rows.tolist())
            continue

        collected_train_idxs.extend(local_train_rows.tolist())
        if len(local_test_rows) > 0:
            collected_test_idxs.extend(local_test_rows.tolist())

        # Sample model subset for this client
        model_mask = sample_model_subset_per_client(
            K, min_frac=model_cov_minmax[0], max_frac=model_cov_minmax[1], rng=rng
        )

        # Generate client-specific model probabilities for Dirichlet sampling
        client_probs = generate_client_model_probabilities(K, MODEL_DIRICHLET_ALPHA, rng=rng)

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
            X_all[local_train_rows], Yacc_train_c, Macc_train_c, Yc_train_c, Mc_train_c,
            mu_log, sigma_log
        )
        test_ds = CentralDataset(
            X_all[local_test_rows],  Yacc_test_c,  Macc_test_c,  Yc_test_c,  Mc_test_c,
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
            "cid": c,
            "train_idx": local_train_rows,
            "test_idx": local_test_rows,
            "model_mask": model_mask,
            "train_ds": train_ds,
            "test_ds": test_ds,
            "train_loader": train_loader,
            "test_loader": test_loader,
        })

        # add this client’s *masked train view* into global_train
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

    global_train_ds = CentralDataset(gX, gYacc, gMacc, gYc, gMc, mu_log, sigma_log)

    # global test stays full coverage (all models available at test)
    global_test_ds = CentralDataset(
        X_all[collected_test_idxs],
        Yacc_all[collected_test_idxs], Macc_all[collected_test_idxs],
        Yc_all[collected_test_idxs],   Mc_all[collected_test_idxs],
        mu_log, sigma_log
    )

    global_train_loader = DataLoader(
        global_train_ds, batch_size=BATCH_SIZE, shuffle=True,
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


# Single-pass client creation using full dataset and pre-computed scalers
clients_spec, global_sets = make_clients_with_local_splits_single_pass(
    X, Y_acc, M_acc, Y_cost, M_cost,
    eval_names_all=eval_names_array,
    mu_log=mu_log, sigma_log=sigma_log,
    n_clients=N_CLIENTS,
    dirichlet_alpha=DIRICHLET_ALPHA,
    model_cov_minmax=MODEL_COV_MINMAX,
    label_keep_frac=LABEL_KEEP_FRAC,
    client_train_frac=CLIENT_TRAIN_FRAC
)

print(f"Created {len(clients_spec)} clients in single pass")
print(f"Global training set size: {len(global_sets['train_indices'])}")
print(f"Global test set size: {len(global_sets['test_indices'])}")

# Extract loaders for convenience
global_train_loader = global_sets["train_loader"]
global_test_loader = global_sets["test_loader"]

if TSNE_FLAG:
    from sklearn.manifold import TSNE
    import matplotlib

    # ---------------------------------------------------------
    # 1) Choose points for t-SNE: ONLY global TRAIN indices
    # ---------------------------------------------------------
    all_train_idx = np.array(global_sets["train_indices"], dtype=int)
    N_train = len(all_train_idx)

    tsne_frac = 1   # set to 1.0 to use ALL train points
    rng = np.random.default_rng(SEED)

    n_tsne = int(round(N_train * tsne_frac))
    tsne_indices = rng.choice(all_train_idx, size=n_tsne, replace=False)
    tsne_indices = np.sort(tsne_indices)

    print(f"Running t-SNE on {len(tsne_indices)} TRAIN samples out of {N_train} total TRAIN...")

    X_subset = X[tsne_indices]
    eval_names_subset = eval_names_array[tsne_indices]

    tsne = TSNE(
        n_components=2,
        random_state=SEED,
        perplexity=30,
        init="random",
        learning_rate="auto",
    )
    X_tsne = tsne.fit_transform(X_subset)   # [N', 2]
    print("t-SNE finished. Shape:", X_tsne.shape)

    # ---------------------------------------------------------
    # 2) Task keys for coloring (merge MMLU + Chinese)
    # ---------------------------------------------------------
    task_keys = []
    for name in eval_names_subset:
        lname = str(name).lower()
        if "mmlu" in lname:
            task_keys.append("MMLU (all)")
        elif "chinese" in lname:
            task_keys.append("Chinese (all)")
        else:
            task_keys.append(str(name))
    task_keys = np.array(task_keys)

    # Put merged groups first in legend, then the rest (stable)
    merged_first = ["MMLU (all)", "Chinese (all)"]
    others = sorted([k for k in set(task_keys.tolist()) if k not in merged_first])
    unique_keys = [k for k in merged_first if k in set(task_keys.tolist())] + others

    key_to_color = {k: i for i, k in enumerate(unique_keys)}
    num_colors = len(unique_keys)

    # Matplotlib 3.7+ friendly colormap (no deprecation warning)
    base = "tab20" if num_colors <= 20 else "gist_ncar"
    cmap = matplotlib.colormaps[base].resampled(num_colors)

    # Map global row index -> position in tsne_indices (for per-client plots)
    idx_map = {idx: j for j, idx in enumerate(tsne_indices)}

    # ---------------------------------------------------------
    # 3) Plot grid
    # ---------------------------------------------------------
    max_pts_per_client = 1000

    fig, axes = plt.subplots(4, 3, figsize=(12, 12), sharex=False, sharey=False)
    axes = axes.flatten()

    # --- Subplot 0: ALL t-SNE TRAIN points (the ones used to compute t-SNE) ---
    ax0 = axes[0]
    for k in unique_keys:
        kid = key_to_color[k]
        mask = (task_keys == k)
        if not np.any(mask):
            continue
        ax0.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            s=8,
            color=cmap(kid),
            alpha=0.7,
            label=k,
        )

    ax0.set_title("All clients (TRAIN used for t-SNE)")
    ax0.set_xticks([]); ax0.set_yticks([])

    # --- Subplots 1..: each client's TRAIN points (restricted to the t-SNE subset) ---
    for subplot_idx, spec in enumerate(clients_spec, start=1):
        if subplot_idx >= len(axes):
            break

        ax = axes[subplot_idx]
        cid = spec["cid"]

        client_train_idx = np.array(spec["train_idx"], dtype=int)
        client_train_idx = np.intersect1d(tsne_indices, client_train_idx, assume_unique=False)

        if client_train_idx.size == 0:
            ax.set_title(f"Client {cid} (no TRAIN in t-SNE subset)")
            ax.set_xticks([]); ax.set_yticks([])
            continue

        if client_train_idx.size > max_pts_per_client:
            client_train_idx = rng.choice(client_train_idx, size=max_pts_per_client, replace=False)

        plot_client_pos = np.array([idx_map[i] for i in client_train_idx], dtype=int)

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

        ax.set_title(f"Client {cid} (TRAIN)")
        ax.set_xticks([]); ax.set_yticks([])

    # Hide unused axes if fewer clients than grid slots
    for j in range(1 + len(clients_spec), len(axes)):
        fig.delaxes(axes[j])

    # ---------------------------------------------------------
    # 4) Suptitle + 6-column legend at top (between title and plots)
    # ---------------------------------------------------------
    fig.suptitle("t-SNE on ALL clients' TRAIN data (subsampled)\nColors: per-task, with MMLU/Chinese merged", y=0.995)

    handles, labels = ax0.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=6,
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, 0.955),
    )

    # Leave space at top for title + legend
    plt.tight_layout(rect=(0, 0, 1, 0.90))

    outpath = os.path.join(OUTPUT_DIR, "tsne_clients_4x3.png")
    plt.savefig(outpath, dpi=200)
    plt.close(fig)

    print("Saved t-SNE plot to", outpath)
    tsne_payload = {
        "tsne_indices": tsne_indices.tolist(),
        "X_tsne": X_tsne.tolist(),
        "task_keys": task_keys.tolist(),
        "unique_keys": unique_keys,
        "clients": [
            {"cid": int(spec["cid"]), "train_idx": np.asarray(spec["train_idx"], dtype=int).tolist()}
            for spec in clients_spec
        ],
    }
    tsne_out = os.path.join(OUTPUT_DIR, "tsne.json")
    with open(tsne_out, "w", encoding="utf-8") as f:
        json.dump(tsne_payload, f, indent=2)
    print("Saved t-SNE stats to", tsne_out)


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
    """
    One head per LLM:
      - acc:   scalar logit for accuracy (will go through sigmoid)
      - cost:  scalar normalized cost (c_tilde), no activation
    """
    def __init__(self, d_in: int):
        super().__init__()
        self.acc = nn.Linear(d_in, 1)
        self.cost = nn.Linear(d_in, 1)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h: [B, d_in]
        acc_logit = self.acc(h)    # [B, 1]
        cost_tilde = self.cost(h)  # [B, 1]
        return acc_logit, cost_tilde

class MLPQueryRouter(nn.Module):
    def __init__(self, q_dim: int, n_models: int, hidden=(512,512), dropout=0.1,
                 scaler: Optional[PerModelLogZScaler]=None):
        super().__init__()

        # ----- shared trunk (unchanged) -----
        layers = []
        d = q_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)

        # remember basic config so other code can query it
        self.q_dim = q_dim
        self.n_models = n_models
        self.scaler = scaler

        # ----- NEW: per-model heads -----
        # One PerModelHead per LLM, each outputs (acc_logit, cost_tilde)
        self.model_heads = nn.ModuleList([
            PerModelHead(d_in=d) for _ in range(n_models)
        ])

    def forward(self, x: torch.Tensor):
        # shared representation
        h = self.trunk(x)  # [B, d]

        acc_logits_list = []
        cost_tilde_list = []

        # run all model-specific heads
        for head in self.model_heads:
            acc_logit, cost_tilte_single = head(h)          # each [B, 1]
            acc_logits_list.append(acc_logit)
            cost_tilde_list.append(cost_tilte_single)

        # stack into [B, K]
        acc_logits = torch.cat(acc_logits_list, dim=1)      # [B, K]
        c_tilde   = torch.cat(cost_tilde_list, dim=1)       # [B, K]

        # same as before: sigmoid on accuracy, cost is linear
        p_acc = torch.sigmoid(acc_logits)

        out = {"p_acc": p_acc, "c_tilde": c_tilde}
        if self.scaler is not None:
            out["c_raw"] = self.scaler.denormalize(c_tilde)
        return out

    def masked_losses(self, batch, w_acc=W_ACC, w_cost=W_COST):
        x = batch["x"].to(DEVICE)
        out = self.forward(x)
        p = out["p_acc"]        # [B, K]
        c_til = out["c_tilde"]  # [B, K]

        y_acc  = batch["y_acc"].to(DEVICE)
        m_acc  = batch["m_acc"].to(DEVICE)
        y_ctil = batch["y_cost_tilde"].to(DEVICE)
        m_cost = batch["m_cost"].to(DEVICE)

        # MSE for accuracy probability (unchanged)
        L_acc = ((p - y_acc).pow(2) * m_acc).sum() / (m_acc.sum() + 1e-8)
        # Smooth-L1 for normalized cost (unchanged)
        L_cost = (F.smooth_l1_loss(c_til, y_ctil, reduction="none") * m_cost).sum() / (m_cost.sum() + 1e-8)
        return {"L_total": w_acc*L_acc + w_cost*L_cost, "L_acc": L_acc, "L_cost": L_cost}

class WeightedEnsembleRouter(nn.Module):
    def __init__(self, global_model, local_model, w_global=0.5, w_local=0.5):
        super().__init__()
        assert abs(w_global + w_local - 1.0) < 1e-6
        self.global_model = global_model
        self.local_model = local_model
        self.w_global = w_global
        self.w_local = w_local
        self.scaler = global_model.scaler  # assume shared

    def forward(self, x: torch.Tensor):
        out_g = self.global_model(x)
        out_l = self.local_model(x)

        # Accuracy: average probabilities (already what you want)
        p_acc = self.w_global * out_g["p_acc"] + self.w_local * out_l["p_acc"]

        # Cost: average RAW cost (after denormalization)
        if self.scaler is None:
            raise ValueError("Ensemble expects scaler so both routers output c_raw.")
        c_raw_g = out_g["c_raw"]
        c_raw_l = out_l["c_raw"]
        c_raw = self.w_global * c_raw_g + self.w_local * c_raw_l

        # (Optional) also expose c_tilde consistent with this raw cost
        # so downstream code that expects c_tilde won't break.
        y_log = torch.log1p(c_raw.clamp_min(0.0))
        c_tilde = (y_log - self.scaler.mu) / (self.scaler.sigma + 1e-8)
        c_tilde = torch.clamp(c_tilde, -3.0, 3.0)

        return {"p_acc": p_acc, "c_raw": c_raw, "c_tilde": c_tilde}


class AdaptiveEnsembleRouter(nn.Module):
    def __init__(
        self,
        global_model: MLPQueryRouter,
        local_model: MLPQueryRouter,
        w_global_acc: torch.Tensor,
        w_local_acc: torch.Tensor,
        w_global_cost: torch.Tensor,
        w_local_cost: torch.Tensor,
    ):
        super().__init__()
        self.global_model = global_model
        self.local_model = local_model
        self.register_buffer("w_global_acc", w_global_acc.float())
        self.register_buffer("w_local_acc", w_local_acc.float())
        self.register_buffer("w_global_cost", w_global_cost.float())
        self.register_buffer("w_local_cost", w_local_cost.float())
        self.scaler = global_model.scaler

    def forward(self, x: torch.Tensor):
        out_g = self.global_model(x)
        out_l = self.local_model(x)

        w_g_acc = self.w_global_acc.view(1, -1)
        w_l_acc = self.w_local_acc.view(1, -1)
        p_acc = w_g_acc * out_g["p_acc"] + w_l_acc * out_l["p_acc"]

        if self.scaler is None:
            raise ValueError("AdaptiveEnsemble expects scaler so routers output c_raw.")
        c_raw_g = out_g["c_raw"] if "c_raw" in out_g else self.scaler.denormalize(out_g["c_tilde"])
        c_raw_l = out_l["c_raw"] if "c_raw" in out_l else self.scaler.denormalize(out_l["c_tilde"])
        w_g_cost = self.w_global_cost.view(1, -1)
        w_l_cost = self.w_local_cost.view(1, -1)
        c_raw = w_g_cost * c_raw_g + w_l_cost * c_raw_l

        y_log = torch.log1p(c_raw.clamp_min(0.0))
        c_tilde = (y_log - self.scaler.mu) / (self.scaler.sigma + 1e-8)
        c_tilde = torch.clamp(c_tilde, -3.0, 3.0)

        return {"p_acc": p_acc, "c_raw": c_raw, "c_tilde": c_tilde}


@torch.no_grad()
def compute_adaptive_ensemble_weights(
    global_model: MLPQueryRouter,
    local_model: MLPQueryRouter,
    loader: DataLoader,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    global_model.eval()
    local_model.eval()
    n_models = global_model.n_models

    sum_abs_acc_g = torch.zeros(n_models, dtype=torch.float64)
    sum_abs_acc_l = torch.zeros(n_models, dtype=torch.float64)
    sum_abs_cost_g = torch.zeros(n_models, dtype=torch.float64)
    sum_abs_cost_l = torch.zeros(n_models, dtype=torch.float64)
    cnt_acc = torch.zeros(n_models, dtype=torch.float64)
    cnt_cost = torch.zeros(n_models, dtype=torch.float64)

    for batch in loader:
        x = batch["x"].to(DEVICE)
        y_acc = batch["y_acc"].to(DEVICE)
        m_acc = batch["m_acc"].to(DEVICE)
        y_cost_tilde = batch["y_cost_tilde"].to(DEVICE)
        m_cost = batch["m_cost"].to(DEVICE)

        out_g = global_model(x)
        out_l = local_model(x)

        err_acc_g = torch.abs(out_g["p_acc"] - y_acc) * m_acc
        err_acc_l = torch.abs(out_l["p_acc"] - y_acc) * m_acc
        sum_abs_acc_g += err_acc_g.sum(dim=0).detach().cpu()
        sum_abs_acc_l += err_acc_l.sum(dim=0).detach().cpu()
        cnt_acc += m_acc.sum(dim=0).detach().cpu()

        if global_model.scaler is None:
            y_cost_raw = y_cost_tilde
            c_raw_g = out_g["c_raw"] if "c_raw" in out_g else out_g["c_tilde"]
            c_raw_l = out_l["c_raw"] if "c_raw" in out_l else out_l["c_tilde"]
        else:
            y_cost_raw = global_model.scaler.denormalize(y_cost_tilde)
            c_raw_g = out_g["c_raw"] if "c_raw" in out_g else global_model.scaler.denormalize(out_g["c_tilde"])
            c_raw_l = out_l["c_raw"] if "c_raw" in out_l else global_model.scaler.denormalize(out_l["c_tilde"])

        err_cost_g = torch.abs(c_raw_g - y_cost_raw) * m_cost
        err_cost_l = torch.abs(c_raw_l - y_cost_raw) * m_cost
        sum_abs_cost_g += err_cost_g.sum(dim=0).detach().cpu()
        sum_abs_cost_l += err_cost_l.sum(dim=0).detach().cpu()
        cnt_cost += m_cost.sum(dim=0).detach().cpu()

    err_acc_g = sum_abs_acc_g / (cnt_acc + eps)
    err_acc_l = sum_abs_acc_l / (cnt_acc + eps)
    err_cost_g = sum_abs_cost_g / (cnt_cost + eps)
    err_cost_l = sum_abs_cost_l / (cnt_cost + eps)

    inv_acc_g = 1.0 / torch.clamp(err_acc_g, min=eps)
    inv_acc_l = 1.0 / torch.clamp(err_acc_l, min=eps)
    inv_cost_g = 1.0 / torch.clamp(err_cost_g, min=eps)
    inv_cost_l = 1.0 / torch.clamp(err_cost_l, min=eps)

    denom_acc = inv_acc_g + inv_acc_l
    denom_cost = inv_cost_g + inv_cost_l

    w_global_acc = inv_acc_g / denom_acc
    w_local_acc = inv_acc_l / denom_acc
    w_global_cost = inv_cost_g / denom_cost
    w_local_cost = inv_cost_l / denom_cost

    mask_acc = cnt_acc > 0
    mask_cost = cnt_cost > 0
    w_global_acc = torch.where(mask_acc, w_global_acc, torch.full_like(w_global_acc, 0.5))
    w_local_acc = torch.where(mask_acc, w_local_acc, torch.full_like(w_local_acc, 0.5))
    w_global_cost = torch.where(mask_cost, w_global_cost, torch.full_like(w_global_cost, 0.5))
    w_local_cost = torch.where(mask_cost, w_local_cost, torch.full_like(w_local_cost, 0.5))

    return (
        w_global_acc.float(),
        w_local_acc.float(),
        w_global_cost.float(),
        w_local_cost.float(),
    )

def _print_adaptive_weights(
    cid: int,
    w_global_acc: torch.Tensor,
    w_local_acc: torch.Tensor,
    w_global_cost: torch.Tensor,
    w_local_cost: torch.Tensor,
):
    print(f"[AdaptiveEnsemble] Client {cid} per-model weights:")
    wga = w_global_acc.detach().cpu().numpy()
    wla = w_local_acc.detach().cpu().numpy()
    wgc = w_global_cost.detach().cpu().numpy()
    wlc = w_local_cost.detach().cpu().numpy()
    for k, name in enumerate(MODEL_NAMES):
        print(
            f"  {name}: acc(g={wga[k]:.3f}, l={wla[k]:.3f}) "
            f"cost(g={wgc[k]:.3f}, l={wlc[k]:.3f})"
        )

@torch.no_grad()
def predict_heads(model: MLPQueryRouter, loader: DataLoader):
    model.eval()
    P_list, Craw_list, Macc_list, Mcost_list, Yacc_list, Ycost_tilde_list = [], [], [], [], [], []
    for batch in loader:
        x = batch["x"].to(DEVICE)
        out = model.forward(x)
        assert "c_raw" in out, "Cost must be raw for evaluation (scaled costs break lambda sweep)."
        P_list.append(out["p_acc"].cpu())
        Craw_list.append(out["c_raw"].cpu() if "c_raw" in out else out["c_tilde"].cpu())
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

    # Denormalize GT cost (for oracle realization)
    if hasattr(model, "scaler") and (model.scaler is not None):
        mu = model.scaler.mu.cpu(); sigma = model.scaler.sigma.cpu()
        Y_log = Ycost_til * (sigma + 1e-8) + mu
        Ycost_raw = torch.expm1(Y_log).clamp_min(0.0)
    else:
        Ycost_raw = Ycost_til
    return P, C, Macc, Mcost, Yacc, Ycost_raw

def evaluate_curve(model: MLPQueryRouter, loader: DataLoader, lambda_grid: np.ndarray):
    """
    For each lambda:
      choose model m* = argmax (pred_acc - λ * pred_cost)
      realize (acc, cost) from ground-truth on chosen arms (where observed)
      average → (cost, acc, coverage)
    Returns (sorted_points_by_cost, normalized_auc).
    """
    P, Cpred, Macc, Mcost, Yacc, Ycost_raw = predict_heads(model, loader)

    points = []
    rows = torch.arange(P.size(0))
    for lam in lambda_grid:
        U = P - float(lam) * Cpred
        choice = torch.argmax(U, dim=1)
        obs = (Macc[rows, choice] > 0.5) & (Mcost[rows, choice] > 0.5)
        if obs.sum() == 0:
            points.append((np.nan, np.nan, 0)); continue
        yacc  = Yacc[rows[obs], choice[obs]]
        ycost = Ycost_raw[rows[obs], choice[obs]]
        points.append((float(ycost.mean()), float(yacc.mean()), int(obs.sum())))

    pts = [(c,a,cv) for (c,a,cv) in points if (c==c) and (a==a)]
    if len(pts) < 2: return pts, 0.0
    pts_sorted = sorted(pts, key=lambda t: t[0])
    costs = np.array([t[0] for t in pts_sorted], dtype=np.float64)
    accs  = np.array([t[1] for t in pts_sorted], dtype=np.float64)
    area = np.trapz(accs, costs)
    crng = float(costs.max() - costs.min()) if len(costs) > 1 else 0.0
    auc = float(area / (crng + 1e-12)) if crng > 0 else 0.0
    return pts_sorted, auc


class Client:
    def __init__(self, cid: int, train_loader: DataLoader, test_loader: DataLoader):
        self.cid = cid
        self.loader = train_loader  
        self.train_loader = train_loader
        self.test_loader = test_loader

    def local_train_from(self, base_model: nn.Module, epochs=1, lr=2e-4, weight_decay=1e-4):
        # clone base weights (global at that round)
        model = MLPQueryRouter(
            q_dim=base_model.trunk[0].in_features,
            n_models=base_model.n_models,
            hidden=HIDDEN, dropout=DROPOUT, scaler=base_model.scaler
        ).to(DEVICE)
        model.load_state_dict(base_model.state_dict(), strict=True)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()
        for _ in range(epochs):
            for batch in self.train_loader:  # Use train_loader explicitly
                for k,v in batch.items():
                    if isinstance(v, torch.Tensor): batch[k] = v.to(DEVICE)
                losses = model.masked_losses(batch)
                opt.zero_grad(set_to_none=True)
                losses["L_total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        # delta & weight for FedAvg
        delta = {k: (model.state_dict()[k] - base_model.state_dict()[k]).detach().cpu()
                 for k in model.state_dict()}
        weight = len(self.train_loader.dataset)
        return model, delta, weight

class FedAvgServer:
    def __init__(self, global_model: MLPQueryRouter):
        self.model = global_model
        self.history = {
            "round": [],
            "global_val_auc": [],           # global model on central VAL
            "per_client_val_auc_on_global": [],  # [{cid: auc_on_global_VAL}] if requested
            "per_client_curve_on_global": [],    # [{cid: curve_on_global_VAL}]
        }

    def aggregate(self, deltas: List[Dict[str, torch.Tensor]], weights: List[int]):
        if not deltas: return
        total = float(sum(weights))
        avg_delta = {}
        for k in deltas[0].keys():
            avg_delta[k] = sum(deltas[i][k] * (weights[i]/total) for i in range(len(deltas)))
        with torch.no_grad():
            state = self.model.state_dict()
            for k in state:
                state[k].add_(avg_delta[k].to(state[k].device))
            self.model.load_state_dict(state, strict=True)

def clone_from_global(server: FedAvgServer):
    model = MLPQueryRouter(
        q_dim=server.model.trunk[0].in_features,
        n_models=server.model.n_models,
        hidden=HIDDEN, dropout=DROPOUT, scaler=server.model.scaler
    ).to(DEVICE)
    model.load_state_dict(server.model.state_dict(), strict=True)
    return model

def federated_train_and_track(
    server: FedAvgServer,
    clients: List[Client],
    global_test_loader: DataLoader,
    rounds: int = MAX_ROUNDS,
    participate_frac: float = PARTICIPATION_FRAC,
    local_epochs: int = LOCAL_EPOCHS,
    lr_local: float = LR_LOCAL,
    weight_decay: float = WEIGHT_DECAY,
    lambda_grid: np.ndarray = LAMBDA_GRID,
    track_client_ids_on_global: Optional[List[int]] = None,
):
    rng = np.random.default_rng(SEED)

    # --- Round 0: evaluate the initial global model BEFORE any training ---
    per_client_auc_global0 = {}
    per_client_curve_global0 = {}

    _, g_auc0 = evaluate_curve(server.model, global_test_loader, lambda_grid)
    if track_client_ids_on_global:
        for cid in track_client_ids_on_global:
            cl_loader = clients[cid].test_loader
            c_curve, c_auc = evaluate_curve(server.model, cl_loader, lambda_grid)
            per_client_auc_global0[cid] = c_auc
            per_client_curve_global0[cid] = c_curve

    server.history["round"].append(0)
    server.history["global_val_auc"].append(g_auc0)
    server.history["per_client_val_auc_on_global"].append(per_client_auc_global0)
    server.history["per_client_curve_on_global"].append(per_client_curve_global0)

    print(f"[Round 00] Global model AUC (TEST): {g_auc0:.4f}")

    best_auc = g_auc0
    best_state = {k: v.detach().cpu() for k, v in server.model.state_dict().items()}

    # --- Rounds 1..R: standard FedAvg ---
    for r in range(1, rounds + 1):
        # local steps
        m = max(1, int(round(participate_frac * len(clients))))
        chosen = rng.choice(len(clients), size=m, replace=False)
        deltas, weights = [], []

        for idx in chosen:
            _, d, w = clients[idx].local_train_from(
                server.model,
                epochs=local_epochs,
                lr=lr_local,
                weight_decay=weight_decay,
            )
            deltas.append(d); weights.append(w)

        # aggregate
        server.aggregate(deltas, weights)

        # evaluate global model on collected global TEST
        _, g_auc = evaluate_curve(server.model, global_test_loader, lambda_grid)
        per_client_auc_global = {}
        per_client_curve_global = {}

        # also evaluate on selected clients' local TEST datasets
        if track_client_ids_on_global:
            for cid in track_client_ids_on_global:
                cl_loader = clients[cid].test_loader
                c_curve, c_auc = evaluate_curve(server.model, cl_loader, lambda_grid)
                per_client_auc_global[cid] = c_auc
                per_client_curve_global[cid] = c_curve

        server.history["round"].append(r)
        server.history["global_val_auc"].append(g_auc)
        server.history["per_client_val_auc_on_global"].append(per_client_auc_global)
        server.history["per_client_curve_on_global"].append(per_client_curve_global)

        print(f"[Round {r:02d}] Global model AUC (TEST): {g_auc:.4f}")

        if g_auc > best_auc:
            best_auc = g_auc
            best_state = {k: v.detach().cpu() for k, v in server.model.state_dict().items()}

    return server


clients = []
for spec in clients_spec:
    clients.append(Client(
        cid=spec["cid"],
        train_loader=spec["train_loader"],
        test_loader=spec["test_loader"]
    ))

# Global model + server
q_dim = X.shape[1]
scaler = PerModelLogZScaler(torch.tensor(mu_log), torch.tensor(sigma_log))
global_model = MLPQueryRouter(
    q_dim=q_dim,
    n_models=K,        # number of LLMs
    hidden=HIDDEN,
    dropout=DROPOUT,
    scaler=scaler
).to(DEVICE)
server = FedAvgServer(global_model)

# EXP-1 tracking setting: evaluate the global model (each round) on these clients' local TEST sets as well.
TRACK_CLIENTS_FOR_GLOBAL_AUC: List[int] = [i for i in range(len(clients))]   # e.g., [0,1,2]; leave [] for only global TEST curve

# Train FL and record history
server = federated_train_and_track(
    server, clients, global_test_loader,
    rounds=MAX_ROUNDS, participate_frac=PARTICIPATION_FRAC,
    local_epochs=LOCAL_EPOCHS, lr_local=LR_LOCAL, weight_decay=WEIGHT_DECAY,
    lambda_grid=LAMBDA_GRID, track_client_ids_on_global=TRACK_CLIENTS_FOR_GLOBAL_AUC
)


from torch.utils.data import ConcatDataset, DataLoader

central_train_ds = ConcatDataset([c.train_loader.dataset for c in clients])

central_train_loader = DataLoader(
    central_train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,             
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
)
def train_centralized_on_union(
    train_loader: DataLoader,
    q_dim: int,
    n_models: int,
    scaler: PerModelLogZScaler,
    global_test_loader: DataLoader,
    epochs: int = CENTRALIZED_EPOCHS,
    lr: float = LR_LOCAL,
    weight_decay: float = WEIGHT_DECAY,
    early_stop_patience: Optional[int] = None,
):
    model = MLPQueryRouter(
        q_dim=q_dim,
        n_models=n_models,
        hidden=HIDDEN,
        dropout=DROPOUT,
        scaler=scaler
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc, best_state = -1.0, None
    patience = early_stop_patience

    auc_history = []

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            losses = model.masked_losses(batch)
            opt.zero_grad(set_to_none=True)
            losses["L_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        _, auc_ep = evaluate_curve(model, global_test_loader, LAMBDA_GRID)
        auc_history.append(float(auc_ep))
        print(f"[Centralized][Epoch {ep:03d}] AUC(TEST): {auc_ep:.4f}")

        # early stop logic unchanged (but now uses auc_ep we already computed)
        if early_stop_patience is not None:
            if auc_ep > best_auc + 1e-6:
                best_auc = auc_ep
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                patience = early_stop_patience
            else:
                patience -= 1
                if patience <= 0:
                    break

    model.eval()
    return model, auc_history  


centralized_model, centralized_auc_hist = train_centralized_on_union(
    central_train_loader,
    q_dim=q_dim,
    n_models=K,
    scaler=scaler,
    global_test_loader=global_test_loader,
    epochs=CENTRALIZED_EPOCHS,
    lr=CENTRALIZED_LR,
    weight_decay=CENTRALIZED_WD,
    early_stop_patience=None
)
def plot_centralized_auc_vs_epochs(auc_hist, outpath):
    fig, ax = plt.subplots(figsize=(8,5))
    xs = list(range(1, len(auc_hist)+1))
    ax.plot(xs, auc_hist, marker="o", linewidth=2.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized AUC")
    ax.set_title("Centralized Training — AUC vs Epoch (Global TEST)")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close(fig)

plot_centralized_auc_vs_epochs(
    centralized_auc_hist,
    os.path.join(OUTPUT_DIR, "centralized_auc_vs_epochs.png")
)


def plot_federated_vs_centralized_on_global(
    server: FedAvgServer,
    centralized_model: MLPQueryRouter,
    global_test_loader: DataLoader,
):
    fed_curve, fed_auc = evaluate_curve(server.model, global_test_loader, LAMBDA_GRID)
    cent_curve, cent_auc = evaluate_curve(centralized_model, global_test_loader, LAMBDA_GRID)

    def _clean(pts):
        pts = [(c, a, cv) for (c, a, cv) in pts if (c == c) and (a == a)]
        return sorted(pts, key=lambda t: t[0])

    fed_pts = _clean(fed_curve)
    cent_pts = _clean(cent_curve)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        [p[0] for p in fed_pts], [p[1] for p in fed_pts],
        marker="o", linewidth=2.5,
        label=f"Federated (AUC={fed_auc:.3f})"
    )
    ax.plot(
        [p[0] for p in cent_pts], [p[1] for p in cent_pts],
        marker="o", linestyle="--", linewidth=2.5,
        label=f"Centralized (AUC={cent_auc:.3f})"
    )
    ax.set_xlabel("Cost"); ax.set_ylabel("Accuracy")
    ax.set_title("Federated vs Centralized Router (Global TEST)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "federated_vs_centralized_global_test.png"))

plot_federated_vs_centralized_on_global(server, centralized_model, global_test_loader)


def plot_global_auc_vs_rounds(server_history: Dict[str, list], client_ids: Optional[List[int]] = None):
    rounds = server_history["round"]
    g_aucs = server_history["global_val_auc"]  # Actually TEST now, but keeping same key

    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(rounds, g_aucs, marker="o", linewidth=2.5, label="Global Split (TEST)")
    if client_ids:
        for cid in client_ids:
            series = [server_history["per_client_val_auc_on_global"][t].get(cid, np.nan) for t in range(len(rounds))]
            ax.plot(rounds, series, marker=".", linestyle="--", label=f"Client {cid} Local TEST")
    ax.set_xlabel("Communication Round"); ax.set_ylabel("Normalized AUC")
    ax.set_title("Global Model — Normalized AUC vs Rounds")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "global_auc_vs_rounds.png"))

plot_global_auc_vs_rounds(server.history, client_ids=PLOT_CLIENT_IDS)

def plot_global_model_curves(
    server: FedAvgServer,
    clients: List[Client],
    global_test_loader: DataLoader,
    split: str = "test",               # Only "test" now since no val
    also_client_ids: Optional[List[int]] = None
):
    assert split == "test", "Only 'test' split available in new setup"
    curves = {}

    g_curve, g_auc = evaluate_curve(server.model, global_test_loader, LAMBDA_GRID)
    curves[f"GlobalModel/{split.upper()}"] = g_curve

    if also_client_ids:
        for cid in also_client_ids:
            cl_loader = clients[cid].test_loader
            c_curve, c_auc = evaluate_curve(server.model, cl_loader, LAMBDA_GRID)
            curves[f"GlobalModel@Client{cid}Local{split.upper()}"] = c_curve

    # plot
    fig, ax = plt.subplots(figsize=(8,5))
    for name, pts in curves.items():
        pts = [(c,a,cv) for (c,a,cv) in pts if (c==c) and (a==a)]
        pts = sorted(pts, key=lambda t: t[0])
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Cost"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Global Model Acc–Cost Curves ({split.upper()})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "global_model_curves.png"))


# Example: plot global acc–cost on TEST and (optionally) on specified clients' local TEST sets
plot_global_model_curves(server, clients, global_test_loader, split="test", also_client_ids=PLOT_CLIENT_IDS)


def train_client_to_convergence_local_only(
    clients: List[Client],
    q_dim: int, n_models: int, scaler: PerModelLogZScaler,
    global_test_loader: DataLoader = None,
    epochs: int = CENTRALIZED_EPOCHS, lr: float = 3*CENTRALIZED_LR, weight_decay: float = CENTRALIZED_WD,
    early_stop_patience: Optional[int] = None,
    early_stop_on: str = "client"
):
    results = {}
    auc_histories = {}

    for client in clients:
        model = MLPQueryRouter(
            q_dim=q_dim,
            n_models=n_models,
            hidden=HIDDEN,
            dropout=DROPOUT,
            scaler=scaler
        ).to(DEVICE)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        es_loader = client.test_loader if early_stop_on == "client" else global_test_loader

        best_auc, best_state = -1.0, None
        patience = early_stop_patience

        auc_hist = []

        for ep in range(1, epochs+1):
            model.train()
            for batch in client.train_loader:
                for k,v in batch.items():
                    if isinstance(v, torch.Tensor): batch[k] = v.to(DEVICE)
                losses = model.masked_losses(batch)
                opt.zero_grad(set_to_none=True)
                losses["L_total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            _, auc_ep = evaluate_curve(model, es_loader, LAMBDA_GRID)
            auc_hist.append(float(auc_ep))

            if early_stop_patience is not None:
                if auc_ep > best_auc + 1e-6:
                    best_auc = auc_ep
                    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    patience = early_stop_patience
                else:
                    patience -= 1
                    if patience <= 0:
                        break

        model.eval()
        results[client.cid] = model
        auc_histories[client.cid] = auc_hist  
        print(f"Client {client.cid}: finished local-only training.")

    return results, auc_histories  


# Train local-only models (from scratch) — adjust epochs/patience as desired
local_only_models, local_only_auc_hists = train_client_to_convergence_local_only(
    clients, q_dim=q_dim, n_models=K, scaler=scaler,
    global_test_loader=global_test_loader,
    epochs=CENTRALIZED_EPOCHS+7, lr=CENTRALIZED_LR*4, weight_decay=CENTRALIZED_WD*2,
    early_stop_patience=None, early_stop_on="client"
)

def _curve_xy_from_points(points):
    pts = [(float(c), float(a)) for (c, a, cv) in points if (c == c) and (a == a)]
    if not pts:
        return np.array([]), np.array([])
    pts.sort(key=lambda t: t[0])
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    uniq_xs, inv = np.unique(xs, return_inverse=True)
    ys_sum = np.zeros_like(uniq_xs, dtype=np.float64)
    counts = np.zeros_like(uniq_xs, dtype=np.float64)
    for i, idx in enumerate(inv):
        ys_sum[idx] += ys[i]
        counts[idx] += 1.0
    return uniq_xs, ys_sum / counts

def _interp_extrap(x_new: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if xs.size == 0:
        return np.full_like(x_new, np.nan, dtype=np.float64)
    if xs.size == 1:
        return np.full_like(x_new, ys[0], dtype=np.float64)
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    y_new = np.interp(x_new, xs, ys)
    left = x_new < xs[0]
    if np.any(left):
        denom = xs[1] - xs[0]
        slope = (ys[1] - ys[0]) / denom if denom != 0 else 0.0
        y_new[left] = ys[0] + slope * (x_new[left] - xs[0])
    right = x_new > xs[-1]
    if np.any(right):
        denom = xs[-1] - xs[-2]
        slope = (ys[-1] - ys[-2]) / denom if denom != 0 else 0.0
        y_new[right] = ys[-1] + slope * (x_new[right] - xs[-1])
    return y_new

def _common_x_grid(curves_by_method: dict[str, list[tuple[np.ndarray, np.ndarray]]],
                   num_points: int = 200) -> Optional[np.ndarray]:
    xs_all = []
    for curves in curves_by_method.values():
        for xs, _ in curves:
            if xs.size:
                xs_all.append(xs)
    if not xs_all:
        return None
    min_x = min(xs.min() for xs in xs_all)
    max_x = max(xs.max() for xs in xs_all)
    if max_x == min_x:
        return np.array([min_x], dtype=np.float64)
    num_points = max(2, int(num_points))
    return np.linspace(min_x, max_x, num_points, dtype=np.float64)

def _average_curves_on_grid(curves: list[tuple[np.ndarray, np.ndarray]],
                            x_grid: np.ndarray) -> Optional[np.ndarray]:
    if not curves:
        return None
    ys_stack = []
    for xs, ys in curves:
        if xs.size == 0:
            continue
        y_interp = _interp_extrap(x_grid, xs, ys)
        if np.all(np.isnan(y_interp)):
            continue
        ys_stack.append(y_interp)
    if not ys_stack:
        return None
    return np.mean(np.vstack(ys_stack), axis=0)

def plot_local_only_auc_histories_grid(
    auc_hists: Dict[int, list],
    outpath: str,
    nrows: int = 4,
    ncols: int = 3,
):
    client_ids = sorted(auc_hists.keys())
    n_clients = len(client_ids)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3 * nrows),
        sharex=False, sharey=False
    )
    axes = axes.flatten()

    for idx, cid in enumerate(client_ids):
        if idx >= nrows * ncols:
            break
        ax = axes[idx]
        hist = auc_hists[cid]
        xs = list(range(1, len(hist)+1))
        ax.plot(xs, hist, marker="o", linewidth=1.8)
        ax.set_title(f"Client {cid} (best={max(hist):.2f})")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("AUC")

    for j in range(n_clients, nrows * ncols):
        fig.delaxes(axes[j])

    fig.suptitle("Local-only Training — AUC vs Epoch (per-client LOCAL TEST)", y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(outpath)
    plt.close(fig)

    cap = nrows * ncols
    avg_curves = []
    for cid in client_ids[:cap]:
        hist = np.array(auc_hists[cid], dtype=np.float64)
        if hist.size == 0:
            continue
        xs = np.arange(1, hist.size + 1, dtype=np.float64)
        mask = np.isfinite(hist)
        if not np.any(mask):
            continue
        avg_curves.append((xs[mask], hist[mask]))

    if avg_curves:
        xs_all = [xs for xs, _ in avg_curves if xs.size]
        min_x = min(xs.min() for xs in xs_all)
        max_x = max(xs.max() for xs in xs_all)
        num_points = max(2, int(round(max_x - min_x)) + 1)
        x_grid = _common_x_grid({"Local-only": avg_curves}, num_points=num_points)
        if x_grid is not None:
            y_avg = _average_curves_on_grid(avg_curves, x_grid)
            if y_avg is not None:
                fig, ax = plt.subplots(figsize=(7.5, 4.5))
                ax.plot(x_grid, y_avg, linewidth=2.2, label="Local-only avg")
                ax.set_xlabel("Epoch")
                ax.set_ylabel("AUC")
                ax.set_title("Local-only Training - Average AUC vs Epoch (per-client LOCAL TEST)")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend()
                plt.tight_layout()
                avg_outpath = outpath.replace("_grid.png", "_avg.png")
                if avg_outpath == outpath:
                    base, ext = os.path.splitext(outpath)
                    avg_outpath = f"{base}_avg{ext}"
                plt.savefig(avg_outpath, dpi=200)
                plt.close(fig)

plot_local_only_auc_histories_grid(
    local_only_auc_hists,
    os.path.join(OUTPUT_DIR, "local_only_auc_vs_epochs_grid.png"),
    nrows=4, ncols=3
)

def plot_local_vs_global_on_global_split(
    server: FedAvgServer,
    local_models: Dict[int, MLPQueryRouter],
    clients: List[Client],
    global_test_loader: DataLoader,
    split: str = "test",                   # Only "test" now
    client_ids: Optional[List[int]] = None,
    top_k: Optional[int] = None           # optionally plot top-k by AUC to avoid clutter
):
    assert split == "test", "Only 'test' split available in new setup"
    loader = global_test_loader

    # Global curve
    curves = {}
    g_curve, g_auc = evaluate_curve(server.model, loader, LAMBDA_GRID)
    curves["GlobalModel"] = (g_curve, g_auc)

    # Local clients
    if client_ids is None:
        client_ids = sorted(local_models.keys())
    for cid in client_ids:
        c_model = local_models[cid]
        c_curve, c_auc = evaluate_curve(c_model, loader, LAMBDA_GRID)
        curves[f"Client{cid}"] = (c_curve, c_auc)

    # Optional top-k selection (by AUC)
    items = [(name, pts, auc) for name,(pts,auc) in curves.items() if name!="GlobalModel"]
    items_sorted = sorted(items, key=lambda t: t[2], reverse=True)
    if top_k is not None:
        items_sorted = items_sorted[:top_k]

    # Plot
    fig, ax = plt.subplots(figsize=(9,6))
    # global first
    pts = sorted([p for p in curves["GlobalModel"][0] if (p[0]==p[0] and p[1]==p[1])], key=lambda t: t[0])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=3, label=f"GlobalModel ({curves['GlobalModel'][1]:.3f})")

    for name, pts, auc in items_sorted:
        pts = sorted([p for p in pts if (p[0]==p[0] and p[1]==p[1])], key=lambda t: t[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"{name} ({auc:.3f})")

    ax.set_xlabel("Cost"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Local-only Clients vs Global Model on GLOBAL {split.upper()} Split")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "local_vs_global_on_global_split.png"))


# Example: compare on global TEST
plot_local_vs_global_on_global_split(server, local_only_models, clients, global_test_loader, split="test", client_ids=PLOT_CLIENT_IDS)

def plot_local_vs_global_on_local_splits(
    server: FedAvgServer,
    local_models: Dict[int, MLPQueryRouter],
    clients: List[Client],
    split: str = "test",                  # Only "test" now
    client_ids: Optional[List[int]] = None
):
    """
    For each chosen client, evaluate:
      - local model on that client's local split (solid line)
      - global model on that client's local split (dashed line)
    Use the same color per client; legend distinguishes solid vs dashed.
    """
    assert split == "test", "Only 'test' split available in new setup"
    if client_ids is None:
        client_ids = sorted(local_models.keys())

    fig, ax = plt.subplots(figsize=(10,6))
    for idx, cid in enumerate(client_ids):
        local_loader = clients[cid].test_loader
        # local client model (solid)
        c_model = local_models[cid]
        c_curve, c_auc = evaluate_curve(c_model, local_loader, LAMBDA_GRID)
        pts = sorted([p for p in c_curve if (p[0]==p[0] and p[1]==p[1])], key=lambda t: t[0])
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        line_local, = ax.plot(xs, ys, marker="o", label=f"Client{cid} local ({c_auc:.3f})")

        # global model on the SAME local split (dashed, same color)
        g_curve, g_auc = evaluate_curve(server.model, local_loader, LAMBDA_GRID)
        pts_g = sorted([p for p in g_curve if (p[0]==p[0] and p[1]==p[1])], key=lambda t: t[0])
        ax.plot([p[0] for p in pts_g], [p[1] for p in pts_g],
                marker="o", linestyle="--", color=line_local.get_color(),
                label=f"Global@Client{cid} ({g_auc:.3f})")

    ax.set_xlabel("Cost"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Local-only vs Global on LOCAL Client Splits ({split.upper()})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(ncol=2)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "local_vs_global_on_local_splits.png"))

def plot_local_vs_global_on_local_splits_grid(
    server: FedAvgServer,
    local_models: Dict[int, MLPQueryRouter],
    clients: List[Client],
    split: str = "test",
    nrows: int = 4,
    ncols: int = 3,
):
    """
    4x3 grid: each subplot is a client,
    showing local-only vs global model on that client's LOCAL split.
    Suptitle reports avg AUC across plotted clients for each method.
    """
    assert split == "test", "Only 'test' split available in new setup"

    client_ids = sorted(local_models.keys())
    cap = nrows * ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3 * nrows),
        sharex=False, sharey=False
    )
    axes = axes.flatten()

    def _clean(pts):
        pts = [(c, a, cv) for (c, a, cv) in pts if (c == c) and (a == a)]
        return sorted(pts, key=lambda t: t[0])

    def _mean_or_nan(xs):
        xs = [x for x in xs if (x == x)]
        return float(np.mean(xs)) if xs else float("nan")

    local_aucs, global_aucs = [], []
    avg_curves = {"Local": [], "Global": []}
    used = 0

    for idx, cid in enumerate(client_ids[:cap]):
        ax = axes[idx]
        used += 1

        local_loader = clients[cid].test_loader
        c_curve, c_auc = evaluate_curve(local_models[cid], local_loader, LAMBDA_GRID)
        g_curve, g_auc = evaluate_curve(server.model,     local_loader, LAMBDA_GRID)

        c_pts = _clean(c_curve)
        g_pts = _clean(g_curve)

        if c_pts:
            ax.plot(
                [p[0] for p in c_pts], [p[1] for p in c_pts],
                marker="o", linestyle="-", linewidth=1.5,
                label=f"Local ({c_auc:.2f})"
            )
            local_aucs.append(c_auc)
            avg_curves["Local"].append(_curve_xy_from_points(c_pts))

        if g_pts:
            ax.plot(
                [p[0] for p in g_pts], [p[1] for p in g_pts],
                marker="o", linestyle="--", linewidth=1.5,
                label=f"Global ({g_auc:.2f})"
            )
            global_aucs.append(g_auc)
            avg_curves["Global"].append(_curve_xy_from_points(g_pts))

        ax.set_title(f"Client {cid}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    # Remove unused axes
    for j in range(used, cap):
        fig.delaxes(axes[j])

    avg_local = _mean_or_nan(local_aucs)
    avg_global = _mean_or_nan(global_aucs)

    fig.suptitle(
        f"Local-only vs Global on LOCAL TEST splits | avg AUC: Local={avg_local:.2f}, Global={avg_global:.2f}",
        y=0.98
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = os.path.join(OUTPUT_DIR, "local_vs_global_on_local_splits_grid.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved:", out_path)

def plot_adaptive_ensemble_on_local_splits_grid(
    global_model: MLPQueryRouter,
    local_models: Dict[int, MLPQueryRouter],
    adaptive_ensembles: Dict[int, AdaptiveEnsembleRouter],
    clients: List[Client],
    split: str = "test",
    nrows: int = 4,
    ncols: int = 3,
):
    """
    4x3 grid: each subplot is a client,
    showing Global vs Local-only vs Adaptive-ensemble on that client's LOCAL split.
    """
    assert split == "test", "Only 'test' split available in new setup"

    client_ids = sorted(set(local_models.keys()) & set(adaptive_ensembles.keys()))
    cap = nrows * ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3 * nrows),
        sharex=False, sharey=False
    )
    axes = axes.flatten()

    def _clean(pts):
        pts = [(c, a, cv) for (c, a, cv) in pts if (c == c) and (a == a)]
        return sorted(pts, key=lambda t: t[0])

    def _mean_or_nan(xs):
        xs = [x for x in xs if (x == x)]
        return float(np.mean(xs)) if xs else float("nan")

    local_aucs, global_aucs, adaptive_aucs = [], [], []
    avg_curves = {"Global": [], "Local-only": [], "Adaptive": []}
    used = 0

    for idx, cid in enumerate(client_ids[:cap]):
        ax = axes[idx]
        used += 1

        local_loader = clients[cid].test_loader
        g_curve, g_auc = evaluate_curve(global_model, local_loader, LAMBDA_GRID)
        l_curve, l_auc = evaluate_curve(local_models[cid], local_loader, LAMBDA_GRID)
        a_curve, a_auc = evaluate_curve(adaptive_ensembles[cid], local_loader, LAMBDA_GRID)

        g_pts = _clean(g_curve)
        l_pts = _clean(l_curve)
        a_pts = _clean(a_curve)

        if g_pts:
            ax.plot(
                [p[0] for p in g_pts], [p[1] for p in g_pts],
                linestyle="--", linewidth=1.5,
                label=f"Global ({g_auc:.2f})"
            )
            global_aucs.append(g_auc)
            avg_curves["Global"].append(_curve_xy_from_points(g_pts))

        if l_pts:
            ax.plot(
                [p[0] for p in l_pts], [p[1] for p in l_pts],
                linestyle="-.", linewidth=1.5,
                label=f"Local ({l_auc:.2f})"
            )
            local_aucs.append(l_auc)
            avg_curves["Local-only"].append(_curve_xy_from_points(l_pts))

        if a_pts:
            ax.plot(
                [p[0] for p in a_pts], [p[1] for p in a_pts],
                linestyle="-", linewidth=1.5,
                label=f"Adaptive ({a_auc:.2f})"
            )
            adaptive_aucs.append(a_auc)
            avg_curves["Adaptive"].append(_curve_xy_from_points(a_pts))

        ax.set_title(f"Client {cid}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    for j in range(used, cap):
        fig.delaxes(axes[j])

    avg_g = _mean_or_nan(global_aucs)
    avg_l = _mean_or_nan(local_aucs)
    avg_a = _mean_or_nan(adaptive_aucs)

    fig.suptitle(
        "Adaptive ensemble on LOCAL TEST splits | "
        f"avg AUC: Global={avg_g:.2f}, Local-only={avg_l:.2f}, Adaptive={avg_a:.2f}",
        y=0.98
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = os.path.join(OUTPUT_DIR, "adaptive_ensemble_vs_global_local_grid.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved:", out_path)

    x_grid = _common_x_grid(avg_curves, num_points=200)
    if x_grid is not None:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        y_g = _average_curves_on_grid(avg_curves["Global"], x_grid)
        y_l = _average_curves_on_grid(avg_curves["Local-only"], x_grid)
        y_a = _average_curves_on_grid(avg_curves["Adaptive"], x_grid)
        if y_g is not None:
            ax.plot(x_grid, y_g, linestyle="--", linewidth=2.5, label="Global avg")
        if y_l is not None:
            ax.plot(x_grid, y_l, linestyle="-.", linewidth=2.5, label="Local-only avg")
        if y_a is not None:
            ax.plot(x_grid, y_a, linestyle="-", linewidth=2.5, label="Adaptive avg")
        ax.set_xlabel("Cost")
        ax.set_ylabel("Accuracy")
        ax.set_title("Adaptive ensemble on LOCAL TEST splits - Average (interp/extrap)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        avg_out_path = os.path.join(OUTPUT_DIR, "adaptive_ensemble_vs_global_local_avg.png")
        plt.savefig(avg_out_path, dpi=200)
        plt.close(fig)


# Example: all clients, 4x3 grid of local vs global on LOCAL TEST splits
plot_local_vs_global_on_local_splits_grid(
    server, local_only_models, clients,
    split="test", nrows=4, ncols=3
)



from typing import Dict, Optional

def clone_from_global(server: FedAvgServer) -> MLPQueryRouter:
    model = MLPQueryRouter(
        q_dim=server.model.trunk[0].in_features,
        n_models=server.model.n_models,
        hidden=HIDDEN, dropout=DROPOUT, scaler=server.model.scaler
    ).to(DEVICE)
    model.load_state_dict(server.model.state_dict(), strict=True)
    return model

def personalize_clients_from_global(
    server: FedAvgServer,
    clients: list[Client],
    local_epochs: int = 1,
    lr: float = LR_LOCAL,
    weight_decay: float = WEIGHT_DECAY,
) -> Dict[int, MLPQueryRouter]:
    """
    For each client:
      - start from the FINAL global model
      - run a fixed number of local updates on the client's *training* loader
      - return a dict {cid: personalized_model}
    """
    personalized: Dict[int, MLPQueryRouter] = {}
    for c in clients:
        model = clone_from_global(server)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()
        for _ in range(local_epochs):
            for batch in c.train_loader:  # client's TRAIN loader
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(DEVICE)
                losses = model.masked_losses(batch)
                opt.zero_grad(set_to_none=True)
                losses["L_total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        model.eval()
        personalized[c.cid] = model
        print(f"[Personalized] Client {c.cid} — finished {local_epochs} local epoch(s).")
    return personalized



def _sorted_clean_points(points):
    pts = [(c,a,cv) for (c,a,cv) in points if (c==c) and (a==a)]
    return sorted(pts, key=lambda t: t[0])

def plot_personalized_vs_global_on_local_data(
    server: FedAvgServer,
    clients: list[Client],
    personalized: Dict[int, MLPQueryRouter],
    split: str = "test",                 # Only "test" now
    lambda_grid: Optional[np.ndarray] = None,
    client_ids: Optional[list[int]] = None,
):
    """
    For each client:
      - evaluate GLOBAL (final) model on the client's LOCAL split → dashed line
      - evaluate PERSONALIZED (client) model on the client's LOCAL split → solid line
      - use the same color for that client's two lines
    """
    assert split == "test", "Only 'test' split available in new setup"
    if lambda_grid is None:
        # fallback if not defined upstream
        lambda_grid = np.geomspace(1e-1, 1e3, 25)

    if client_ids is None:
        client_ids = sorted(personalized.keys())

    fig, ax = plt.subplots(figsize=(11, 7))

    for cid in client_ids:
        cl_loader = clients[cid].test_loader

        # Personalized client curve (solid)
        p_curve, p_auc = evaluate_curve(personalized[cid], cl_loader, lambda_grid)
        p_pts = _sorted_clean_points(p_curve)
        if len(p_pts) == 0:
            print(f"Client {cid}: no valid personalized curve points on {split}.")
            continue
        line_personal, = ax.plot(
            [p[0] for p in p_pts], [p[1] for p in p_pts],
            marker="o", linewidth=2.5, label=f"Client{cid} personalized ({p_auc:.3f})"
        )
        color = line_personal.get_color()

        # Global model on the same client's local split (dashed; same color)
        g_curve, g_auc = evaluate_curve(server.model, cl_loader, lambda_grid)
        g_pts = _sorted_clean_points(g_curve)
        if len(g_pts) == 0:
            print(f"Client {cid}: no valid global curve points on {split}.")
            continue
        ax.plot(
            [p[0] for p in g_pts], [p[1] for p in g_pts],
            marker="o", linestyle="--", linewidth=2.0, color=color,
            label=f"Global@Client{cid} ({g_auc:.3f})"
        )

    ax.set_xlabel("Cost")
    ax.set.ylabel("Accuracy")
    ax.set_title(f"Personalized vs Global on LOCAL Client Data ({split.upper()})\n"
                 f"Solid = Personalized, Dashed = Global (same color per client)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "personalized_vs_global_on_local_data.png"))


# 1) Personalize each client from the final global model
personalized_models = personalize_clients_from_global(
    server, clients,
    local_epochs=2,         # ← set your desired number of local fine-tuning epochs
    lr=LR_LOCAL/6,
    weight_decay=WEIGHT_DECAY
)

# Build weighted (0.5, 0.5) ensembles of global + local-only per client
ensemble_models: Dict[int, MLPQueryRouter] = {}
for cid, local_model in local_only_models.items():
    ensemble = WeightedEnsembleRouter(
        server.model,        # final global model
        local_model,         # client-specific local-only model
        w_global=0.5,
        w_local=0.5,
    )
    ensemble.to(DEVICE)
    ensemble.eval()
    ensemble_models[cid] = ensemble

print(f"Built {len(ensemble_models)} global+local ensembles (0.5, 0.5 weights).")

def build_adaptive_ensemble_models(
    global_model: MLPQueryRouter,
    local_models: Dict[int, MLPQueryRouter],
    clients: list[Client],
) -> Dict[int, AdaptiveEnsembleRouter]:
    adaptive: Dict[int, AdaptiveEnsembleRouter] = {}
    for cid, local_model in local_models.items():
        train_loader = clients[cid].train_loader
        w_ga, w_la, w_gc, w_lc = compute_adaptive_ensemble_weights(
            global_model, local_model, train_loader
        )
        _print_adaptive_weights(cid, w_ga, w_la, w_gc, w_lc)
        ens = AdaptiveEnsembleRouter(
            global_model, local_model,
            w_global_acc=w_ga, w_local_acc=w_la,
            w_global_cost=w_gc, w_local_cost=w_lc
        )
        ens.to(DEVICE)
        ens.eval()
        adaptive[cid] = ens
    return adaptive

adaptive_ensemble_models = build_adaptive_ensemble_models(
    server.model, local_only_models, clients
)
print(f"Built {len(adaptive_ensemble_models)} adaptive ensembles.")

# Grid: Global vs Local-only vs Adaptive ensemble on local splits
plot_adaptive_ensemble_on_local_splits_grid(
    server.model, local_only_models, adaptive_ensemble_models, clients,
    split="test", nrows=4, ncols=3
)


def _sorted_clean_points(points):
    pts = [(c,a,cv) for (c,a,cv) in points if (c==c) and (a==a)]
    return sorted(pts, key=lambda t: t[0])

def plot_three_way_personalization_on_local_data(
    server: FedAvgServer,
    clients: list[Client],
    personalized: dict[int, MLPQueryRouter],   # from personalize_clients_from_global(...)
    local_only: dict[int, MLPQueryRouter],     # from train_client_to_convergence_local_only(...)
    split: str = "test",                         # Only "test" now
    lambda_grid: np.ndarray | None = None,
    client_ids: list[int] | None = None,
    top_k_by_auc: int | None = None,           # optionally show only top-k clients by personalized AUC
):
    """
    For each selected client, evaluate on that client's LOCAL split:
      - personalized model (solid)
      - global model (dashed)
      - local-only model (dot-dashed)
    The legend shows the average AUC for each method across all plotted clients.
    """
    import matplotlib.pyplot as plt
    assert split == "test", "Only 'test' split available in new setup"
    if lambda_grid is None:
        lambda_grid = np.geomspace(1e-1, 1e3, 25)

    # choose client set = intersection of available personalized & local-only models
    avail = sorted(set(personalized.keys()) & set(local_only.keys()))
    if client_ids is None:
        client_ids = avail
    else:
        client_ids = [cid for cid in client_ids if cid in avail]

    # rank by personalized AUC if requested
    if top_k_by_auc is not None:
        auc_list = []
        for cid in client_ids:
            cl_loader = clients[cid].test_loader
            _, p_auc = evaluate_curve(personalized[cid], cl_loader, lambda_grid)
            auc_list.append((cid, p_auc))
        client_ids = [cid for cid,_ in sorted(auc_list, key=lambda t: t[1], reverse=True)[:top_k_by_auc]]

    fig, ax = plt.subplots(figsize=(11, 7))

    # Collect AUCs for averaging
    aucs_personalized = []
    aucs_global = []
    aucs_local = []

    for cid in client_ids:
        cl_loader = clients[cid].test_loader

        # Personalized (solid)
        p_curve, p_auc = evaluate_curve(personalized[cid], cl_loader, lambda_grid)
        p_pts = _sorted_clean_points(p_curve)
        if not p_pts:
            print(f"[Client {cid}] No valid personalized points on {split}."); 
            continue
        (line_pers,) = ax.plot(
            [p[0] for p in p_pts], [p[1] for p in p_pts],
            marker="s", linewidth=2.5, label=None  # Remove per-client legend entries
        )
        color = line_pers.get_color()
        aucs_personalized.append(p_auc)

        # Global (dashed)
        g_curve, g_auc = evaluate_curve(server.model, cl_loader, lambda_grid)
        g_pts = _sorted_clean_points(g_curve)
        if g_pts:
            ax.plot(
                [p[0] for p in g_pts], [p[1] for p in g_pts],
                marker="^", linestyle="--", linewidth=2.0, color=color, label=None
            )
            aucs_global.append(g_auc)
        else:
            print(f"[Client {cid}] No valid global points on {split}.")

        # Local-only (dot-dashed)
        l_curve, l_auc = evaluate_curve(local_only[cid], cl_loader, lambda_grid)
        l_pts = _sorted_clean_points(l_curve)
        if l_pts:
            ax.plot(
                [p[0] for p in l_pts], [p[1] for p in l_pts],
                marker="o", linestyle="-.", linewidth=2.0, color=color, label=None
            )
            aucs_local.append(l_auc)
        else:
            print(f"[Client {cid}] No valid local-only points on {split}.")

    # Compute averages, taking care in case of empty list
    import numpy as np
    avg_auc_personalized = np.mean(aucs_personalized) if aucs_personalized else float('nan')
    avg_auc_global = np.mean(aucs_global) if aucs_global else float('nan')
    avg_auc_local = np.mean(aucs_local) if aucs_local else float('nan')

    # Add single legend with average AUCs
    legend_labels = [
        f"Personalized (avg AUC={avg_auc_personalized:.3f})",
        f"Global (avg AUC={avg_auc_global:.3f})",
        f"Local-only (avg AUC={avg_auc_local:.3f})"
    ]
    # Build dummy (invisible) lines for legend
    leg_lines = [
        ax.plot([], [], marker="s", color="black", linewidth=2.5)[0],         # Personalized
        ax.plot([], [], marker="^", color="black", linestyle="--", linewidth=2.0)[0], # Global
        ax.plot([], [], marker="o", color="black", linestyle="-.", linewidth=2.0)[0], # Local-only
    ]
    ax.set(xlabel="Cost", ylabel="Accuracy")
    ax.set_title(f"Three-way Routing Acc–Cost on LOCAL Client Data ({split.upper()})\n"
                 f"Solid = Personalized, Dashed = Global, Dot-dashed = Local-only (same color per client)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(leg_lines, legend_labels, ncol=1, fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT_DIR, "three_way_personalization_on_local_data.png"))


def plot_three_way_personalization_on_local_data_grid(
    server: FedAvgServer,
    clients: list[Client],
    personalized: dict[int, MLPQueryRouter],
    local_only: dict[int, MLPQueryRouter],
    split: str = "test",
    lambda_grid: np.ndarray | None = None,
    nrows: int = 4,
    ncols: int = 3,
):
    """
    4x3 grid of subplots: one panel per client,
    showing Personalized vs Global vs Local-only on that client's LOCAL split.
    Suptitle reports avg AUC across plotted clients for each method.
    """
    assert split == "test", "Only 'test' split available in new setup"
    if lambda_grid is None:
        lambda_grid = LAMBDA_GRID

    client_ids = sorted(set(personalized.keys()) & set(local_only.keys()))
    cap = nrows * ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3 * nrows),
        sharex=False, sharey=False
    )
    axes = axes.flatten()

    def _clean(pts):
        pts = [(c, a, cv) for (c, a, cv) in pts if (c == c) and (a == a)]
        return sorted(pts, key=lambda t: t[0])

    def _mean_or_nan(xs):
        xs = [x for x in xs if (x == x)]
        return float(np.mean(xs)) if xs else float("nan")

    pers_aucs, glob_aucs, local_aucs = [], [], []
    avg_curves = {"Personalized": [], "Global": [], "Local-only": []}
    used = 0

    for idx, cid in enumerate(client_ids[:cap]):
        ax = axes[idx]
        used += 1

        cl_loader = clients[cid].test_loader

        p_curve, p_auc = evaluate_curve(personalized[cid], cl_loader, lambda_grid)
        g_curve, g_auc = evaluate_curve(server.model,      cl_loader, lambda_grid)
        l_curve, l_auc = evaluate_curve(local_only[cid],   cl_loader, lambda_grid)

        p_pts = _clean(p_curve)
        g_pts = _clean(g_curve)
        l_pts = _clean(l_curve)

        if p_pts:
            ax.plot(
                [p[0] for p in p_pts], [p[1] for p in p_pts],
                marker="s", linestyle="-", linewidth=1.5,
                label=f"Pers ({p_auc:.2f})"
            )
            pers_aucs.append(p_auc)
            avg_curves["Personalized"].append(_curve_xy_from_points(p_pts))

        if g_pts:
            ax.plot(
                [p[0] for p in g_pts], [p[1] for p in g_pts],
                marker="^", linestyle="--", linewidth=1.5,
                label=f"Glob ({g_auc:.2f})"
            )
            glob_aucs.append(g_auc)
            avg_curves["Global"].append(_curve_xy_from_points(g_pts))

        if l_pts:
            ax.plot(
                [p[0] for p in l_pts], [p[1] for p in l_pts],
                marker="o", linestyle="-.", linewidth=1.5,
                label=f"Local ({l_auc:.2f})"
            )
            local_aucs.append(l_auc)
            avg_curves["Local-only"].append(_curve_xy_from_points(l_pts))

        ax.set_title(f"Client {cid}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    for j in range(used, cap):
        fig.delaxes(axes[j])

    avg_p = _mean_or_nan(pers_aucs)
    avg_g = _mean_or_nan(glob_aucs)
    avg_l = _mean_or_nan(local_aucs)

    fig.suptitle(
        f"Three-way personalization on LOCAL TEST splits | avg AUC: Pers={avg_p:.2f}, Glob={avg_g:.2f}, Local={avg_l:.2f}",
        y=0.98
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = os.path.join(OUTPUT_DIR, "three_way_personalization_grid.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved:", out_path)

    x_grid = _common_x_grid(avg_curves, num_points=200)
    if x_grid is not None:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        y_pers = _average_curves_on_grid(avg_curves["Personalized"], x_grid)
        y_glob = _average_curves_on_grid(avg_curves["Global"], x_grid)
        y_local = _average_curves_on_grid(avg_curves["Local-only"], x_grid)
        if y_pers is not None:
            ax.plot(x_grid, y_pers, linestyle="-", linewidth=2.5, label="Personalized avg")
        if y_glob is not None:
            ax.plot(x_grid, y_glob, linestyle="--", linewidth=2.5, label="Global avg")
        if y_local is not None:
            ax.plot(x_grid, y_local, linestyle="-.", linewidth=2.5, label="Local-only avg")
        ax.set_xlabel("Cost")
        ax.set_ylabel("Accuracy")
        ax.set_title("Three-way personalization on LOCAL TEST splits - Average (interp/extrap)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        avg_out_path = os.path.join(OUTPUT_DIR, "three_way_personalization_avg.png")
        plt.savefig(avg_out_path, dpi=200)
        plt.close(fig)


def plot_personalization_methods_on_local_data_grid(
    server: FedAvgServer,
    clients: list[Client],
    personalized: dict[int, MLPQueryRouter],
    ensembles: dict[int, MLPQueryRouter],
    adaptive_ensembles: dict[int, AdaptiveEnsembleRouter],
    local_only: dict[int, MLPQueryRouter],
    split: str = "test",
    lambda_grid: np.ndarray | None = None,
    nrows: int = 4,
    ncols: int = 3,
):
    """
    4x3 grid: for each client, compare
      - Global model
      - Local-only model
      - Fine-tuned personalization (Personalized-FT)
      - Global+Local ensemble (0.5, 0.5)
      - Adaptive ensemble (per-model weights)
    on that client's LOCAL split.

    Each subplot has its own legend with per-client AUCs.
    Suptitle reports avg AUC across plotted clients for each method.
    """
    assert split == "test", "Only 'test' split available in new setup"
    if lambda_grid is None:
        lambda_grid = LAMBDA_GRID

    client_ids = sorted(
        set(personalized.keys()) &
        set(ensembles.keys()) &
        set(adaptive_ensembles.keys()) &
        set(local_only.keys())
    )
    cap = nrows * ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3 * nrows),
        sharex=False, sharey=False
    )
    axes = axes.flatten()

    def _clean(pts):
        pts = [(c, a, cv) for (c, a, cv) in pts if (c == c) and (a == a)]
        return sorted(pts, key=lambda t: t[0])

    def _mean_or_nan(xs):
        xs = [x for x in xs if (x == x)]
        return float(np.mean(xs)) if xs else float("nan")

    aucs = {
        "Global": [],
        "Local-only": [],
        "Personalized-FT": [],
        "Ensemble(0.5G+0.5L)": [],
        "AdaptiveEnsemble": [],
    }
    avg_curves = {
        "Global": [],
        "Local-only": [],
        "Personalized-FT": [],
        "Ensemble(0.5G+0.5L)": [],
        "AdaptiveEnsemble": [],
    }

    used = 0
    for idx, cid in enumerate(client_ids[:cap]):
        ax = axes[idx]
        used += 1

        cl_loader = clients[cid].test_loader

        g_curve, g_auc = evaluate_curve(server.model,        cl_loader, lambda_grid)
        l_curve, l_auc = evaluate_curve(local_only[cid],     cl_loader, lambda_grid)
        p_curve, p_auc = evaluate_curve(personalized[cid],   cl_loader, lambda_grid)
        e_curve, e_auc = evaluate_curve(ensembles[cid],      cl_loader, lambda_grid)
        a_curve, a_auc = evaluate_curve(adaptive_ensembles[cid], cl_loader, lambda_grid)

        g_pts = _clean(g_curve)
        l_pts = _clean(l_curve)
        p_pts = _clean(p_curve)
        e_pts = _clean(e_curve)
        a_pts = _clean(a_curve)

        if g_pts:
            ax.plot(
                [p[0] for p in g_pts], [p[1] for p in g_pts],
                linestyle="--", linewidth=1.5,
                label=f"Global ({g_auc:.2f})"
            )
            aucs["Global"].append(g_auc)
            avg_curves["Global"].append(_curve_xy_from_points(g_pts))

        if l_pts:
            ax.plot(
                [p[0] for p in l_pts], [p[1] for p in l_pts],
                linestyle="-.", linewidth=1.5,
                label=f"Local-only ({l_auc:.2f})"
            )
            aucs["Local-only"].append(l_auc)
            avg_curves["Local-only"].append(_curve_xy_from_points(l_pts))

        if p_pts:
            ax.plot(
                [p[0] for p in p_pts], [p[1] for p in p_pts],
                linestyle="-", linewidth=1.5,
                label=f"Personalized-FT ({p_auc:.2f})"
            )
            aucs["Personalized-FT"].append(p_auc)
            avg_curves["Personalized-FT"].append(_curve_xy_from_points(p_pts))

        if e_pts:
            ax.plot(
                [p[0] for p in e_pts], [p[1] for p in e_pts],
                linestyle="-", linewidth=1.5,
                label=f"Ensemble ({e_auc:.2f})"
            )
            aucs["Ensemble(0.5G+0.5L)"].append(e_auc)
            avg_curves["Ensemble(0.5G+0.5L)"].append(_curve_xy_from_points(e_pts))

        if a_pts:
            ax.plot(
                [p[0] for p in a_pts], [p[1] for p in a_pts],
                linestyle="-", linewidth=1.5,
                label=f"Adaptive ({a_auc:.2f})"
            )
            aucs["AdaptiveEnsemble"].append(a_auc)
            avg_curves["AdaptiveEnsemble"].append(_curve_xy_from_points(a_pts))

        ax.set_title(f"Client {cid}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    for j in range(used, cap):
        fig.delaxes(axes[j])

    avg_g = _mean_or_nan(aucs["Global"])
    avg_l = _mean_or_nan(aucs["Local-only"])
    avg_p = _mean_or_nan(aucs["Personalized-FT"])
    avg_e = _mean_or_nan(aucs["Ensemble(0.5G+0.5L)"])
    avg_a = _mean_or_nan(aucs["AdaptiveEnsemble"])

    fig.suptitle(
        "Personalization methods on LOCAL TEST splits | "
        f"avg AUC: Global={avg_g:.2f}, Local-only={avg_l:.2f}, "
        f"Pers-FT={avg_p:.2f}, Ensemble={avg_e:.2f}, Adaptive={avg_a:.2f}",
        y=0.98
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = os.path.join(OUTPUT_DIR, "personalization_methods_finetune_vs_ensemble_grid.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved:", out_path)

    x_grid = _common_x_grid(avg_curves, num_points=200)
    if x_grid is not None:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        y_g = _average_curves_on_grid(avg_curves["Global"], x_grid)
        y_l = _average_curves_on_grid(avg_curves["Local-only"], x_grid)
        y_p = _average_curves_on_grid(avg_curves["Personalized-FT"], x_grid)
        y_e = _average_curves_on_grid(avg_curves["Ensemble(0.5G+0.5L)"], x_grid)
        y_a = _average_curves_on_grid(avg_curves["AdaptiveEnsemble"], x_grid)
        if y_g is not None:
            ax.plot(x_grid, y_g, linestyle="--", linewidth=2.5, label="Global avg")
        if y_l is not None:
            ax.plot(x_grid, y_l, linestyle="-.", linewidth=2.5, label="Local-only avg")
        if y_p is not None:
            ax.plot(x_grid, y_p, linestyle="-", linewidth=2.5, label="Personalized-FT avg")
        if y_e is not None:
            ax.plot(x_grid, y_e, linestyle="-", linewidth=2.5, label="Ensemble avg")
        if y_a is not None:
            ax.plot(x_grid, y_a, linestyle="-", linewidth=2.5, label="Adaptive avg")
        ax.set_xlabel("Cost")
        ax.set_ylabel("Accuracy")
        ax.set_title("Personalization methods on LOCAL TEST splits - Average (interp/extrap)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()
        plt.tight_layout()
        avg_out_path = os.path.join(OUTPUT_DIR, "personalization_methods_finetune_vs_ensemble_avg.png")
        plt.savefig(avg_out_path, dpi=200)
        plt.close(fig)


# Grid summary: global vs local-only vs personalization methods (FT, ensemble, adaptive)
plot_personalization_methods_on_local_data_grid(
    server, clients,
    personalized=personalized_models,
    ensembles=ensemble_models,
    adaptive_ensembles=adaptive_ensemble_models,
    local_only=local_only_models,
    split="test",
    lambda_grid=LAMBDA_GRID,
    nrows=4,
    ncols=3,
)


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

client_ids_all = sorted([c.cid for c in clients])

per_client_local = {}
for cid in client_ids_all:
    local_loader = clients[cid].test_loader
    per_client_local[str(cid)] = {
        "global": _eval_curve(server.model, local_loader),
        "local_only": _eval_curve(local_only_models[cid], local_loader),
        "personalized": _eval_curve(personalized_models[cid], local_loader),
        "ensemble": _eval_curve(ensemble_models[cid], local_loader),
        "adaptive_ensemble": _eval_curve(adaptive_ensemble_models[cid], local_loader),
    }

per_client_on_global = {}
for cid in client_ids_all:
    per_client_on_global[str(cid)] = {
        "local_only": _eval_curve(local_only_models[cid], global_test_loader),
        "personalized": _eval_curve(personalized_models[cid], global_test_loader),
        "ensemble": _eval_curve(ensemble_models[cid], global_test_loader),
        "adaptive_ensemble": _eval_curve(adaptive_ensemble_models[cid], global_test_loader),
    }

results = {
    "method": "mlp",
    "lambda_grid": _to_serializable(LAMBDA_GRID),
    "dataset": {
        "num_samples": int(N),
        "num_models": int(K),
    },
    "federated": {
        "history": _to_serializable(server.history),
    },
    "centralized": {
        "auc_history": _to_serializable(centralized_auc_hist),
    },
    "evaluations": {
        "global_test": {
            "global_model": _eval_curve(server.model, global_test_loader),
            "centralized_model": _eval_curve(centralized_model, global_test_loader),
        },
        "per_client_local_test": per_client_local,
        "per_client_on_global_test": per_client_on_global,
    },
    "local_only_auc_histories": _to_serializable(local_only_auc_hists),
}

out_json = os.path.join(OUTPUT_DIR, "mlp_experiment_results.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Saved experiment results to", out_json)
