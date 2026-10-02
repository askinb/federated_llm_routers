import os
if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = "./kmeans_out"
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
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

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
EMB_COL = "all_mpnet_base_v2_embedding"  # or other embedding columns in the parquet

CHAR_WORD_LEN_FLAG = False
TSNE_FLAG = False
CLIENT_TRAIN_FRAC, CLIENT_TEST_FRAC = 0.75, 0.25

K_LOCAL = 15    # number of local clusters per client
K_GLOBAL = 20   # number of global clusters at server
CENTRALIZED_K = 30   # number of global clusters at server for centralized training

KMEANS_LOCAL_N_INIT = 3
KMEANS_GLOBAL_N_INIT = 3
KMEANS_LOCAL_MAX_ITER = 30
KMEANS_GLOBAL_MAX_ITER = 30

BATCH_SIZE = 256

LAMBDA_GRID = np.geomspace(1e-2, 1e7, 100)

N_CLIENTS = 10
DIRICHLET_ALPHA = 0.6          # task non-IID
MODEL_COV_MINMAX = (1, 1)  # fraction of models available per client
LABEL_KEEP_FRAC = 0.1          # per-entry label keep probability

EPS = 1e-8

FALLBACK_ACC = 0.0      # Zero accuracy when no samples available
FALLBACK_COST = 1e6     # Large cost when no samples available

PLOT_CLIENT_IDS = [i for i in range(N_CLIENTS)]

MODEL_DIRICHLET_ALPHA = 0.45

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

# no central splits; we partition directly by clients
eval_names_array = eval_names.to_numpy()
print("Full dataset ready for client partitioning:", X.shape)

class CentralDataset(Dataset):
    """
    Stores features + (possibly sparse) labels for accuracy and cost.
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

def partition_by_evalname_indices_new(eval_names: np.ndarray, n_clients: int, alpha: float = 0.5, rng=None):
    """
    New partitioning logic: For each category, sample a Dirichlet distribution over clients.
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

    # NEW: collect the *masked* client-train arrays for centralized training
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
            # keep client test, but skip train contributions
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

        # NEW: add this client’s *masked train view* into centralized/global train
        gX_parts.append(X_all[local_train_rows])
        gYacc_parts.append(Yacc_train_c)
        gMacc_parts.append(Macc_train_c)
        gYc_parts.append(Yc_train_c)
        gMc_parts.append(Mc_train_c)

    collected_train_idxs = np.array(collected_train_idxs, dtype=int)
    collected_test_idxs  = np.array(collected_test_idxs, dtype=int)

    # NEW: global train == union of client-train views (masked/sparsified)
    if len(gX_parts) > 0:
        gX    = np.concatenate(gX_parts, axis=0)
        gYacc = np.concatenate(gYacc_parts, axis=0)
        gMacc = np.concatenate(gMacc_parts, axis=0)
        gYc   = np.concatenate(gYc_parts, axis=0)
        gMc   = np.concatenate(gMc_parts, axis=0)
    else:
        # degenerate edge case
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
        "train_indices": collected_train_idxs,  # indices in original X_all (labels are masked in global_train_ds)
        "test_indices": collected_test_idxs,
    }


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

# Client wrapper (mainly to hold datasets & loaders)
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

    fig, axes = plt.subplots(4, 3, figsize=(12, 12), sharex=True, sharey=True)
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


def _kmeans_assign_labels_torch(
    X_t: torch.Tensor,
    centers_t: torch.Tensor,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Assign each row in X_t to its nearest center (squared L2). Returns (N,) long tensor."""
    N = X_t.shape[0]
    if N == 0:
        return torch.empty((0,), device=X_t.device, dtype=torch.long)

    # d(x,c)^2 = ||x||^2 + ||c||^2 - 2 x·c
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
        # (Optional) Keep a NumPy fallback if you want parity debugging
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
        # Optional NumPy fallback
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

class WeightedEnsembleRouter(nn.Module):
    """
    Combines two routers via a convex combination of their predicted
    accuracy and cost estimates (used for personalization).
    """
    def __init__(self, global_router: KMeansRouter, local_router: KMeansRouter,
                 w_global: float = 0.5, w_local: float = 0.5):
        super().__init__()
        assert abs(w_global + w_local - 1.0) < 1e-6, "Weights must sum to 1"
        self.global_router = global_router
        self.local_router = local_router
        self.w_global = w_global
        self.w_local = w_local

    def forward(self, x: torch.Tensor):
        out_g = self.global_router(x)
        out_l = self.local_router(x)
        p_acc = self.w_global * out_g["p_acc"] + self.w_local * out_l["p_acc"]
        c_raw = self.w_global * out_g["c_raw"] + self.w_local * out_l["c_raw"]
        return {"p_acc": p_acc, "c_raw": c_raw}

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

    # Step 3 & 4: compute global per-cluster per-model stats
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

def build_ensemble_routers(
    global_router: KMeansRouter,
    local_routers: Dict[int, KMeansRouter],
    w_global: float = 0.5,
    w_local: float = 0.5,
) -> Dict[int, WeightedEnsembleRouter]:
    """
    Personalized routers via simple ensemble: per client, combine
    global router + local-only router.
    """
    ensembles: Dict[int, WeightedEnsembleRouter] = {}
    for cid, loc_router in local_routers.items():
        ens = WeightedEnsembleRouter(global_router, loc_router, w_global=w_global, w_local=w_local)
        ens.to(DEVICE)
        ens.eval()
        ensembles[cid] = ens
    return ensembles

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
      choose model m* = argmax (pred_acc - λ * pred_cost)
      realize (acc, cost) from ground-truth on chosen arms (where observed)
      average → (cost, acc, coverage)
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
ensemble_routers = build_ensemble_routers(
    fed_router, local_only_routers,
    w_global=0.5, w_local=0.5
)

print(f"Built federated router, centralized router, "
      f"{len(local_only_routers)} local-only routers, and "
      f"{len(ensemble_routers)} ensemble routers.")

def plot_federated_vs_centralized_on_global(
    fed_router: KMeansRouter,
    cent_router: KMeansRouter,
    global_test_loader: DataLoader,
):
    fed_curve, fed_auc = evaluate_curve(fed_router, global_test_loader, LAMBDA_GRID)
    cent_curve, cent_auc = evaluate_curve(cent_router, global_test_loader, LAMBDA_GRID)

    def _clean(pts):
        pts = [(c,a,cv) for (c,a,cv) in pts if (c==c) and (a==a)]
        return sorted(pts, key=lambda t: t[0])

    fed_pts = _clean(fed_curve)
    cent_pts = _clean(cent_curve)

    fig, ax = plt.subplots(figsize=(8,5))
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
    ax.set_title("Federated vs Centralized K-means Router (Global TEST)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "federated_vs_centralized_global_test.png")
    plt.savefig(out_path)
    print("Saved:", out_path)

plot_federated_vs_centralized_on_global(fed_router, cent_router, global_test_loader)

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

def plot_global_auc_vs_rounds(server_history: Dict[str, list], client_ids: Optional[List[int]] = None):
    rounds = server_history["round"]
    g_aucs = server_history["global_val_auc"]

    fig, ax = plt.subplots(figsize=(9,5))
    ax.plot(rounds, g_aucs, marker="o", linewidth=2.5, label="Global TEST")
    if client_ids:
        for cid in client_ids:
            series = [server_history["per_client_val_auc_on_global"][t].get(cid, np.nan) for t in range(len(rounds))]
            ax.plot(rounds, series, marker=".", linestyle="--", label=f"Client {cid} Local TEST")
    ax.set_xlabel("Round"); ax.set_ylabel("Normalized AUC")
    ax.set_title("Global Router — Normalized AUC vs Rounds (K-means, single round)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "global_auc_vs_rounds.png")
    plt.savefig(out_path)
    print("Saved:", out_path)

router_history = build_single_round_history(fed_router, clients, global_test_loader)
plot_global_auc_vs_rounds(router_history, client_ids=PLOT_CLIENT_IDS)

def plot_global_model_curves(
    global_router: KMeansRouter,
    clients: List[Client],
    global_test_loader: DataLoader,
    split: str = "test",
    also_client_ids: Optional[List[int]] = None
):
    assert split == "test"
    curves = {}

    g_curve, g_auc = evaluate_curve(global_router, global_test_loader, LAMBDA_GRID)
    curves[f"GlobalRouter/{split.upper()}"] = g_curve

    if also_client_ids:
        for cid in also_client_ids:
            cl_loader = clients[cid].test_loader
            c_curve, c_auc = evaluate_curve(global_router, cl_loader, LAMBDA_GRID)
            curves[f"GlobalRouter@Client{cid}Local{split.upper()}"] = c_curve

    fig, ax = plt.subplots(figsize=(8,5))
    for name, pts in curves.items():
        pts = [(c,a,cv) for (c,a,cv) in pts if (c==c) and (a==a)]
        pts = sorted(pts, key=lambda t: t[0])
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Cost"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Global K-means Router Acc–Cost Curves ({split.upper()})")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "global_model_curves.png")
    plt.savefig(out_path)
    print("Saved:", out_path)

plot_global_model_curves(fed_router, clients, global_test_loader, split="test",
                         also_client_ids=PLOT_CLIENT_IDS)

def plot_local_vs_global_on_global_split(
    global_router: KMeansRouter,
    local_routers: Dict[int, KMeansRouter],
    clients: List[Client],
    global_test_loader: DataLoader,
    split: str = "test",
    client_ids: Optional[List[int]] = None,
    top_k: Optional[int] = None,
):
    assert split == "test"
    loader = global_test_loader

    curves = {}
    g_curve, g_auc = evaluate_curve(global_router, loader, LAMBDA_GRID)
    curves["GlobalRouter"] = (g_curve, g_auc)

    if client_ids is None:
        client_ids = sorted(local_routers.keys())
    for cid in client_ids:
        c_model = local_routers[cid]
        c_curve, c_auc = evaluate_curve(c_model, loader, LAMBDA_GRID)
        curves[f"Client{cid}LocalOnly"] = (c_curve, c_auc)

    items = [(name, pts, auc) for name,(pts,auc) in curves.items() if name!="GlobalRouter"]
    items_sorted = sorted(items, key=lambda t: t[2], reverse=True)
    if top_k is not None:
        items_sorted = items_sorted[:top_k]

    fig, ax = plt.subplots(figsize=(9,6))
    # global first
    pts = sorted([p for p in curves["GlobalRouter"][0] if (p[0]==p[0] and p[1]==p[1])], key=lambda t: t[0])
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", linewidth=3,
            label=f"GlobalRouter ({curves['GlobalRouter'][1]:.3f})")

    for name, pts, auc in items_sorted:
        pts = sorted([p for p in pts if (p[0]==p[0] and p[1]==p[1])], key=lambda t: t[0])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o",
                label=f"{name} ({auc:.3f})")

    ax.set_xlabel("Cost"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Local-only Clients vs Global Router on GLOBAL {split.upper()} Split")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(ncol=2)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "local_vs_global_on_global_split.png")
    plt.savefig(out_path)
    print("Saved:", out_path)

plot_local_vs_global_on_global_split(
    fed_router, local_only_routers, clients,
    global_test_loader, split="test",
    client_ids=PLOT_CLIENT_IDS
)

def plot_local_vs_global_on_local_splits_grid(
    global_router: KMeansRouter,
    local_routers: Dict[int, KMeansRouter],
    clients: List[Client],
    split: str = "test",
    nrows: int = 4,
    ncols: int = 3,
):
    assert split == "test"

    import numpy as np

    client_ids = sorted(local_routers.keys())
    cap = nrows * ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3 * nrows),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    def _clean(pts):
        pts = [(c, a, cv) for (c, a, cv) in pts if (c == c) and (a == a)]
        return sorted(pts, key=lambda t: t[0])

    def _mean_or_nan(xs):
        xs = [x for x in xs if (x == x)]
        return float(np.mean(xs)) if xs else float("nan")

    local_aucs, global_aucs = [], []
    used = 0

    for idx, cid in enumerate(client_ids[:cap]):
        ax = axes[idx]
        used += 1

        local_loader = clients[cid].test_loader

        c_curve, c_auc = evaluate_curve(local_routers[cid], local_loader, LAMBDA_GRID)
        g_curve, g_auc = evaluate_curve(global_router,      local_loader, LAMBDA_GRID)

        c_pts = _clean(c_curve)
        g_pts = _clean(g_curve)

        if c_pts:
            ax.plot(
                [p[0] for p in c_pts], [p[1] for p in c_pts],
                marker="o", linestyle="-", linewidth=1.5,
                label=f"Local ({c_auc:.2f})"
            )
            local_aucs.append(c_auc)

        if g_pts:
            ax.plot(
                [p[0] for p in g_pts], [p[1] for p in g_pts],
                marker="o", linestyle="--", linewidth=1.5,
                label=f"Global ({g_auc:.2f})"
            )
            global_aucs.append(g_auc)

        ax.set_title(f"Client {cid}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    # Remove unused axes
    for j in range(used, cap):
        fig.delaxes(axes[j])

    avg_local = _mean_or_nan(local_aucs)
    avg_global = _mean_or_nan(global_aucs)

    fig.suptitle(
        f"Local-only vs Global on LOCAL TEST splits (K-means) | "
        f"avg AUC: Local={avg_local:.2f}, Global={avg_global:.2f}",
        y=0.98
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = os.path.join(OUTPUT_DIR, "local_vs_global_on_local_splits_grid.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved:", out_path)

plot_local_vs_global_on_local_splits_grid(
    fed_router, local_only_routers, clients,
    split="test", nrows=4, ncols=3
)

def plot_personalization_methods_on_local_data_grid(
    global_router,
    personalized_ensembles,
    local_only,
    clients,
    split="test",
    lambda_grid=LAMBDA_GRID,
    nrows=4,
    ncols=3,
):
    assert split == "test"

    import numpy as np

    client_ids = sorted(set(personalized_ensembles.keys()) & set(local_only.keys()))
    cap = nrows * ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4 * ncols, 3 * nrows),
        sharex=True, sharey=True
    )
    axes = axes.flatten()

    def _clean(pts):
        pts = [(c, a, cv) for (c, a, cv) in pts if (c == c) and (a == a)]
        return sorted(pts, key=lambda t: t[0])

    def _mean_or_nan(xs):
        xs = [x for x in xs if (x == x)]
        return float(np.mean(xs)) if xs else float("nan")

    aucs_dict = {
        "Global": [],
        "Local-only": [],
        "Ensemble(0.5G+0.5L)": [],
    }

    used = 0
    for idx, cid in enumerate(client_ids[:cap]):
        ax = axes[idx]
        used += 1

        cl_loader = clients[cid].test_loader

        g_curve, g_auc = evaluate_curve(global_router,             cl_loader, lambda_grid)
        l_curve, l_auc = evaluate_curve(local_only[cid],           cl_loader, lambda_grid)
        e_curve, e_auc = evaluate_curve(personalized_ensembles[cid], cl_loader, lambda_grid)

        g_pts = _clean(g_curve)
        l_pts = _clean(l_curve)
        e_pts = _clean(e_curve)

        if g_pts:
            ax.plot(
                [p[0] for p in g_pts], [p[1] for p in g_pts],
                linestyle="--", linewidth=1.5, color="C0",
                label=f"Global ({g_auc:.2f})"
            )
            aucs_dict["Global"].append(g_auc)

        if l_pts:
            ax.plot(
                [p[0] for p in l_pts], [p[1] for p in l_pts],
                linestyle="-.", linewidth=1.5, color="C1",
                label=f"Local-only ({l_auc:.2f})"
            )
            aucs_dict["Local-only"].append(l_auc)

        if e_pts:
            ax.plot(
                [p[0] for p in e_pts], [p[1] for p in e_pts],
                linestyle="-", marker="o", linewidth=1.5, color="C3",
                label=f"Ensemble ({e_auc:.2f})"
            )
            aucs_dict["Ensemble(0.5G+0.5L)"].append(e_auc)

        ax.set_title(f"Client {cid}")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    # Remove unused axes
    for j in range(used, cap):
        fig.delaxes(axes[j])

    avg_g = _mean_or_nan(aucs_dict["Global"])
    avg_l = _mean_or_nan(aucs_dict["Local-only"])
    avg_e = _mean_or_nan(aucs_dict["Ensemble(0.5G+0.5L)"])

    fig.suptitle(
        "Personalization methods on LOCAL TEST splits (K-means) | "
        f"avg AUC: Global={avg_g:.2f}, Local-only={avg_l:.2f}, Ensemble={avg_e:.2f}",
        y=0.98
    )
    plt.tight_layout(rect=(0, 0, 1, 0.94))

    out_path = os.path.join(OUTPUT_DIR, "personalization_methods_finetune_vs_ensemble_grid.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print("Saved:", out_path)


plot_personalization_methods_on_local_data_grid(
    fed_router,
    personalized_ensembles=ensemble_routers,
    local_only=local_only_routers,
    clients=clients,
    split="test",
    lambda_grid=LAMBDA_GRID,
    nrows=4,
    ncols=3,
)

print("\nAll K-means based experiments finished. Figures are in:", OUTPUT_DIR)

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
        "ensemble": _eval_curve(ensemble_routers[cid], local_loader),
    }

per_client_on_global = {}
for cid in client_ids_all:
    per_client_on_global[str(cid)] = {
        "local_only": _eval_curve(local_only_routers[cid], global_test_loader),
        "ensemble": _eval_curve(ensemble_routers[cid], global_test_loader),
    }

# Compute model-wise active sample counts for every client (based on train set mask)
per_client_model_active_samples = {}
for cid in client_ids_all:
    train_ds = clients[cid].train_ds   # CentralDataset
    # The mask for available labels for accepted (Yacc, Macc) and correct (Yc, Mc) models
    # We count for each model the number of train samples where the label is available (i.e., mask>0)
    # We'll use Macc (accuracies): shape [n_samples, K_MODELS]
    if hasattr(train_ds, "Macc"):
        mask = train_ds.Macc
    else:
        mask = train_ds[1]  # Just in case: (X, Yacc, Macc, ...)
    # Sum across axis=0 for the model dimension (gives number of samples per model with label available)
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
