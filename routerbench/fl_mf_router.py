
import os
if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = "./mf_out"
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
from torch.utils.data import Dataset, DataLoader, ConcatDataset

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

EMB_COL = "all_mpnet_base_v2_embedding"

CLIENT_TRAIN_FRAC, CLIENT_TEST_FRAC = 0.75, 0.25

D_EMBED = 64
NOISE_ALPHA = 0.05
BATCH_SIZE = 128
LR_LOCAL = 3e-4
WEIGHT_DECAY = 3e-4
LOCAL_EPOCHS = 1
MAX_ROUNDS = 10
PARTICIPATION_FRAC = 0.6
CENTRALIZED_LR = 5e-5
CENTRALIZED_WD = 3e-4
CENTRALIZED_EPOCHS = 15

LR_LOCAL_GRID = [1e-4, 3e-4, 1e-3]
MAX_ROUNDS_GRID = [5, 10, 15]

W_ACC = 1.0
W_COST = 1.0

LAMBDA_GRID = np.geomspace(1e-2, 1e7, 100)

N_CLIENTS = 10
DIRICHLET_ALPHA = 0.6
MODEL_COV_MINMAX = (1, 1)
LABEL_KEEP_FRAC = 0.1
MODEL_DIRICHLET_ALPHA = 0.45

EPS = 1e-8


cost_cols = [c for c in df.columns if c.endswith("|total_cost")]
candidate_models = [c[:-len("|total_cost")] for c in cost_cols]
MODEL_NAMES = sorted([m for m in candidate_models if m in df.columns])
K = len(MODEL_NAMES)
K_MODELS = K
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
    X_list.append(e.astype(np.float32))

X = np.stack(X_list).astype(np.float32)
N = len(df)
print("X shape:", X.shape)

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

mu_log, sigma_log = fit_logz_scaler(Y_cost, M_cost)
print("Cost normalization scalers computed from full dataset")
print("Scaler sample (mu, sigma):", mu_log[:3], sigma_log[:3])


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

def sample_model_subset_per_client(K, min_frac=0.6, max_frac=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    frac = rng.uniform(min_frac, max_frac)
    k = max(1, int(round(frac * K)))
    chosen = rng.choice(K, size=k, replace=False)
    mask = np.zeros(K, dtype=bool); mask[chosen] = True
    return mask

def generate_client_model_probabilities(n_models: int, dirichlet_alpha: Optional[float] = None, rng=None):
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
    mu_log: np.ndarray, sigma_log: np.ndarray,
    n_clients: int = 10,
    dirichlet_alpha: float = 0.5,
    model_cov_minmax=(0.6, 1.0),
    label_keep_frac: float = 0.7,
    client_train_frac: float = 0.75,
):
    rng = np.random.default_rng(SEED)

    client_row_idxs = partition_by_evalname_indices_new(
        eval_names_all, n_clients, dirichlet_alpha, rng=rng
    )

    clients_spec = []
    collected_train_idxs = []
    collected_test_idxs = []

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
            if len(local_test_rows) > 0:
                collected_test_idxs.extend(local_test_rows.tolist())
            continue

        collected_train_idxs.extend(local_train_rows.tolist())
        if len(local_test_rows) > 0:
            collected_test_idxs.extend(local_test_rows.tolist())

        model_mask = sample_model_subset_per_client(
            K, min_frac=model_cov_minmax[0], max_frac=model_cov_minmax[1], rng=rng
        )

        client_probs = generate_client_model_probabilities(K, MODEL_DIRICHLET_ALPHA, rng=rng)
        print(f"Client {c} model probabilities:", client_probs)

        Yacc_train_c, Macc_train_c, Yc_train_c, Mc_train_c = apply_client_masks(
            Yacc_all[local_train_rows], Macc_all[local_train_rows],
            Yc_all[local_train_rows],   Mc_all[local_train_rows],
            model_mask=model_mask, label_keep_frac=label_keep_frac,
            client_model_probs=client_probs, rng=rng
        )

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

        gX_parts.append(X_all[local_train_rows])
        gYacc_parts.append(Yacc_train_c)
        gMacc_parts.append(Macc_train_c)
        gYc_parts.append(Yc_train_c)
        gMc_parts.append(Mc_train_c)

    collected_train_idxs = np.array(collected_train_idxs, dtype=int)
    collected_test_idxs  = np.array(collected_test_idxs, dtype=int)

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

global_train_loader = global_sets["train_loader"]
global_test_loader = global_sets["test_loader"]


class PerModelLogZScaler(nn.Module):
    def __init__(self, mu_log: torch.Tensor, sigma_log: torch.Tensor):
        super().__init__()
        assert mu_log.shape == sigma_log.shape
        self.register_buffer("mu", mu_log.clone().detach())
        self.register_buffer("sigma", sigma_log.clone().detach())

    def denormalize(self, y_tilde: torch.Tensor):
        y_log = y_tilde * (self.sigma + 1e-8) + self.mu
        return torch.expm1(y_log).clamp_min(0.0)


class MFQueryRouter(nn.Module):
    def __init__(self, q_dim: int, n_models: int, d_embed: int = 64,
                 noise_alpha: float = 0.05, scaler: Optional[PerModelLogZScaler] = None):
        super().__init__()
        self.q_dim = q_dim
        self.n_models = n_models
        self.d_embed = d_embed
        self.noise_alpha = noise_alpha
        self.scaler = scaler

        self.model_embeds = nn.Embedding(n_models, d_embed)
        nn.init.xavier_uniform_(self.model_embeds.weight)

        self.text_proj = nn.Linear(q_dim, d_embed)

        self.acc_head = nn.Linear(d_embed, 1)
        self.cost_head = nn.Linear(d_embed, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        v_q = self.text_proj(x)

        if self.training and self.noise_alpha > 0:
            v_q = v_q + self.noise_alpha * torch.randn_like(v_q)

        model_ids = torch.arange(self.n_models, device=x.device)
        v_m = self.model_embeds(model_ids)

        h = v_q.unsqueeze(1) * v_m.unsqueeze(0)

        acc_logits = self.acc_head(h).squeeze(-1)
        c_tilde = self.cost_head(h).squeeze(-1)

        p_acc = torch.sigmoid(acc_logits)

        out = {"p_acc": p_acc, "c_tilde": c_tilde}
        if self.scaler is not None:
            out["c_raw"] = self.scaler.denormalize(c_tilde)
        return out

    def masked_losses(self, batch, w_acc=1.0, w_cost=1.0):
        x = batch["x"].to(next(self.parameters()).device)
        out = self.forward(x)
        p = out["p_acc"]
        c_til = out["c_tilde"]

        y_acc = batch["y_acc"].to(p.device)
        m_acc = batch["m_acc"].to(p.device)
        y_ctil = batch["y_cost_tilde"].to(p.device)
        m_cost = batch["m_cost"].to(p.device)

        L_acc = ((p - y_acc).pow(2) * m_acc).sum() / (m_acc.sum() + 1e-8)
        L_cost = (F.smooth_l1_loss(c_til, y_ctil, reduction="none") * m_cost).sum() / (m_cost.sum() + 1e-8)

        L_total = w_acc * L_acc + w_cost * L_cost
        return {"L_total": L_total, "L_acc": L_acc, "L_cost": L_cost}


class WeightedEnsembleRouter(nn.Module):
    def __init__(self, global_model, local_model, w_global=0.5, w_local=0.5):
        super().__init__()
        assert abs(w_global + w_local - 1.0) < 1e-6
        self.global_model = global_model
        self.local_model = local_model
        self.w_global = w_global
        self.w_local = w_local
        self.scaler = global_model.scaler

    def forward(self, x: torch.Tensor):
        out_g = self.global_model(x)
        out_l = self.local_model(x)

        p_acc = self.w_global * out_g["p_acc"] + self.w_local * out_l["p_acc"]

        if self.scaler is None:
            raise ValueError("Ensemble expects scaler so both routers output c_raw.")
        c_raw_g = out_g["c_raw"]
        c_raw_l = out_l["c_raw"]
        c_raw = self.w_global * c_raw_g + self.w_local * c_raw_l

        y_log = torch.log1p(c_raw.clamp_min(0.0))
        c_tilde = (y_log - self.scaler.mu) / (self.scaler.sigma + 1e-8)
        c_tilde = torch.clamp(c_tilde, -3.0, 3.0)

        return {"p_acc": p_acc, "c_raw": c_raw, "c_tilde": c_tilde}


@torch.no_grad()
def predict_heads(model: MFQueryRouter, loader: DataLoader):
    model.eval()
    P_list, Craw_list, Macc_list, Mcost_list, Yacc_list, Ycost_tilde_list = [], [], [], [], [], []
    for batch in loader:
        x = batch["x"].to(DEVICE)
        out = model.forward(x)
        assert "c_raw" in out, "Cost must be raw for evaluation (scaled costs break lambda sweep)."
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
        mu = model.scaler.mu.cpu(); sigma = model.scaler.sigma.cpu()
        Y_log = Ycost_til * (sigma + 1e-8) + mu
        Ycost_raw = torch.expm1(Y_log).clamp_min(0.0)
    else:
        Ycost_raw = Ycost_til
    return P, C, Macc, Mcost, Yacc, Ycost_raw

def evaluate_curve(model: MFQueryRouter, loader: DataLoader, lambda_grid: np.ndarray):
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
    def __init__(self, cid: int, train_ds, test_ds, train_loader: DataLoader, test_loader: DataLoader):
        self.cid = cid
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.loader = train_loader
        self.train_loader = train_loader
        self.test_loader = test_loader

    def local_train_from(self, base_model: MFQueryRouter, epochs=1, lr=3e-4, weight_decay=3e-4):
        model = MFQueryRouter(
            q_dim=base_model.q_dim,
            n_models=base_model.n_models,
            d_embed=base_model.d_embed,
            noise_alpha=base_model.noise_alpha,
            scaler=base_model.scaler,
        ).to(DEVICE)
        model.load_state_dict(base_model.state_dict(), strict=True)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()
        for _ in range(epochs):
            for batch in self.train_loader:
                batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                losses = model.masked_losses(batch, w_acc=W_ACC, w_cost=W_COST)
                opt.zero_grad(set_to_none=True)
                losses["L_total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

        delta = {}
        for k in model.state_dict():
            delta[k] = (model.state_dict()[k] - base_model.state_dict()[k]).detach().cpu()

        weight = len(self.train_loader.dataset)

        if hasattr(self.train_ds, 'Macc'):
            macc = self.train_ds.Macc
        else:
            macc = self.train_ds[1]
        if isinstance(macc, torch.Tensor):
            model_data_mask = (macc.sum(dim=0) > 0).cpu().numpy()
        else:
            model_data_mask = (np.array(macc).sum(axis=0) > 0)

        return model, delta, weight, model_data_mask


class FedAvgServer:
    def __init__(self, global_model: MFQueryRouter):
        self.model = global_model
        self.history = {"round": [], "global_val_auc": []}

    def aggregate(self, deltas: List[Dict[str, torch.Tensor]], weights: List[int],
                  model_masks: List[np.ndarray]):
        if not deltas:
            return

        total_w = float(sum(weights))
        state = self.model.state_dict()
        embed_key = "model_embeds.weight"

        for k in state:
            if k == embed_key:
                continue
            avg = sum(d[k] * (w / total_w) for d, w in zip(deltas, weights))
            state[k].add_(avg.to(state[k].device))

        n_models = state[embed_key].shape[0]
        for m in range(n_models):
            contributors = [i for i in range(len(deltas)) if model_masks[i][m]]
            if not contributors:
                continue
            total_w_m = float(sum(weights[i] for i in contributors))
            if total_w_m <= 0:
                continue
            avg_m = sum(deltas[i][embed_key][m] * (weights[i] / total_w_m)
                        for i in contributors)
            state[embed_key][m].add_(avg_m.to(state[embed_key].device))

        self.model.load_state_dict(state, strict=True)


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
):
    rng = np.random.default_rng(SEED + 999)

    _, g_auc0 = evaluate_curve(server.model, global_test_loader, lambda_grid)
    server.history["round"].append(0)
    server.history["global_val_auc"].append(g_auc0)
    best_auc = g_auc0
    best_state = {k: v.cpu().clone() for k, v in server.model.state_dict().items()}

    print(f"[Round 00] Global model AUC (TEST): {g_auc0:.4f}")

    for r in range(1, rounds + 1):
        m = max(1, int(round(participate_frac * len(clients))))
        chosen = rng.choice(len(clients), size=m, replace=False)
        deltas, weights, model_masks = [], [], []
        for idx in chosen:
            _, d, w, mm = clients[idx].local_train_from(
                server.model, epochs=local_epochs, lr=lr_local, weight_decay=weight_decay
            )
            deltas.append(d); weights.append(w); model_masks.append(mm)
        server.aggregate(deltas, weights, model_masks)
        _, g_auc = evaluate_curve(server.model, global_test_loader, lambda_grid)
        server.history["round"].append(r)
        server.history["global_val_auc"].append(g_auc)
        print(f"[Round {r:02d}] Global AUC: {g_auc:.4f}")
        if g_auc > best_auc:
            best_auc = g_auc
            best_state = {k: v.cpu().clone() for k, v in server.model.state_dict().items()}

    return server, best_state, best_auc


clients = []
for spec in clients_spec:
    clients.append(Client(
        cid=spec["cid"],
        train_ds=spec["train_ds"],
        test_ds=spec["test_ds"],
        train_loader=spec["train_loader"],
        test_loader=spec["test_loader"],
    ))

q_dim = X.shape[1]
scaler = PerModelLogZScaler(torch.tensor(mu_log), torch.tensor(sigma_log))


best_global_auc = -1.0
best_config = None
best_fed_state = None
sweep_results = {}

for lr in LR_LOCAL_GRID:
    for rounds in MAX_ROUNDS_GRID:
        key = f"lr_{lr}_rounds_{rounds}"
        print(f"\n=== Sweep: {key} ===")

        torch.manual_seed(SEED)
        sweep_model = MFQueryRouter(
            q_dim=q_dim,
            n_models=K_MODELS,
            d_embed=D_EMBED,
            noise_alpha=NOISE_ALPHA,
            scaler=scaler,
        ).to(DEVICE)
        sweep_server = FedAvgServer(sweep_model)

        sweep_server, s_best_state, s_best_auc = federated_train_and_track(
            sweep_server, clients, global_test_loader,
            rounds=rounds, participate_frac=PARTICIPATION_FRAC,
            lr_local=lr, weight_decay=WEIGHT_DECAY,
            local_epochs=LOCAL_EPOCHS,
            lambda_grid=LAMBDA_GRID,
        )

        sweep_results[key] = {
            "lr": lr,
            "rounds": rounds,
            "best_auc": float(s_best_auc),
            "auc_history": [float(v) for v in sweep_server.history["global_val_auc"]],
        }

        if s_best_auc > best_global_auc:
            best_global_auc = s_best_auc
            best_config = key
            best_fed_state = s_best_state

print(f"\nBest config: {best_config} with AUC={best_global_auc:.4f}")


best_lr = sweep_results[best_config]["lr"]
best_rounds = sweep_results[best_config]["rounds"]

torch.manual_seed(SEED)
global_model = MFQueryRouter(
    q_dim=q_dim,
    n_models=K_MODELS,
    d_embed=D_EMBED,
    noise_alpha=NOISE_ALPHA,
    scaler=scaler,
).to(DEVICE)
global_model.load_state_dict({k: v.to(DEVICE) for k, v in best_fed_state.items()}, strict=True)
server = FedAvgServer(global_model)
server.history = sweep_server.history if best_config == key else {
    "round": list(range(best_rounds + 1)),
    "global_val_auc": sweep_results[best_config]["auc_history"],
}


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
    lr: float = CENTRALIZED_LR,
    weight_decay: float = CENTRALIZED_WD,
    early_stop_patience: Optional[int] = None,
):
    torch.manual_seed(SEED)
    model = MFQueryRouter(
        q_dim=q_dim,
        n_models=n_models,
        d_embed=D_EMBED,
        noise_alpha=NOISE_ALPHA,
        scaler=scaler,
    ).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_auc, best_state = -1.0, None
    patience = early_stop_patience
    auc_history = []

    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {k: (v.to(DEVICE) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            losses = model.masked_losses(batch, w_acc=W_ACC, w_cost=W_COST)
            opt.zero_grad(set_to_none=True)
            losses["L_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        _, auc_ep = evaluate_curve(model, global_test_loader, LAMBDA_GRID)
        auc_history.append(float(auc_ep))
        print(f"[Centralized][Epoch {ep:03d}] AUC(TEST): {auc_ep:.4f}")

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
    early_stop_patience=None,
)


def train_client_to_convergence_local_only(
    clients: List[Client],
    q_dim: int,
    n_models: int,
    scaler: PerModelLogZScaler,
    global_test_loader: DataLoader = None,
    epochs: int = CENTRALIZED_EPOCHS,
    lr: float = 3 * CENTRALIZED_LR,
    weight_decay: float = CENTRALIZED_WD,
    early_stop_patience: Optional[int] = None,
    early_stop_on: str = "client",
):
    results = {}
    auc_histories = {}

    for client in clients:
        torch.manual_seed(SEED + client.cid)
        model = MFQueryRouter(
            q_dim=q_dim,
            n_models=n_models,
            d_embed=D_EMBED,
            noise_alpha=NOISE_ALPHA,
            scaler=scaler,
        ).to(DEVICE)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        es_loader = client.test_loader if early_stop_on == "client" else global_test_loader
        best_auc, best_state = -1.0, None
        patience = early_stop_patience
        auc_hist = []

        for ep in range(1, epochs + 1):
            model.train()
            for batch in client.train_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor): batch[k] = v.to(DEVICE)
                losses = model.masked_losses(batch, w_acc=W_ACC, w_cost=W_COST)
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

local_only_models, local_only_auc_hists = train_client_to_convergence_local_only(
    clients, q_dim=q_dim, n_models=K, scaler=scaler,
    global_test_loader=global_test_loader,
    epochs=CENTRALIZED_EPOCHS + 7,
    lr=3 * CENTRALIZED_LR,
    weight_decay=CENTRALIZED_WD,
    early_stop_patience=None,
    early_stop_on="client",
)


def personalize_clients_from_global(
    server: FedAvgServer,
    clients: List[Client],
    local_epochs: int = 1,
    lr: float = LR_LOCAL,
    weight_decay: float = WEIGHT_DECAY,
) -> Dict[int, MFQueryRouter]:
    personalized: Dict[int, MFQueryRouter] = {}
    for c in clients:
        model = MFQueryRouter(
            q_dim=server.model.q_dim,
            n_models=server.model.n_models,
            d_embed=server.model.d_embed,
            noise_alpha=server.model.noise_alpha,
            scaler=server.model.scaler,
        ).to(DEVICE)
        model.load_state_dict(server.model.state_dict(), strict=True)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        model.train()
        for _ in range(local_epochs):
            for batch in c.train_loader:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(DEVICE)
                losses = model.masked_losses(batch, w_acc=W_ACC, w_cost=W_COST)
                opt.zero_grad(set_to_none=True)
                losses["L_total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        model.eval()
        personalized[c.cid] = model
        print(f"[Personalized] Client {c.cid} — finished {local_epochs} local epoch(s).")
    return personalized

personalized_models = personalize_clients_from_global(
    server, clients,
    local_epochs=1,
    lr=LR_LOCAL / 2,
    weight_decay=WEIGHT_DECAY,
)

ensemble_models: Dict[int, WeightedEnsembleRouter] = {}
for cid, local_model in local_only_models.items():
    ensemble = WeightedEnsembleRouter(
        server.model,
        local_model,
        w_global=0.5,
        w_local=0.5,
    )
    ensemble.to(DEVICE)
    ensemble.eval()
    ensemble_models[cid] = ensemble

print(f"Built {len(ensemble_models)} global+local ensembles (0.5, 0.5 weights).")


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
    }

per_client_on_global = {}
for cid in client_ids_all:
    per_client_on_global[str(cid)] = {
        "local_only": _eval_curve(local_only_models[cid], global_test_loader),
        "personalized": _eval_curve(personalized_models[cid], global_test_loader),
        "ensemble": _eval_curve(ensemble_models[cid], global_test_loader),
    }

per_client_model_active_samples = {}
for cid in client_ids_all:
    client = clients[cid]
    train_ds = getattr(client, "train_ds", None)
    if train_ds is None:
        per_client_model_active_samples[str(cid)] = []
        continue
    mask = None
    if hasattr(train_ds, "Macc"):
        mask = train_ds.Macc
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            sample_counts = mask.detach().cpu().numpy().sum(axis=0).astype(int).tolist()
        else:
            sample_counts = np.array(mask).sum(axis=0).astype(int).tolist()
    else:
        sample_counts = []
    per_client_model_active_samples[str(cid)] = sample_counts

results = {
    "method": "mf_embedllm",
    "sweep_results": {k: _to_serializable(v) for k, v in sweep_results.items()},
    "best_config": best_config,
    "lambda_grid": _to_serializable(LAMBDA_GRID),
    "config": {
        "D_EMBED": D_EMBED,
        "NOISE_ALPHA": NOISE_ALPHA,
        "BATCH_SIZE": BATCH_SIZE,
        "LR_LOCAL": LR_LOCAL,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "LOCAL_EPOCHS": LOCAL_EPOCHS,
        "MAX_ROUNDS": MAX_ROUNDS,
        "PARTICIPATION_FRAC": PARTICIPATION_FRAC,
        "CENTRALIZED_LR": CENTRALIZED_LR,
        "CENTRALIZED_WD": CENTRALIZED_WD,
        "CENTRALIZED_EPOCHS": CENTRALIZED_EPOCHS,
        "N_CLIENTS": N_CLIENTS,
        "DIRICHLET_ALPHA": DIRICHLET_ALPHA,
        "MODEL_DIRICHLET_ALPHA": MODEL_DIRICHLET_ALPHA,
        "LABEL_KEEP_FRAC": LABEL_KEEP_FRAC,
        "CLIENT_TRAIN_FRAC": CLIENT_TRAIN_FRAC,
        "SEED": SEED,
        "best_lr": best_lr,
        "best_rounds": best_rounds,
    },
    "dataset": {
        "num_samples": int(N),
        "num_models": int(K),
        "model_names": MODEL_NAMES,
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
    "per_client_model_active_samples": per_client_model_active_samples,
}

out_json = os.path.join(OUTPUT_DIR, "mf_experiment_results.json")
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)
print("Saved experiment results to", out_json)
