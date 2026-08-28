import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Enforce deterministic behavior for a given fine-tuning seed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_gene_txt(path):
    with open(path, encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]

def detect_gene_column(df):
    for c in [
        "gene", "Gene", "GENE", "symbol", "SYMBOL",
        "gene_name", "GeneSymbol", "node", "Unnamed: 0"
    ]:
        if c in df.columns:
            return c
    for c in df.columns:
        if df[c].dtype == object:
            return c
    raise ValueError("Could not identify the gene column.")

def zscore_np(x):
    mean = x.mean(axis=0, keepdims=True)
    std  = x.std(axis=0,  keepdims=True)
    std[std < 1e-8] = 1.0
    return (x - mean) / std

def compute_auprc(y_true, y_prob):
    """Return the area under the precision-recall curve."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    try:
        return float(average_precision_score(y_true, y_prob))
    except ValueError:
        return np.nan

def make_train_val_split(trainval_idx, y_trainval, seed):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    tr_rel, va_rel = next(sss.split(trainval_idx, y_trainval))
    return trainval_idx[tr_rel], trainval_idx[va_rel]
