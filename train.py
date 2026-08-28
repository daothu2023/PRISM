
# ============================================================
# FINE-TUNE FROM ONE OFFLINE PRETRAIN CHECKPOINT
# CPDB + scRNA EMBEDDING FUSION
# ============================================================
#
# For each run and fold:
#   - instantiate a new model;
#   - restore the same pretrained checkpoint;
#   - load both the shared GCN encoder and fusion MLP;
#   - fine-tune on the target-cancer training split;
#   - select the best checkpoint using the validation split;
#   - evaluate once on the held-out test split.
# ============================================================

import os
import gc

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F

from model import EmbeddingFusionModel
from utils import (
    compute_auprc,
    detect_gene_column,
    make_train_val_split,
    read_gene_txt,
    seed_everything,
    zscore_np,
)

from sklearn.model_selection import StratifiedKFold

# ============================================================
# 0) CONFIGURATION
# ============================================================
ROOT = "..."
TARGET = "BRCA"

PRETRAIN_CHECKPOINT_EPOCH = 150

if PRETRAIN_CHECKPOINT_EPOCH not in {100, 150}:
    raise ValueError(
        "PRETRAIN_CHECKPOINT_EPOCH must be either 100 or 150."
    )

PRETRAIN_CHECKPOINT = (
    f"{ROOT}/OFFLINE_TARGET_AUX_PRETRAIN_{TARGET}/"
    f"target_aux_pretrain_epoch_"
    f"{PRETRAIN_CHECKPOINT_EPOCH}.pt"
)

CPDB_PATH = f"{ROOT}/CPDB_PPI_FULL.csv"
NEG_TXT = f"{ROOT}/2187false.txt"

# ============================================================
# 1) HYPERPARAMETERS
# ============================================================
N_RUNS = 10
N_FOLDS = 5

SPLIT_SEED = 42

BASE_FINETUNE_SEED = 2026

HIDDEN_DIM = 128
OUT_DIM = 128
NODE_CLS_HIDDEN = 128
DROPOUT = 0.5

SELF_LOOPS = True

# --- Direct end-to-end training ---
MAX_SCRNA_GRAPHS_PER_EPOCH = 8
MAX_SCRNA_GRAPHS_FOR_EVAL = 8

FT_LR = 1e-3
FT_WEIGHT_DECAY = 1e-3
FT_MAX_EPOCHS = 500
FT_PATIENCE = 80

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("DEVICE:", DEVICE)

# ============================================================
# 3) LOAD PPI + NEG LABELS
# ============================================================
print("\n[INFO] Loading the CPDB PPI network ...")
cpdb_df = pd.read_csv(CPDB_PATH)
if not {"u", "v"}.issubset(cpdb_df.columns):
    raise ValueError(f"CPDB file must contain u and v columns: {CPDB_PATH}")
cpdb_df["u"] = cpdb_df["u"].astype(str)
cpdb_df["v"] = cpdb_df["v"].astype(str)
cpdb_gene_set = set(cpdb_df["u"]) | set(cpdb_df["v"])
neg_genes = set(read_gene_txt(NEG_TXT))

# ============================================================
# 4) LOAD TARGET OMICS FEATURES
# ============================================================
print(f"\n[INFO] Loading omics features for {TARGET} ...")
feature_df = pd.read_csv(f"{ROOT}/multiomics_features_{TARGET}.csv")
gene_col = detect_gene_column(feature_df)
feature_df[gene_col] = feature_df[gene_col].astype(str)
feature_df = feature_df.drop_duplicates(subset=[gene_col]).reset_index(drop=True)
feature_cols = [column for column in feature_df.columns if column != gene_col]
feature_values = (
    feature_df[feature_cols]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0.0)
    .to_numpy(dtype=np.float32)
)
feature_values = zscore_np(feature_values)
feature_genes = feature_df[gene_col].tolist()

universe = cpdb_gene_set & set(feature_genes)
global_genes = sorted(universe)
gene_to_idx = {gene: index for index, gene in enumerate(global_genes)}
N = len(global_genes)
IN_DIM = feature_values.shape[1]
print(f"[INFO] {TARGET}: {len(feature_genes)} feature genes, {IN_DIM} features")
print(f"[INFO] Gene universe: {N} genes")

# ============================================================
# 5) BUILD TARGET FEATURE TENSOR AND LABELS
# ============================================================
feature_row = {gene: index for index, gene in enumerate(feature_genes)}
feature_indices = [feature_row[gene] for gene in global_genes]
X_target = torch.tensor(feature_values[feature_indices], dtype=torch.float)

positive_genes = set(read_gene_txt(f"{ROOT}/{TARGET}true.txt"))
negative_genes = neg_genes - (positive_genes & neg_genes)

y_target = torch.full((N,), -1, dtype=torch.long)
for index, gene in enumerate(global_genes):
    if gene in positive_genes:
        y_target[index] = 1
    elif gene in negative_genes:
        y_target[index] = 0

del feature_df, feature_values

gc.collect()

# ============================================================
# 6) BUILD PPI GRAPH
# ============================================================
cpdb_filt  = cpdb_df[
    cpdb_df["u"].isin(gene_to_idx) & cpdb_df["v"].isin(gene_to_idx)
].copy()
c_src      = cpdb_filt["u"].map(gene_to_idx).values
c_dst      = cpdb_filt["v"].map(gene_to_idx).values
edge_index = torch.tensor(np.vstack([c_src,c_dst]), dtype=torch.long).to(DEVICE)
print(f"\n[INFO] CPDB edges: {len(cpdb_filt)}")

X_target_dev = X_target.to(DEVICE)
y_target_dev = y_target.to(DEVICE).float()

# ============================================================
# 7) BUILD MULTI-GRAPH COLLECTIONS
# ============================================================
SCRNA_ROOT_DIR = f"{ROOT}/{TARGET}_scRNA_patient_graphs"

def load_scrna_patient_graph(csv_path):
    df = pd.read_csv(csv_path)

    if not {"u", "v"}.issubset(
        df.columns
    ):
        raise ValueError(
            f"Missing required u/v columns in {csv_path}."
        )

    df["u"] = df["u"].astype(str)
    df["v"] = df["v"].astype(str)

    df = df[
        df["u"].isin(gene_to_idx)
        & df["v"].isin(gene_to_idx)
    ].copy()

    df = df[
        df["u"] != df["v"]
    ].copy()

    if len(df) == 0:
        return None

    src = (
        df["u"]
        .map(gene_to_idx)
        .to_numpy(dtype=np.int64)
    )

    dst = (
        df["v"]
        .map(gene_to_idx)
        .to_numpy(dtype=np.int64)
    )

    # Input graphs are undirected; GCNConv requires both edge directions.
    edge_index_np = np.vstack([
        np.concatenate([src, dst]),
        np.concatenate([dst, src])
    ])

    graph_id = (
        os.path.basename(csv_path)
        .replace(
            "_scRNA_graph_undirected.csv",
            ""
        )
        .replace(".csv", "")
    )

    present_nodes = np.unique(
        np.concatenate([src, dst])
    ).astype(np.int64)

    return {
        "graph_type": "scrna",
        "graph_id": graph_id,
        "edge_index": torch.tensor(
            edge_index_np,
            dtype=torch.long
        ),
        "present_nodes": torch.tensor(
            present_nodes,
            dtype=torch.long
        ),
        "path": csv_path
    }

print(f"\n[INFO] Building CPDB and scRNA graphs for {TARGET} ...")

TARGET_GRAPHS = [{
    "graph_type": "cpdb",
    "graph_id": f"{TARGET}_CPDB",
    "edge_index": edge_index.detach().cpu(),
    "present_nodes": torch.arange(N, dtype=torch.long),
    "path": CPDB_PATH,
}]

sc_root = SCRNA_ROOT_DIR
n_scrna = 0

if sc_root is not None:
    graph_dir = os.path.join(sc_root, "cpdb_like_undirected")

    if not os.path.isdir(graph_dir):
        print(
            f"    [WARNING] scRNA directory not found for {TARGET}: "
            f"{graph_dir}. The model will use CPDB only."
        )
        graph_dir = None

    csv_paths = (
        sorted(
            os.path.join(graph_dir, filename)
            for filename in os.listdir(graph_dir)
            if filename.endswith(".csv")
        )
        if graph_dir is not None
        else []
    )

    for csv_path in csv_paths:
        try:
            graph = load_scrna_patient_graph(csv_path)
            if graph is not None:
                TARGET_GRAPHS.append(graph)
                n_scrna += 1
        except Exception as exc:
            print(f"    [SKIP] {csv_path}: {exc}")
else:
    print(f"    [INFO] {TARGET} has no configured scRNA directory; using CPDB only.")

print(f"[INFO] {TARGET}: CPDB=1 | scRNA={n_scrna} | total={len(TARGET_GRAPHS)}")

def select_scrna_graphs_for_target(epoch, training):
    scrna_graphs = [
        graph for graph in TARGET_GRAPHS
        if graph["graph_type"] == "scrna"
    ]
    max_graphs = (
        MAX_SCRNA_GRAPHS_PER_EPOCH
        if training else MAX_SCRNA_GRAPHS_FOR_EVAL
    )
    if max_graphs is None or len(scrna_graphs) <= max_graphs:
        return scrna_graphs
    if training:
        rng = np.random.default_rng(SPLIT_SEED + epoch * 1009)
        chosen = rng.choice(len(scrna_graphs), size=max_graphs, replace=False)
        return [scrna_graphs[int(i)] for i in sorted(chosen)]
    return scrna_graphs[:max_graphs]

def get_cpdb_graph():
    for graph in TARGET_GRAPHS:
        if graph["graph_type"] == "cpdb":
            return graph
    raise RuntimeError(f"CPDB graph not found for {TARGET}.")

def graph_to_device(graph):
    return graph["edge_index"].to(
        DEVICE,
        non_blocking=True
    )

# ============================================================
# 9) MASKED-MEAN scRNA EMBEDDING FUSION
# ============================================================
def build_target_graph_list(
    epoch,
    training
):
    return (
        [get_cpdb_graph()]
        + select_scrna_graphs_for_target(
            epoch=epoch,
            training=training
        )
    )

def forward_embedding_fusion(model, x, graphs):
    cpdb_graphs = [
        graph for graph in graphs
        if graph["graph_type"] == "cpdb"
    ]
    scrna_graphs = [
        graph for graph in graphs
        if graph["graph_type"] == "scrna"
    ]

    if len(cpdb_graphs) != 1:
        raise RuntimeError(
            f"Expected exactly one CPDB graph, received {len(cpdb_graphs)}."
        )

    cpdb_edge_index = graph_to_device(cpdb_graphs[0])
    h_cpdb = model.encode(x, cpdb_edge_index)

    embedding_sum = torch.zeros(
        (N, OUT_DIM), dtype=x.dtype, device=DEVICE
    )
    embedding_count = torch.zeros(N, dtype=x.dtype, device=DEVICE)

    for graph in scrna_graphs:
        sc_edge_index = graph_to_device(graph)
        h_scrna = model.encode(x, sc_edge_index)
        present_idx = graph["present_nodes"].to(DEVICE, non_blocking=True)

        embedding_sum.index_add_(0, present_idx, h_scrna[present_idx])
        embedding_count.index_add_(
            0,
            present_idx,
            torch.ones(
                present_idx.numel(), dtype=x.dtype, device=DEVICE
            ),
        )

    h_scrna_mean = embedding_sum / embedding_count.clamp_min(1.0).unsqueeze(1)
    h_scrna_mean = h_scrna_mean.masked_fill(
        (embedding_count == 0).unsqueeze(1), 0.0
    )

    logits = model.classify(h_cpdb, h_scrna_mean, x)
    return logits, embedding_count

# ============================================================
# 10) DIRECT END-TO-END TRAINING PER FOLD
# ============================================================

# ============================================================
# LOAD OFFLINE PRETRAIN CHECKPOINT
# ============================================================
if not os.path.exists(
    PRETRAIN_CHECKPOINT
):
    raise FileNotFoundError(
        f"Checkpoint not found:\n"
        f"{PRETRAIN_CHECKPOINT}\n"
        "Run the pretraining script first."
    )

PRETRAIN_STATE = torch.load(
    PRETRAIN_CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

for required_key in [
    "target",
    "saved_epoch",
    "input_dim",
    "hidden_dim",
    "out_dim",
    "node_cls_hidden",
    "model_state_dict"
]:
    if required_key not in PRETRAIN_STATE:
        raise KeyError(
            f"Checkpoint is missing key: "
            f"{required_key}"
        )

if PRETRAIN_STATE[
    "target"
] != TARGET:
    raise ValueError(
        "Checkpoint target does not match TARGET."
    )

if int(
    PRETRAIN_STATE[
        "saved_epoch"
    ]
) != PRETRAIN_CHECKPOINT_EPOCH:
    raise ValueError(
        "Checkpoint epoch does not match PRETRAIN_CHECKPOINT_EPOCH."
    )

if int(
    PRETRAIN_STATE[
        "input_dim"
    ]
) != IN_DIM:
    raise ValueError(
        "Checkpoint input_dim does not match IN_DIM."
    )

if int(
    PRETRAIN_STATE[
        "hidden_dim"
    ]
) != HIDDEN_DIM:
    raise ValueError(
        "Checkpoint hidden_dim does not match HIDDEN_DIM."
    )

if int(
    PRETRAIN_STATE[
        "out_dim"
    ]
) != OUT_DIM:
    raise ValueError(
        "Checkpoint out_dim does not match OUT_DIM."
    )

if int(
    PRETRAIN_STATE[
        "node_cls_hidden"
    ]
) != NODE_CLS_HIDDEN:
    raise ValueError(
        "Checkpoint node_cls_hidden does not match NODE_CLS_HIDDEN."
    )

PRETRAIN_MODEL_STATE = (
    PRETRAIN_STATE[
        "model_state_dict"
    ]
)

print(
    f"\n[INFO] Loaded checkpoint: "
    f"{PRETRAIN_CHECKPOINT}"
)

def train_end_to_end_fusion(
    train_idx,
    val_idx,
    test_idx,
    run_id,
    fold_id,
):
    print(
        f"\n  {'=' * 78}\n"
        f"  [END-TO-END EMBEDDING FUSION] {TARGET} "
        f"(Run {run_id:02d} Fold {fold_id:02d})\n"
        f"  {'=' * 78}"
    )

    model = EmbeddingFusionModel(
        IN_DIM,
        HIDDEN_DIM,
        OUT_DIM,
        NODE_CLS_HIDDEN,
        DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(PRETRAIN_MODEL_STATE)

    print(
        f"    Loaded Shared GCN and Fusion MLP from pretraining epoch "
        f"{PRETRAIN_CHECKPOINT_EPOCH}."
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=FT_LR,
        weight_decay=FT_WEIGHT_DECAY,
    )

    y_np = y_target.numpy()
    x_target = X_target_dev
    target_labels = y_target_dev
    train_tensor = torch.as_tensor(
        train_idx, dtype=torch.long, device=DEVICE
    )

    n_pos = int((y_np[train_idx] == 1).sum())
    n_neg = int((y_np[train_idx] == 0).sum())
    pos_weight = torch.tensor(
        min(n_neg / max(n_pos, 1), 10.0),
        dtype=torch.float,
        device=DEVICE,
    )

    best_state = None
    best_val_auprc = -1.0
    patience_counter = 0

    eval_graphs = build_target_graph_list(epoch=0, training=False)

    for epoch in range(1, FT_MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        train_graphs = build_target_graph_list(epoch=epoch, training=True)
        logits, _ = forward_embedding_fusion(model, x_target, train_graphs)
        loss = F.binary_cross_entropy_with_logits(
            logits[train_tensor],
            target_labels[train_tensor],
            pos_weight=pos_weight,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits, _ = forward_embedding_fusion(
                model, x_target, eval_graphs
            )
            eval_prob = torch.sigmoid(eval_logits).cpu().numpy()

        train_auprc = compute_auprc(
            y_np[train_idx], eval_prob[train_idx]
        )
        val_auprc = compute_auprc(
            y_np[val_idx], eval_prob[val_idx]
        )

        current_val_auprc = val_auprc
        if np.isfinite(current_val_auprc) and current_val_auprc > best_val_auprc:
            best_val_auprc = current_val_auprc
            patience_counter = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 50 == 0:
            print(
                f"    [E{epoch:03d}] "
                f"loss={loss.item():.4f} | "
                f"sc_graphs={len(train_graphs) - 1} | "
                f"train_AUPRC={train_auprc:.4f}"
            )

        if patience_counter >= FT_PATIENCE:
            if epoch != 1 and epoch % 50 != 0:
                print(
                    f"    [E{epoch:03d}] "
                    f"loss={loss.item():.4f} | "
                    f"sc_graphs={len(train_graphs) - 1} | "
                    f"train_AUPRC={train_auprc:.4f}"
                )
            print(f"  [FINE-TUNE] Early stopping at epoch {epoch}")
            break

    if best_state is None:
        raise RuntimeError("No valid best model state was recorded.")

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        final_logits, _ = forward_embedding_fusion(
            model, x_target, eval_graphs
        )
        final_prob = torch.sigmoid(final_logits).cpu().numpy()

    test_auprc = compute_auprc(
        y_np[test_idx], final_prob[test_idx]
    )

    print(
        f"  [FOLD RESULT] Run {run_id:02d} | Fold {fold_id:02d} | "
        f"test_AUPRC={test_auprc:.4f}"
    )

    return test_auprc

# ============================================================
# 11) MAIN — FIXED FIVE-FOLD CROSS-VALIDATION × 10 RUNS
# ============================================================
y_target_np = y_target.numpy()
labeled_mask = y_target_np != -1
labeled_idx = np.where(labeled_mask)[0]
y_labeled = y_target_np[labeled_mask]

fixed_skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SPLIT_SEED,
)

fixed_splits = []
for fold, (trainval_rel, test_rel) in enumerate(
    fixed_skf.split(labeled_idx, y_labeled),
    start=1,
):
    trainval_idx = labeled_idx[trainval_rel]
    test_idx = labeled_idx[test_rel]
    train_idx, val_idx = make_train_val_split(
        trainval_idx,
        y_target_np[trainval_idx],
        SPLIT_SEED + fold,
    )
    fixed_splits.append((fold, train_idx, val_idx, test_idx))

run_auprcs = []

for run in range(1, N_RUNS + 1):
    fold_auprcs = []

    print(
        f"\n{'=' * 88}\n"
        f"[RUN {run:02d}/{N_RUNS}] "
        f"reusing the same five folds and pretraining checkpoint\n"
        f"{'=' * 88}"
    )

    for fold, train_idx, val_idx, test_idx in fixed_splits:
        finetune_seed = BASE_FINETUNE_SEED + run * 1000 + fold
        seed_everything(finetune_seed)

        test_auprc = train_end_to_end_fusion(
            train_idx,
            val_idx,
            test_idx,
            run,
            fold,
        )
        fold_auprcs.append(float(test_auprc))

    run_mean = float(np.nanmean(fold_auprcs))
    run_auprcs.append(run_mean)

    fold_text = ", ".join(f"{value:.4f}" for value in fold_auprcs)
    print(
        f"\n[RUN RESULT] Run {run:02d} | "
        f"Fold test AUPRCs=[{fold_text}] | "
        f"Mean test AUPRC={run_mean:.4f}"
    )

run_auprcs = np.asarray(run_auprcs, dtype=float)
final_mean = float(np.nanmean(run_auprcs))
final_std = float(np.nanstd(run_auprcs, ddof=0))

print("\n" + "=" * 88)
print("[FINAL TEST AUPRC OVER 10 RUNS]")
print("=" * 88)

for run_id, run_auprc in enumerate(run_auprcs, start=1):
    print(f"  Run {run_id:02d}: {run_auprc:.4f}")

print("-" * 88)
print(f"  Final test AUPRC: {final_mean:.4f} ± {final_std:.4f}")
print("=" * 88)

