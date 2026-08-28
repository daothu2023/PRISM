import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from model import EmbeddingFusionModel
from utils import (
    compute_auprc,
    detect_gene_column,
    read_gene_txt,
    seed_everything,
    zscore_np,
)


# ============================================================
# 0) CONFIGURATION
# ============================================================
ROOT = "..."
TARGET = "BRCA"

CANCERS = [
    "BRCA",
    "BLCA",
    "LUAD",
    "LIHC",
    "LUSC",
    "THCA",
    "ESCA",
    "PRAD",
    "STAD",
    "COAD",
    "CESC",
]

CPDB_PATH = f"{ROOT}/CPDB_PPI_FULL.csv"
NEGATIVE_GENE_PATH = f"{ROOT}/2187false.txt"

OUTPUT_DIR = f"{ROOT}/OFFLINE_TARGET_AUX_PRETRAIN_{TARGET}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRETRAIN_SEED = 42
PRETRAIN_EPOCHS = 150
CHECKPOINT_EPOCH = 150

HIDDEN_DIM = 128
OUT_DIM = 128
NODE_CLS_HIDDEN = 128
DROPOUT = 0.5
SELF_LOOPS = True

MAX_SCRNA_GRAPHS_PER_EPOCH = 8
MAX_SCRNA_GRAPHS_FOR_EVAL = 8

PRETRAIN_LR = 5e-4
PRETRAIN_WEIGHT_DECAY = 5e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 1) DATA LOADING
# ============================================================
def load_cpdb():
    print("\n[INFO] Loading the CPDB PPI network ...")

    cpdb_df = pd.read_csv(CPDB_PATH)

    if not {"u", "v"}.issubset(cpdb_df.columns):
        raise ValueError(
            "The CPDB file must contain 'u' and 'v' columns."
        )

    cpdb_df["u"] = cpdb_df["u"].astype(str)
    cpdb_df["v"] = cpdb_df["v"].astype(str)

    cpdb_gene_set = set(cpdb_df["u"]) | set(cpdb_df["v"])
    return cpdb_df, cpdb_gene_set


def load_target_features(cpdb_gene_set):
    print(f"\n[INFO] Loading omics features for {TARGET} ...")

    feature_path = f"{ROOT}/multiomics_features_{TARGET}.csv"
    feature_df = pd.read_csv(feature_path)

    gene_column = detect_gene_column(feature_df)
    feature_df[gene_column] = feature_df[gene_column].astype(str)
    feature_df = feature_df.drop_duplicates(
        subset=[gene_column]
    ).reset_index(drop=True)

    feature_columns = [
        column
        for column in feature_df.columns
        if column != gene_column
    ]

    feature_values = (
        feature_df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )

    feature_values = zscore_np(feature_values)
    feature_genes = feature_df[gene_column].tolist()

    gene_universe = sorted(
        cpdb_gene_set & set(feature_genes)
    )

    if not gene_universe:
        raise RuntimeError(
            "The target-specific gene universe is empty."
        )

    gene_to_row = {
        gene: row
        for row, gene in enumerate(feature_genes)
    }

    aligned_rows = [
        gene_to_row[gene]
        for gene in gene_universe
    ]

    x_target = torch.tensor(
        feature_values[aligned_rows],
        dtype=torch.float,
    )

    gene_to_idx = {
        gene: idx
        for idx, gene in enumerate(gene_universe)
    }

    input_dim = x_target.shape[1]

    print(
        f"[INFO] {TARGET}: {len(feature_genes)} feature genes, "
        f"{input_dim} features"
    )
    print(
        f"[INFO] Gene universe: {len(gene_universe)} genes"
    )

    return x_target, gene_universe, gene_to_idx, input_dim


def build_cpdb_graph(cpdb_df, gene_to_idx):
    filtered = cpdb_df[
        cpdb_df["u"].isin(gene_to_idx)
        & cpdb_df["v"].isin(gene_to_idx)
    ].copy()

    source = filtered["u"].map(gene_to_idx).to_numpy()
    target = filtered["v"].map(gene_to_idx).to_numpy()

    edge_index = torch.tensor(
        np.vstack([source, target]),
        dtype=torch.long,
    )

    print(f"\n[INFO] CPDB edges: {len(filtered)}")

    return {
        "graph_type": "cpdb",
        "graph_id": f"{TARGET}_CPDB",
        "edge_index": edge_index,
        "present_nodes": torch.arange(
            len(gene_to_idx),
            dtype=torch.long,
        ),
        "path": CPDB_PATH,
    }


def load_scrna_patient_graph(csv_path, gene_to_idx):
    graph_df = pd.read_csv(csv_path)

    if not {"u", "v"}.issubset(graph_df.columns):
        raise ValueError(
            f"{csv_path} must contain 'u' and 'v' columns."
        )

    graph_df["u"] = graph_df["u"].astype(str)
    graph_df["v"] = graph_df["v"].astype(str)

    graph_df = graph_df[
        graph_df["u"].isin(gene_to_idx)
        & graph_df["v"].isin(gene_to_idx)
    ].copy()

    graph_df = graph_df[
        graph_df["u"] != graph_df["v"]
    ].copy()

    if graph_df.empty:
        return None

    source = (
        graph_df["u"]
        .map(gene_to_idx)
        .to_numpy(dtype=np.int64)
    )

    target = (
        graph_df["v"]
        .map(gene_to_idx)
        .to_numpy(dtype=np.int64)
    )

    edge_index_np = np.vstack([
        np.concatenate([source, target]),
        np.concatenate([target, source]),
    ])

    present_nodes = np.unique(
        np.concatenate([source, target])
    ).astype(np.int64)

    graph_id = (
        os.path.basename(csv_path)
        .replace("_scRNA_graph_undirected.csv", "")
        .replace(".csv", "")
    )

    return {
        "graph_type": "scrna",
        "graph_id": graph_id,
        "edge_index": torch.tensor(
            edge_index_np,
            dtype=torch.long,
        ),
        "present_nodes": torch.tensor(
            present_nodes,
            dtype=torch.long,
        ),
        "path": csv_path,
    }


def build_target_graphs(cpdb_graph, gene_to_idx):
    graph_dir = os.path.join(
        ROOT,
        f"{TARGET}_scRNA_patient_graphs",
        "cpdb_like_undirected",
    )

    if not os.path.isdir(graph_dir):
        raise FileNotFoundError(
            f"scRNA graph directory not found: {graph_dir}"
        )

    csv_paths = sorted(
        os.path.join(graph_dir, filename)
        for filename in os.listdir(graph_dir)
        if filename.endswith(".csv")
    )

    if not csv_paths:
        raise RuntimeError(
            f"No scRNA graph CSV files were found in: {graph_dir}"
        )

    graphs = [cpdb_graph]

    for csv_path in csv_paths:
        graph = load_scrna_patient_graph(
            csv_path,
            gene_to_idx,
        )

        if graph is not None:
            graphs.append(graph)

    n_scrna = len(graphs) - 1

    if n_scrna == 0:
        raise RuntimeError(
            f"No valid scRNA graphs were loaded for {TARGET}."
        )

    print(
        f"\n[INFO] {TARGET}: CPDB=1 | "
        f"scRNA={n_scrna} | total={len(graphs)}"
    )

    return graphs


# ============================================================
# 2) AUXILIARY PRETRAIN LABELS
# ============================================================
def build_pretrain_labels(gene_universe, gene_to_idx):
    target_positive_genes = set(
        read_gene_txt(
            f"{ROOT}/{TARGET}true.txt"
        )
    )

    target_negative_genes = set(
        read_gene_txt(
            NEGATIVE_GENE_PATH
        )
    )

    auxiliary_positive_set = set()

    for cancer in CANCERS:
        if cancer == TARGET:
            continue

        positive_path = f"{ROOT}/{cancer}true.txt"

        if not os.path.exists(positive_path):
            raise FileNotFoundError(
                f"Auxiliary positive-label file not found: "
                f"{positive_path}"
            )

        auxiliary_positive_set.update(
            read_gene_txt(positive_path)
        )

    auxiliary_positive_set -= target_positive_genes

    auxiliary_positive_genes = sorted(
        gene
        for gene in auxiliary_positive_set
        if gene in gene_to_idx
    )

    if not auxiliary_positive_genes:
        raise RuntimeError(
            "No valid auxiliary positive genes were found."
        )

    excluded_from_negative = (
        target_positive_genes
        | target_negative_genes
        | set(auxiliary_positive_genes)
    )

    unlabeled_negative_pool = sorted(
        gene
        for gene in gene_universe
        if gene not in excluded_from_negative
    )

    n_positive = len(auxiliary_positive_genes)

    if len(unlabeled_negative_pool) < n_positive:
        raise RuntimeError(
            f"The unlabeled negative pool contains "
            f"{len(unlabeled_negative_pool)} genes, "
            f"but {n_positive} are required."
        )

    rng = np.random.default_rng(PRETRAIN_SEED)

    sampled_negative_genes = sorted(
        rng.choice(
            unlabeled_negative_pool,
            size=n_positive,
            replace=False,
        ).tolist()
    )

    positive_idx = torch.tensor(
        [
            gene_to_idx[gene]
            for gene in auxiliary_positive_genes
        ],
        dtype=torch.long,
        device=DEVICE,
    )

    negative_idx = torch.tensor(
        [
            gene_to_idx[gene]
            for gene in sampled_negative_genes
        ],
        dtype=torch.long,
        device=DEVICE,
    )

    pretrain_idx = torch.cat([
        positive_idx,
        negative_idx,
    ])

    pretrain_labels = torch.cat([
        torch.ones(
            n_positive,
            dtype=torch.float,
            device=DEVICE,
        ),
        torch.zeros(
            n_positive,
            dtype=torch.float,
            device=DEVICE,
        ),
    ])

    print(
        f"\n[INFO] Auxiliary pretraining labels: "
        f"positive={n_positive} | negative={n_positive}"
    )

    return (
        auxiliary_positive_genes,
        sampled_negative_genes,
        pretrain_idx,
        pretrain_labels,
    )


# ============================================================
# 3) GRAPH SELECTION AND FORWARD PASS
# ============================================================
def select_scrna_graphs(
    target_graphs,
    epoch,
    training,
):
    scrna_graphs = [
        graph
        for graph in target_graphs
        if graph["graph_type"] == "scrna"
    ]

    max_graphs = (
        MAX_SCRNA_GRAPHS_PER_EPOCH
        if training
        else MAX_SCRNA_GRAPHS_FOR_EVAL
    )

    if (
        max_graphs is None
        or len(scrna_graphs) <= max_graphs
    ):
        return scrna_graphs

    if training:
        rng = np.random.default_rng(
            PRETRAIN_SEED + epoch * 1009
        )
        selected = rng.choice(
            len(scrna_graphs),
            size=max_graphs,
            replace=False,
        )
        return [
            scrna_graphs[int(index)]
            for index in sorted(selected)
        ]

    return scrna_graphs[:max_graphs]


def build_graph_list(
    target_graphs,
    epoch,
    training,
):
    cpdb_graphs = [
        graph
        for graph in target_graphs
        if graph["graph_type"] == "cpdb"
    ]

    if len(cpdb_graphs) != 1:
        raise RuntimeError(
            "Exactly one CPDB graph is required."
        )

    return [
        cpdb_graphs[0],
        *select_scrna_graphs(
            target_graphs,
            epoch,
            training,
        ),
    ]


def graph_to_device(graph):
    return graph["edge_index"].to(
        DEVICE,
        non_blocking=True,
    )


def forward_embedding_fusion(
    model,
    x,
    graphs,
    num_genes,
    out_dim,
):
    cpdb_graphs = [
        graph
        for graph in graphs
        if graph["graph_type"] == "cpdb"
    ]

    scrna_graphs = [
        graph
        for graph in graphs
        if graph["graph_type"] == "scrna"
    ]

    if len(cpdb_graphs) != 1:
        raise RuntimeError(
            "Exactly one CPDB graph is required."
        )

    cpdb_edge_index = graph_to_device(
        cpdb_graphs[0]
    )

    h_cpdb = model.encode(
        x,
        cpdb_edge_index,
    )

    embedding_sum = torch.zeros(
        (num_genes, out_dim),
        dtype=x.dtype,
        device=DEVICE,
    )

    embedding_count = torch.zeros(
        num_genes,
        dtype=x.dtype,
        device=DEVICE,
    )

    for graph in scrna_graphs:
        sc_edge_index = graph_to_device(graph)

        h_scrna = model.encode(
            x,
            sc_edge_index,
        )

        present_idx = graph[
            "present_nodes"
        ].to(
            DEVICE,
            non_blocking=True,
        )

        embedding_sum = embedding_sum.index_add(
            0,
            present_idx,
            h_scrna[present_idx],
        )

        embedding_count = embedding_count.index_add(
            0,
            present_idx,
            torch.ones(
                present_idx.numel(),
                dtype=x.dtype,
                device=DEVICE,
            ),
        )

    h_scrna_mean = (
        embedding_sum
        / embedding_count
        .clamp_min(1.0)
        .unsqueeze(1)
    )

    no_scrna_mask = embedding_count == 0

    if no_scrna_mask.any():
        h_scrna_mean = h_scrna_mean.masked_fill(
            no_scrna_mask.unsqueeze(1),
            0.0,
        )

    logits = model.classify(
        h_cpdb,
        h_scrna_mean,
        x,
    )

    return logits


# ============================================================
# 4) CHECKPOINTING
# ============================================================
def save_checkpoint(
    model,
    input_dim,
    auxiliary_positive_genes,
    sampled_negative_genes,
):
    checkpoint_path = os.path.join(
        OUTPUT_DIR,
        f"target_aux_pretrain_epoch_{CHECKPOINT_EPOCH}.pt",
    )

    checkpoint_payload = {
        "target": TARGET,
        "saved_epoch": CHECKPOINT_EPOCH,
        "pretrain_seed": PRETRAIN_SEED,
        "input_dim": input_dim,
        "hidden_dim": HIDDEN_DIM,
        "out_dim": OUT_DIM,
        "node_cls_hidden": NODE_CLS_HIDDEN,
        "dropout": DROPOUT,
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "positive_genes": auxiliary_positive_genes,
        "negative_genes": sampled_negative_genes,
        "negative_sampling": "fixed_once_from_unlabeled",
        "architecture": "cpdb_scrna_embedding_fusion",
        "self_loops": SELF_LOOPS,
        "max_scrna_graphs_per_epoch":
            MAX_SCRNA_GRAPHS_PER_EPOCH,
        "max_scrna_graphs_for_eval":
            MAX_SCRNA_GRAPHS_FOR_EVAL,
    }

    torch.save(
        checkpoint_payload,
        checkpoint_path,
    )

    print(f"\n[INFO] Saved checkpoint: {checkpoint_path}")


# ============================================================
# 5) PRETRAINING
# ============================================================
def run_offline_pretraining():
    seed_everything(PRETRAIN_SEED)

    print("DEVICE:", DEVICE)

    cpdb_df, cpdb_gene_set = load_cpdb()

    (
        x_target,
        gene_universe,
        gene_to_idx,
        input_dim,
    ) = load_target_features(cpdb_gene_set)

    cpdb_graph = build_cpdb_graph(
        cpdb_df,
        gene_to_idx,
    )

    target_graphs = build_target_graphs(
        cpdb_graph,
        gene_to_idx,
    )

    (
        auxiliary_positive_genes,
        sampled_negative_genes,
        pretrain_idx,
        pretrain_labels,
    ) = build_pretrain_labels(
        gene_universe,
        gene_to_idx,
    )

    x_target = x_target.to(DEVICE)

    model = EmbeddingFusionModel(
        input_dim,
        HIDDEN_DIM,
        OUT_DIM,
        NODE_CLS_HIDDEN,
        DROPOUT,
        self_loops=SELF_LOOPS,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=PRETRAIN_LR,
        weight_decay=PRETRAIN_WEIGHT_DECAY,
    )

    print(
        f"\n{'=' * 82}\n"
        f"[OFFLINE PRETRAINING — {TARGET}]\n"
        f"{'=' * 82}"
    )

    for epoch in range(
        1,
        PRETRAIN_EPOCHS + 1,
    ):
        model.train()
        optimizer.zero_grad()

        training_graphs = build_graph_list(
            target_graphs,
            epoch=epoch,
            training=True,
        )

        logits = forward_embedding_fusion(
            model,
            x_target,
            training_graphs,
            num_genes=len(gene_universe),
            out_dim=OUT_DIM,
        )

        loss = F.binary_cross_entropy_with_logits(
            logits[pretrain_idx],
            pretrain_labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        if (
            epoch == 1
            or epoch % 10 == 0
            or epoch == PRETRAIN_EPOCHS
        ):
            model.eval()

            with torch.no_grad():
                evaluation_graphs = build_graph_list(
                    target_graphs,
                    epoch=0,
                    training=False,
                )

                evaluation_logits = (
                    forward_embedding_fusion(
                        model,
                        x_target,
                        evaluation_graphs,
                        num_genes=len(gene_universe),
                        out_dim=OUT_DIM,
                    )
                )

                probability = torch.sigmoid(
                    evaluation_logits[pretrain_idx]
                ).cpu().numpy()

            auprc = compute_auprc(
                pretrain_labels.cpu().numpy(),
                probability,
            )

            print(
                f"  [E{epoch:03d}] "
                f"loss={loss.item():.4f} | "
                f"AUPRC={auprc:.4f}"
            )

    save_checkpoint(
        model,
        input_dim,
        auxiliary_positive_genes,
        sampled_negative_genes,
    )

    print("\n[INFO] Offline pretraining completed.")


if __name__ == "__main__":
    run_offline_pretraining()
