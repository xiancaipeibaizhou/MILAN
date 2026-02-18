import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import random
import re
import hashlib
import datetime
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# Sklearn imports
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from torch_geometric.loader import DataLoader 
from torch_geometric.data import Data, Batch
from torch.utils.data import Dataset

# 引入模型 (确保 network_dynamic.py 在同级目录)
from model import MILAN
from hparams_a3 import resolve_hparams
from analys import (
    _attack_best_threshold,
    _collect_attack_scores,
    evaluate_comprehensive_v2,
    evaluate_comprehensive_with_threshold_v2,
)

# ==========================================
# 1. 核心 Utils (直接内置，确保逻辑正确)
# ==========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _as_sequence_batch(batch):
    """处理 DataLoader collate 后的数据结构"""
    if isinstance(batch, (list, tuple)) and len(batch) > 0:
        if isinstance(batch[0], (list, tuple)): return list(batch)
        if hasattr(batch[0], 'edge_index'): return [list(batch)]
    return [batch]

def _seq_collate_fn(batch):
    return batch

def temporal_collate_fn(batch):
    if len(batch) == 0:
        return []
    seq_len = min(len(seq) for seq in batch)
    if seq_len <= 0:
        return []

    batched_seq = []
    for t in range(seq_len):
        graphs_t = [seq[t] for seq in batch]
        batched_seq.append(Batch.from_data_list(graphs_t))
    return batched_seq

class TemporalGraphDataset(Dataset):
    def __init__(self, graph_data_seq, seq_len=1):
        self.graph_data_seq = [g for g in graph_data_seq if g is not None]
        self.seq_len = int(seq_len)
        if self.seq_len < 1: self.seq_len = 1

    def __len__(self):
        return max(0, len(self.graph_data_seq) - self.seq_len + 1)

    def __getitem__(self, idx):
        return self.graph_data_seq[idx : idx + self.seq_len]

def _pad_seq_for_last_frame_coverage(graph_seqs, seq_len):
    if graph_seqs is None or len(graph_seqs) == 0:
        return graph_seqs
    pad_n = max(0, int(seq_len) - 1)
    if pad_n == 0:
        return graph_seqs
    first = graph_seqs[0]
    pads = []
    for _ in range(pad_n):
        pads.append(first.clone() if hasattr(first, "clone") else first)
    return pads + list(graph_seqs)

def create_graph_data_global_fast(group_df, global_num_nodes):
    source_ids = group_df['source_mapped'].values.astype(np.int64)
    destination_ids = group_df['destination_mapped'].values.astype(np.int64)

    edge_index_np = np.stack([source_ids, destination_ids], axis=0)
    edge_index = torch.from_numpy(edge_index_np).long()

    labels_encoded = torch.from_numpy(group_df['Label'].values.astype(np.int64)).long()

    node_features = torch.zeros((global_num_nodes, 1), dtype=torch.float)

    src_ports = torch.from_numpy(group_df['sourcePort'].values.astype(np.float32))
    dst_ports = torch.from_numpy(group_df['destinationPort'].values.astype(np.float32))
    src_idx = torch.from_numpy(source_ids)
    dst_idx = torch.from_numpy(destination_ids)

    node_features.index_copy_(0, src_idx, src_ports.unsqueeze(1))
    node_features.index_copy_(0, dst_idx, dst_ports.unsqueeze(1))

    feature_cols = [
        c
        for c in group_df.columns
        if c
        not in [
            'appName',
            'source',
            'destination',
            'Label',
            'generated',
            'startDateTime',
            'stopDateTime',
            'time_window',
            'source_mapped',
            'destination_mapped',
        ]
    ]
    edge_attr_vals = group_df[feature_cols].values.astype(np.float32)
    edge_attr = torch.from_numpy(edge_attr_vals)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, edge_labels=labels_encoded)

def get_ip_id_hash(ip_str):
    hash_obj = hashlib.md5(str(ip_str).encode())
    return int(hash_obj.hexdigest()[:15], 16)

def create_graph_data_inductive_2012(time_slice):
    time_slice = time_slice.copy()

    time_slice["source"] = time_slice["source"].astype(str).str.strip()
    time_slice["destination"] = time_slice["destination"].astype(str).str.strip()

    src_ips = time_slice["source"].to_numpy()
    dst_ips = time_slice["destination"].to_numpy()

    all_nodes = np.concatenate([src_ips, dst_ips], axis=0)
    unique_nodes, inverse_indices = np.unique(all_nodes, return_inverse=True)

    n_nodes = int(len(unique_nodes))
    src_local = inverse_indices[: len(src_ips)]
    dst_local = inverse_indices[len(src_ips) :]

    edge_index = torch.tensor(np.stack([src_local, dst_local], axis=0), dtype=torch.long)
    n_id = torch.tensor([get_ip_id_hash(ip) for ip in unique_nodes], dtype=torch.long)

    num_edges = int(edge_index.size(1))
    if num_edges <= 0 or n_nodes <= 0:
        return None

    ones = torch.ones(num_edges, dtype=torch.float)
    in_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees.scatter_add_(0, edge_index[0], ones)
    in_degrees.scatter_add_(0, edge_index[1], ones)

    src_port = pd.to_numeric(time_slice.get("sourcePort", 0), errors="coerce").fillna(0).to_numpy()
    is_priv_src = (src_port < 1024).astype(np.float32)
    priv_port_count = torch.zeros(n_nodes, dtype=torch.float)
    priv_port_count.scatter_add_(0, torch.tensor(src_local, dtype=torch.long), torch.tensor(is_priv_src, dtype=torch.float))
    priv_ratio = priv_port_count / (out_degrees + 1e-6)

    pkt_vals = pd.to_numeric(time_slice.get("totalSourcePackets", 0), errors="coerce").fillna(0).to_numpy()
    fwd_pkts = torch.tensor(pkt_vals, dtype=torch.float)
    node_pkt_sum = torch.zeros(n_nodes, dtype=torch.float)
    node_pkt_sum.scatter_add_(0, torch.tensor(src_local, dtype=torch.long), fwd_pkts)

    x = torch.stack(
        [torch.log1p(in_degrees), torch.log1p(out_degrees), priv_ratio, node_pkt_sum],
        dim=1,
    ).float()

    labels = torch.tensor(time_slice["Label"].values.astype(np.int64), dtype=torch.long)

    drop_cols = [
        "appName",
        "source",
        "destination",
        "Label",
        "generated",
        "startDateTime",
        "stopDateTime",
        "time_window",
        "sourcePort",
        "destinationPort",
    ]
    edge_attr_vals = (
        time_slice.drop(columns=drop_cols, errors="ignore")
        .select_dtypes(include=[np.number])
        .values.astype(np.float32)
    )
    edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_labels=labels, n_id=n_id)
    return data

def _compute_ce_class_weights(train_data_seq, num_classes, device):
    all_labels = []
    for graph in train_data_seq:
        if graph is not None:
            all_labels.extend(graph.edge_labels.cpu().numpy().tolist())
            
    if not all_labels: return torch.ones(num_classes).to(device)
    
    counts = np.bincount(all_labels, minlength=num_classes)
    total = len(all_labels)
    weights = np.zeros(num_classes)
    for i in range(num_classes):
        raw_weight = total / (num_classes * counts[i]) if counts[i] > 0 else 1.0
        # === 核心修改：平滑权重 ===
        weights[i] = raw_weight

    print(f"Sample Stats: {counts}")
    # print(f"Smoothed Weights: {weights}")
    return torch.tensor(weights, dtype=torch.float).to(device)

def _save_confusion_matrix_png(cm, class_names, title, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_pct, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.0, vmax=100.0)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           title=title, ylabel='True label', xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    thresh = cm_pct.max() / 2.0 if cm_pct.size > 0 else 0.0
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            ax.text(j, i, f'{cm_pct[i, j]:.2f}%\n({int(cm[i, j])})',
                    ha='center', va='center', color='white' if cm_pct[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)

# ==========================================
# 2. 评估与训练逻辑
# ==========================================

# ==========================================
# 3. 主程序
# ==========================================

def _iscxids2012_sort_key(path):
    base = os.path.basename(path)
    m = re.search(r"Jun(\d{1,2})", base)
    if m:
        return int(m.group(1))
    return base

def _load_one_csv(path):
    df = pd.read_csv(path, low_memory=False)
    return df

def _basic_time_and_label(df):
    df = df.copy()
    df["Label"] = df["Label"].astype(str).str.strip()
    df["Label"] = df["Label"].apply(lambda x: 0 if str(x).lower() == "normal" else 1)

    for col in ["startDateTime", "stopDateTime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=["startDateTime", "stopDateTime"])
    df["Duration of time"] = (df["stopDateTime"] - df["startDateTime"]).dt.total_seconds() / 60.0
    df["time_window"] = df["startDateTime"].dt.floor("min")
    df = df.drop(columns=["stopDateTime", "startDateTime", "generated"], errors="ignore")
    return df

def _encode_and_scale(train_df, val_df, test_df):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    non_null = train_df.notna().mean()
    drop_cols = non_null[non_null <= 0.3].index.tolist()
    drop_cols = [c for c in drop_cols if c not in ["Label", "time_window"]]
    if drop_cols:
        train_df = train_df.drop(columns=drop_cols, errors="ignore")
        val_df = val_df.drop(columns=drop_cols, errors="ignore")
        test_df = test_df.drop(columns=drop_cols, errors="ignore")

    if ("sourceTCPFlagsDescription" in train_df.columns) and ("destinationTCPFlagsDescription" in train_df.columns):
        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

        tcp_cols = ["sourceTCPFlagsDescription", "destinationTCPFlagsDescription"]
        enc.fit(train_df[tcp_cols].fillna(""))

        for df in (train_df, val_df, test_df):
            arr = enc.transform(df[tcp_cols].fillna(""))
            names = enc.get_feature_names_out(tcp_cols)
            df_tcp = pd.DataFrame(arr, columns=names, index=df.index)
            df.drop(columns=tcp_cols, inplace=True, errors="ignore")
            df[names] = df_tcp

    def _fit_unk_mapping(series):
        vals = series.astype(str).fillna("").tolist()
        uniq = sorted(set(vals))
        mapping = {v: i + 1 for i, v in enumerate(uniq)}
        return mapping

    def _apply_unk_mapping(series, mapping):
        s = series.astype(str).fillna("")
        out = s.map(mapping).fillna(0).astype(np.int64)
        return out

    for col in ["protocolName", "sourcePayloadAsUTF", "destinationPayloadAsUTF", "direction"]:
        if col in train_df.columns:
            mapping = _fit_unk_mapping(train_df[col])
            train_df[col] = _apply_unk_mapping(train_df[col], mapping)
            if col in val_df.columns:
                val_df[col] = _apply_unk_mapping(val_df[col], mapping)
            if col in test_df.columns:
                test_df[col] = _apply_unk_mapping(test_df[col], mapping)

    if "sourcePayloadAsBase64" in train_df.columns:
        for df in (train_df, val_df, test_df):
            df["sourcePayloadLength"] = df["sourcePayloadAsBase64"].apply(lambda x: len(str(x)))
            df["destinationPayloadLength"] = df["destinationPayloadAsBase64"].apply(lambda x: len(str(x)))
            df.drop(columns=["sourcePayloadAsBase64", "destinationPayloadAsBase64"], inplace=True, errors="ignore")

    exclude = {"Label", "sourcePort", "destinationPort", "totalSourcePackets", "totalDestinationPackets"}
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    feat_cols = [c for c in numeric_cols if c not in exclude]

    for df in (train_df, val_df, test_df):
        if feat_cols:
            df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan)
            df[feat_cols] = df[feat_cols].fillna(0)

    for col in feat_cols:
        try:
            if float(train_df[col].max()) > 100:
                train_df[col] = np.log1p(train_df[col].abs())
                val_df[col] = np.log1p(val_df[col].abs())
                test_df[col] = np.log1p(test_df[col].abs())
        except Exception:
            continue

    scaler = StandardScaler()
    if feat_cols:
        train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
        val_df[feat_cols] = scaler.transform(val_df[feat_cols])
        test_df[feat_cols] = scaler.transform(test_df[feat_cols])

    return train_df, val_df, test_df

def _build_graph_seq(df):
    seq = []
    grouped = df.groupby("time_window", sort=True)
    for _, group in tqdm(grouped, total=len(grouped), desc="Constructing Graphs", leave=False):
        g = create_graph_data_inductive_2012(group)
        if g is not None:
            seq.append(g)
    return seq

def _compute_sqrt_class_weights_from_graphs(graph_seq, num_classes):
    counts = np.zeros(num_classes, dtype=np.float64)
    for g in graph_seq:
        if g is None:
            continue
        labels = g.edge_labels.detach().cpu().numpy().astype(np.int64)
        counts += np.bincount(labels, minlength=num_classes).astype(np.float64)
    weights = 1.0 / (np.sqrt(counts) + 1.0)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float)

def _edge_label_counts_from_graphs(graph_seq, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for g in graph_seq:
        if g is None:
            continue
        labels = g.edge_labels.detach().cpu().numpy().astype(np.int64)
        if labels.size == 0:
            continue
        counts += np.bincount(labels, minlength=num_classes).astype(np.int64)
    return counts

def run_one_experiment(group, h, train_seq, val_seq, test_seq, edge_dim, device):
    group = (group or "").strip().upper()
    group_tag = group if group else "CUSTOM"

    seq_len = int(h["SEQ_LEN"])
    batch_size = int(h["BATCH_SIZE"])
    num_epochs = int(h["NUM_EPOCHS"])
    lr = float(h["LR"])
    hidden = int(h["HIDDEN"])
    heads = int(h["HEADS"])
    kernels = list(h["KERNELS"])
    patience = int(h["PATIENCE"])
    min_delta = float(h["MIN_DELTA"])
    early_stop_metric = str(h["EARLY_STOP_METRIC"])
    cl_loss_weight = float(h["CL_LOSS_WEIGHT"])
    accum_steps = max(1, int(h.get("ACCUM_STEPS", 1)))
    max_cl_edges = int(h.get("MAX_CL_EDGES", 2048))
    drop_path = float(h.get("DROP_PATH", 0.1))
    dropedge_p = float(h.get("DROPEDGE_P", 0.2))
    cl_view1_dropedge_p = float(h.get("CL_VIEW1_DROPEDGE_P", 0.1))
    cl_view2_dropedge_p = float(h.get("CL_VIEW2_DROPEDGE_P", 0.2))
    warmup_epochs = max(0, int(h.get("WARMUP_EPOCHS", 5)))
    cosine_t0 = max(1, int(h.get("COSINE_T0", 10)))
    cosine_tmult = max(1, int(h.get("COSINE_TMULT", 1)))
    eta_min_ratio = float(h.get("ETA_MIN_RATIO", 0.01))
    target_far = float(h.get("TARGET_FAR", 0.03))

    if len(train_seq) < seq_len or len(val_seq) < 1 or len(test_seq) < 1:
        print(
            f"Skip {group_tag}: SEQ_LEN={seq_len} too long for available sequences "
            f"(train={len(train_seq)}, val={len(val_seq)}, test={len(test_seq)})",
            flush=True,
        )
        return

    class_names = ["Normal", "Attack"]

    val_seqs = _pad_seq_for_last_frame_coverage(val_seq, seq_len)
    test_seqs = _pad_seq_for_last_frame_coverage(test_seq, seq_len)

    train_ds = TemporalGraphDataset(train_seq, seq_len=seq_len)
    val_ds = TemporalGraphDataset(val_seqs, seq_len=seq_len)
    test_ds = TemporalGraphDataset(test_seqs, seq_len=seq_len)

    if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
        print(
            f"Skip {group_tag}: empty dataset windows (train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)})",
            flush=True,
        )
        return

    num_workers = int(os.getenv("NUM_WORKERS", "0"))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(torch.cuda.is_available()),
        "persistent_workers": bool(num_workers > 0),
    }
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=temporal_collate_fn, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate_fn, **loader_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate_fn, **loader_kwargs
    )

    print(
        f"\n=== Experiment {group_tag} ===\n"
        f"SEQ_LEN={seq_len}, BATCH_SIZE={batch_size}, LR={lr}, HIDDEN={hidden}, HEADS={heads}, "
        f"KERNELS={kernels}, NUM_EPOCHS={num_epochs}, PATIENCE={patience}, MIN_DELTA={min_delta}, "
        f"EARLY_STOP_METRIC={early_stop_metric}, CL_LOSS_WEIGHT={cl_loss_weight}, "
        f"ACCUM_STEPS={accum_steps}, DROP_PATH={drop_path}, DROPEDGE_P={dropedge_p}, "
        f"CL_VIEW1_DROPEDGE_P={cl_view1_dropedge_p}, CL_VIEW2_DROPEDGE_P={cl_view2_dropedge_p}, "
        f"WARMUP_EPOCHS={warmup_epochs}, COSINE_T0={cosine_t0}, COSINE_TMULT={cosine_tmult}, ETA_MIN_RATIO={eta_min_ratio}",
        flush=True,
    )

    model = MILAN(
        node_in=4,
        edge_in=edge_dim,
        hidden=hidden,
        num_classes=2,
        seq_len=seq_len,
        heads=heads,
        dropout=0.3,
        max_cl_edges=max_cl_edges,
        kernels=kernels,
        drop_path=drop_path,
        dropedge_p=dropedge_p,
        cl_view1_dropedge_p=cl_view1_dropedge_p,
        cl_view2_dropedge_p=cl_view2_dropedge_p,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    eta_min = max(0.0, lr * float(eta_min_ratio))
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=cosine_t0, T_mult=cosine_tmult, eta_min=eta_min
    )

    weights_cpu = _compute_sqrt_class_weights_from_graphs(train_seq, 2)
    criterion = nn.CrossEntropyLoss(weight=weights_cpu.to(device))

    os.makedirs("models/2012", exist_ok=True)
    os.makedirs("png/2012", exist_ok=True)
    kernel_tag = "-".join(str(k) for k in kernels)
    time_str = datetime.datetime.now().strftime("%m%d_%H%M")
    run_tag = (
        f"{group_tag}_seq{seq_len}_h{hidden}_hd{heads}_k{kernel_tag}"
        f"_lr{lr}_clw{cl_loss_weight}"
        f"_dp{drop_path}_de{dropedge_p}"
        f"_v1{cl_view1_dropedge_p}_v2{cl_view2_dropedge_p}"
        f"_acc{accum_steps}_t0{cosine_t0}_tm{cosine_tmult}_wu{warmup_epochs}"
        f"_{time_str}"
    )
    best_model_path = f"models/2012/best_model_{run_tag}.pth"
    print(f"Best model will be saved to: {best_model_path}", flush=True)

    start_time = time.time()
    best_metric = -float("inf")
    no_improve_epochs = 0

    num_train_steps = max(1, len(train_loader))
    num_opt_steps_per_epoch = max(1, (num_train_steps + accum_steps - 1) // accum_steps)
    warmup_total_steps = int(warmup_epochs) * int(num_opt_steps_per_epoch)

    global_opt_step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_cl_loss = 0.0
        cl_loss_steps = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        optimizer.zero_grad(set_to_none=True)
        for step, batched_seq in enumerate(loop):
            if not batched_seq:
                continue
            batched_seq = [g.to(device) for g in batched_seq]

            out = model(graphs=batched_seq)
            all_preds, cl_loss = out if isinstance(out, tuple) else (out, None)
            last_frame_pred = all_preds[-1]
            last_frame_labels = batched_seq[-1].edge_labels
            last_mask = None
            if hasattr(model, "_last_edge_masks"):
                masks = getattr(model, "_last_edge_masks", None)
                if isinstance(masks, (list, tuple)) and len(masks) > 0:
                    last_mask = masks[-1]
            if last_mask is not None:
                last_frame_labels = last_frame_labels[last_mask]

            main_loss = criterion(last_frame_pred, last_frame_labels)
            if torch.is_tensor(cl_loss):
                full_loss = main_loss + cl_loss_weight * cl_loss
            else:
                full_loss = main_loss
            loss = full_loss / float(accum_steps)

            loss.backward()

            total_loss += float(full_loss.detach().item())
            if torch.is_tensor(cl_loss):
                total_cl_loss += float(cl_loss.detach().item())
                cl_loss_steps += 1

            do_step = ((step + 1) % accum_steps == 0) or ((step + 1) == len(train_loader))
            if do_step:
                if warmup_total_steps > 0 and global_opt_step < warmup_total_steps:
                    warm_lr = lr * float(global_opt_step + 1) / float(warmup_total_steps)
                    for pg in optimizer.param_groups:
                        pg["lr"] = float(warm_lr)
                else:
                    progress = float(epoch) + float(step + 1) / float(num_train_steps)
                    scheduler.step(progress - float(warmup_epochs))

                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_opt_step += 1

            if torch.is_tensor(cl_loss):
                loop.set_postfix(loss=float(full_loss.detach().item()), cl_loss=float(cl_loss.detach().item()))
            else:
                loop.set_postfix(loss=float(full_loss.detach().item()))

        avg_loss = total_loss / max(1, len(train_loader))
        avg_cl_loss = total_cl_loss / max(1, cl_loss_steps)

        (
            avg_val_loss,
            _val_acc,
            _val_prec,
            _val_rec,
            val_f1_macro,
            _val_f1_weighted,
            _val_auc,
            val_asa,
            _val_far,
            y_true_val,
            y_pred_val,
            _y_probs_val,
        ) = evaluate_comprehensive_v2(
            model,
            val_loader,
            device,
            class_names,
            average="macro",
            return_details=True,
            criterion=criterion,
            return_loss=True,
        )
        try:
            val_f1_attack = float(f1_score(y_true_val, y_pred_val, average="binary", pos_label=1, zero_division=0))
        except Exception:
            val_f1_attack = 0.0
        try:
            val_f1_weighted = float(f1_score(y_true_val, y_pred_val, average="weighted", zero_division=0))
        except Exception:
            val_f1_weighted = 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{group_tag}] Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val F1(macro): {val_f1_macro:.4f} | Val F1(weighted): {val_f1_weighted:.4f} | Val F1(attack): {val_f1_attack:.4f} | "
            f"ASA: {val_asa:.4f} | FPR: {_val_far:.4f} | CL Loss: {avg_cl_loss:.4f} | "
            f"LR: {current_lr:.6f}",
            flush=True,
        )

        metric_value = val_f1_macro if early_stop_metric == "val_f1" else val_asa
        metric_display = "Val F1(macro)" if early_stop_metric == "val_f1" else "Val ASA"


        if metric_value > best_metric + min_delta:
            best_metric = metric_value
            no_improve_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seq_len": seq_len,
                    "num_classes": 2,
                    "edge_dim": edge_dim,
                    "class_names": class_names,
                },
                best_model_path,
            )
            print(f"[{group_tag}] New Best Model Saved! ({metric_display}: {best_metric:.4f})", flush=True)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"[{group_tag}] Early Stopping at Epoch {epoch+1}", flush=True)
                break

    print(f"\n[{group_tag}] Loading Best Model for Final Testing...", flush=True)
    if os.path.exists(best_model_path):
        try:
            ckpt = torch.load(best_model_path, map_location=device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                model.load_state_dict(ckpt["state_dict"])
            else:
                model.load_state_dict(ckpt)
        except RuntimeError as e:
            print(f"[{group_tag}] Failed to load checkpoint. Using current weights. Error: {e}", flush=True)

    print(f"\n[{group_tag}] === Post-Training Threshold Optimization ===", flush=True)
    y_true_val, y_score_val = _collect_attack_scores(model, val_loader, device, class_names)
    best_thresh, val_f1, val_far, val_asa = _attack_best_threshold(y_true_val, y_score_val, max_far=target_far)
    print(
        f"[{group_tag}] Best Threshold found on VAL -> th={best_thresh:.4f}, Val F1={val_f1:.4f}, Val FAR={val_far:.4f}, Val ASA={val_asa:.4f}",
        flush=True,
    )

    print(f"\n[{group_tag}] Best Strategy: Threshold = {best_thresh}", flush=True)
    (
        acc,
        prec,
        rec,
        f1_macro,
        f1_weighted,
        auc,
        asa,
        far,
        y_true,
        y_pred,
    ) = evaluate_comprehensive_with_threshold_v2(model, test_loader, device, class_names, threshold=best_thresh, average="macro")
    f1 = f1_macro
    print(
        f"[{group_tag}] Final Test -> ACC: {acc:.4f}, PREC: {prec:.4f}, Rec: {rec:.4f}, "
        f"F1(macro): {f1:.4f}, F1(weighted): {f1_weighted:.4f}, "
        f"AUC: {auc:.4f}, ASA: {asa:.4f}, FAR: {far:.4f}",
        flush=True,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    _save_confusion_matrix_png(
        cm,
        class_names,
        f"{group_tag} Confusion Matrix (Threshold={best_thresh})",
        f"png/2012/FINAL_CM_{run_tag}.png",
    )
    print(f"[{group_tag}] Confusion Matrix Saved.", flush=True)

    # 1. 补算测试集的 F1_Weighted
    try:
        final_f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    except Exception:
        final_f1_weighted = 0.0

    # 2. 写入包含 8 大核心指标的终极 CSV
    log_file = "experiment_results.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("Dataset,Group,SeqLen,Threshold,ACC,PRE,REC,F1_Macro,F1_Weighted,AUC,ASA,FAR\n")
        f.write(
            f"ISCX2012,{group_tag},{seq_len},{best_thresh:.6f},"
            f"{acc:.4f},{prec:.4f},{rec:.4f},"
            f"{f1:.4f},{final_f1_weighted:.4f},{auc:.4f},"
            f"{asa:.4f},{far:.4f}\n"
        )
    print(f"[{group_tag}] Total Time: {time.time() - start_time:.2f}s", flush=True)

def main():
    set_seed(int(os.getenv("SEED", "42")))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}", flush=True)

    base_path = os.getenv("DATA_DIR", "data/ISCXIDS2012")
    csv_paths = [os.path.join(base_path, p) for p in os.listdir(base_path) if p.lower().endswith(".csv")]
    csv_paths = sorted(csv_paths, key=_iscxids2012_sort_key)
    if len(csv_paths) < 6:
        raise FileNotFoundError(f"Expected at least 6 CSVs under {base_path}, got {len(csv_paths)}")

    train_paths = csv_paths[:4]
    val_path = csv_paths[4]
    test_path = csv_paths[5]

    print("Dataset Split:", flush=True)
    print("  Train:", [os.path.basename(p) for p in train_paths], flush=True)
    print("  Val:  ", os.path.basename(val_path), flush=True)
    print("  Test: ", os.path.basename(test_path), flush=True)

    print("Loading CSVs...", flush=True)
    train_list = []
    for p in train_paths:
        df = _load_one_csv(p)
        train_list.append(_basic_time_and_label(df))
        print(f"  -> Loaded {os.path.basename(p)}: {len(df)} rows", flush=True)

    val_raw = _basic_time_and_label(_load_one_csv(val_path))
    test_raw = _basic_time_and_label(_load_one_csv(test_path))
    train_raw = pd.concat(train_list, ignore_index=True) if train_list else val_raw.iloc[0:0].copy()

    print(f"Raw Rows -> Train: {len(train_raw)}, Val: {len(val_raw)}, Test: {len(test_raw)}", flush=True)

    train_df, val_df, test_df = _encode_and_scale(train_raw, val_raw, test_raw)

    print("Building Graph Sequences...", flush=True)
    train_seq = _build_graph_seq(train_df)
    val_seq = _build_graph_seq(val_df)
    test_seq = _build_graph_seq(test_df)

    print(f"Graph Count -> Train={len(train_seq)}, Val={len(val_seq)}, Test={len(test_seq)}", flush=True)
    train_counts = _edge_label_counts_from_graphs(train_seq, num_classes=2)
    val_counts = _edge_label_counts_from_graphs(val_seq, num_classes=2)
    test_counts = _edge_label_counts_from_graphs(test_seq, num_classes=2)
    print(f"Edge Label Count -> Train Normal={int(train_counts[0])} Attack={int(train_counts[1])}", flush=True)
    print(f"Edge Label Count -> Val   Normal={int(val_counts[0])} Attack={int(val_counts[1])}", flush=True)
    print(f"Edge Label Count -> Test  Normal={int(test_counts[0])} Attack={int(test_counts[1])}", flush=True)

    min_val_attack_edges = int(os.getenv("MIN_VAL_ATTACK_EDGES", "50"))
    if int(val_counts[1]) < int(min_val_attack_edges):
        combined = list(train_seq) + list(val_seq)
        n = len(combined)
        if n >= 2:
            start = max(1, int(n * 0.9))
            chosen_start = start
            best_candidate = None
            best_candidate_attack = -1
            while chosen_start > 1:
                candidate = combined[chosen_start:]
                c = _edge_label_counts_from_graphs(candidate, num_classes=2)
                cand_attack = int(c[1])
                if cand_attack > best_candidate_attack:
                    best_candidate_attack = cand_attack
                    best_candidate = chosen_start
                if cand_attack >= int(min_val_attack_edges):
                    break
                step = max(1, int(n * 0.05))
                chosen_start = max(1, chosen_start - step)
            if best_candidate is not None:
                chosen_start = best_candidate
            train_seq = combined[:chosen_start]
            val_seq = combined[chosen_start:]
            train_counts = _edge_label_counts_from_graphs(train_seq, num_classes=2)
            val_counts = _edge_label_counts_from_graphs(val_seq, num_classes=2)
            print(
                f"⚠️ Val split has too few Attack edges (<{int(min_val_attack_edges)}); fallback to split from Train+Val sequence.",
                flush=True,
            )
            print(f"Graph Count (fallback) -> Train={len(train_seq)}, Val={len(val_seq)}", flush=True)
            print(f"Edge Label Count (fallback) -> Train Normal={int(train_counts[0])} Attack={int(train_counts[1])}", flush=True)
            print(f"Edge Label Count (fallback) -> Val   Normal={int(val_counts[0])} Attack={int(val_counts[1])}", flush=True)

    if len(train_seq) > 0:
        edge_dim = int(train_seq[0].edge_attr.shape[1])
    elif len(val_seq) > 0:
        edge_dim = int(val_seq[0].edge_attr.shape[1])
    elif len(test_seq) > 0:
        edge_dim = int(test_seq[0].edge_attr.shape[1])
    else:
        edge_dim = 1
    print(f"Edge Dim: {edge_dim}", flush=True)

    raw_groups = os.getenv("HP_GROUPS", "").strip()
    if raw_groups:
        raw = raw_groups.replace(";", ",").replace("\n", ",").replace(" ", ",")
        groups = [g.strip() for g in raw.split(",") if g.strip()]
    else:
        groups = [os.getenv("HP_GROUP", "").strip()]
    groups = [g for g in groups if str(g).strip() != ""]
    if len(groups) == 0:
        groups = ["BEST"]

    overall_start = time.time()
    for g in groups:
        try:
            h = resolve_hparams(g, env=os.environ, dataset="iscx2012")
            run_one_experiment(g, h, train_seq, val_seq, test_seq, edge_dim, device)
        except torch.OutOfMemoryError as e:
            print(f"[{str(g).strip().upper() or 'CUSTOM'}] CUDA OOM, skip this group. Error: {e}", flush=True)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"[{str(g).strip().upper() or 'CUSTOM'}] CUDA OOM, skip this group. Error: {e}", flush=True)
            else:
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    print(f"\nAll Experiments Done. Total Time: {time.time() - overall_start:.2f}s", flush=True)

if __name__ == "__main__":
    main()

# HP_GROUPS=D6 python train_2012.py
# 运行指令
# cd /root/project/reon
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# MIN_VAL_ATTACK_EDGES=50 \
# HP_GROUPS=IS_EXP1_BASE,IS_EXP2_TINY,IS_EXP3_REG,IS_EXP4_NOCL \
# python MILAN/train_2012.py
# cd /root/project/reon
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# HP_GROUPS=NB_EXP5_LONGSEQ \
# python MILAN/a1_2.py
# 看进度
# tail -f logs_all_abl.txt
