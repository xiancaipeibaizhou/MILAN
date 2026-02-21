import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import hashlib
import os
import time
import glob
import random
import datetime
import gc
import warnings
from tqdm import tqdm
from hparams_a3 import resolve_hparams
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import MILAN

# === 导入核心模型与评估函数 ===
# 确保 analys.py 和 model_Final.py 在同一目录下
try:
    from analys import (
        _attack_best_threshold,
        _collect_attack_scores,
        FocalLoss,
        evaluate_comprehensive_v2,
        evaluate_comprehensive_with_threshold_v2,
        evaluate_with_threshold,
    )
    from model import MILAN
except ImportError:
    print("❌ Error: analys.py or model_Final.py not found. Please check your directory.")
    exit()

# ==========================================
# 辅助函数
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _subnet_key(ip):
    try:
        parts = str(ip).split(".")
        if len(parts) < 3:
            return (0, 0, 0)
        a = int(parts[0])
        b = int(parts[1])
        c = int(parts[2])
        return (a, b, c)
    except Exception:
        return (0, 0, 0)

def get_ip_id_hash(ip_str):
    hash_obj = hashlib.md5(str(ip_str).encode())
    return int(hash_obj.hexdigest()[:15], 16)

UNK_SUBNET_ID = 0

def get_subnet_id_safe(ip_str, subnet_map):
    key = _subnet_key(ip_str)
    return subnet_map.get(key, UNK_SUBNET_ID)

def _parse_timestamp_series(ts):
    ts = ts.astype(str)
    sample = ts.replace({"nan": np.nan, "None": np.nan}).dropna().head(2000)
    if len(sample) == 0:
        return pd.to_datetime(ts, errors="coerce", dayfirst=True)

    candidates = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%d/%m/%Y %I:%M:%S %p",
        "%m/%d/%Y %I:%M:%S %p",
    ]

    best_fmt = None
    best_ok = -1
    for fmt in candidates:
        dt = pd.to_datetime(sample, format=fmt, errors="coerce")
        ok = int(dt.notna().sum())
        if ok > best_ok:
            best_ok = ok
            best_fmt = fmt

    if best_fmt is not None and best_ok >= int(len(sample) * 0.9):
        return pd.to_datetime(ts, format=best_fmt, errors="coerce")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return pd.to_datetime(ts, errors="coerce", dayfirst=True)

# ==========================================
# 1. 稀疏图构建函数 (修复版)
# ==========================================
def create_graph_data_inductive(time_slice, subnet_map, label_encoder, time_window):
    # 避免 SettingWithCopyWarning
    time_slice = time_slice.copy()
    
    # 确保 IP 为字符串
    time_slice['Src IP'] = time_slice['Src IP'].astype(str).str.strip()
    time_slice['Dst IP'] = time_slice['Dst IP'].astype(str).str.strip()

    # 使用 Hash ID
    src_ids = time_slice['Src IP'].apply(get_ip_id_hash).values.astype(np.int64)
    dst_ids = time_slice['Dst IP'].apply(get_ip_id_hash).values.astype(np.int64)

    # 标签处理
    if 'Label' in time_slice.columns:
        if pd.api.types.is_numeric_dtype(time_slice['Label']):
            labels = time_slice['Label'].values.astype(int)
        else:
            labels = np.zeros(len(time_slice), dtype=int)
    else:
        labels = np.zeros(len(time_slice), dtype=int)

    # 构建局部图索引
    all_nodes_in_slice = np.concatenate([src_ids, dst_ids])
    unique_nodes, inverse_indices = np.unique(all_nodes_in_slice, return_inverse=True)
    
    n_nodes = len(unique_nodes)
    src_local = inverse_indices[:len(src_ids)]
    dst_local = inverse_indices[len(src_ids):]
    
    edge_index = torch.tensor(np.array([src_local, dst_local]), dtype=torch.long)
    n_id = torch.tensor(unique_nodes, dtype=torch.long)
    
    # --- 节点特征 ---
    ones = torch.ones(edge_index.size(1), dtype=torch.float)
    in_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees = torch.zeros(n_nodes, dtype=torch.float)
    
    if edge_index.size(1) > 0:
        out_degrees.scatter_add_(0, edge_index[0], ones)
        in_degrees.scatter_add_(0, edge_index[1], ones)

    # 特征: 特权端口
    src_port_col = 'Src Port' if 'Src Port' in time_slice.columns else 'Source Port'
    src_port = pd.to_numeric(time_slice.get(src_port_col, 0), errors='coerce').fillna(0).values
    is_priv_src = (src_port < 1024).astype(np.float32)
    
    priv_port_count = torch.zeros(n_nodes, dtype=torch.float)
    if edge_index.size(1) > 0:
        priv_port_count.scatter_add_(0, edge_index[0], torch.tensor(is_priv_src, dtype=torch.float))
    priv_ratio = priv_port_count / (out_degrees + 1e-6)

    # 特征: 流量聚合
    pkt_col = None
    for cand in ['Total Fwd Packets', 'Total Fwd Packet', 'Tot Fwd Pkts', 'Fwd Packets', 'Total Fwd Pkts']:
        if cand in time_slice.columns:
            pkt_col = cand
            break
    
    if pkt_col is None:
        fwd_pkts = torch.zeros(edge_index.size(1), dtype=torch.float)
    else:
        fwd_pkts = torch.tensor(
            pd.to_numeric(time_slice[pkt_col], errors='coerce').fillna(0).values,
            dtype=torch.float,
        )
    
    node_pkt_sum = torch.zeros(n_nodes, dtype=torch.float)
    if edge_index.size(1) > 0:
        node_pkt_sum.scatter_add_(0, edge_index[0], fwd_pkts)
    
    x = torch.stack(
        [torch.log1p(in_degrees), torch.log1p(out_degrees), priv_ratio, node_pkt_sum],
        dim=-1,
    ).float()

    subnet_id = None
    if subnet_map is not None:
        subnet_ids_for_node = {}
        unique_ips = pd.concat([time_slice['Src IP'], time_slice['Dst IP']]).unique()
        for ip_str in unique_ips:
            hid = get_ip_id_hash(ip_str)
            subnet_ids_for_node[hid] = get_subnet_id_safe(ip_str, subnet_map)

        subnet_id = torch.tensor(
            [subnet_ids_for_node.get(int(h), UNK_SUBNET_ID) for h in unique_nodes],
            dtype=torch.long,
        )

    # --- 边特征 ---
    drop_cols = ['Src IP', 'Dst IP', 'Flow ID', 'Label', 'Timestamp', 'Src Port', 'Dst Port', 
                 'Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'time_idx']
    edge_attr_vals = time_slice.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).values
    edge_attr = torch.tensor(edge_attr_vals, dtype=torch.float)

    if edge_index.size(1) > 0:
        edge_labels = torch.tensor(labels, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, edge_labels=edge_labels, n_id=n_id)
        if subnet_id is not None:
            data.subnet_id = subnet_id
        return data
    else:
        return None

# ==========================================
# 2. Dataset & Collate
# ==========================================
class TemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data_seq, seq_len=8):
        self.graph_data_seq = [g for g in graph_data_seq if g is not None]
        self.seq_len = seq_len
    
    def __len__(self):
        return max(0, len(self.graph_data_seq) - self.seq_len + 1)

    def __getitem__(self, idx):
        return self.graph_data_seq[idx : idx + self.seq_len]

def temporal_collate_fn(batch):
    if len(batch) == 0: return []
    seq_len = len(batch[0])
    batched_seq = []
    for t in range(seq_len):
        graphs_at_t = [sample[t] for sample in batch]
        batched_seq.append(Batch.from_data_list(graphs_at_t))
    return batched_seq

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


def run_one_experiment(
    group,
    h,
    base_train_seqs,
    base_val_seqs,
    base_test_seqs,
    edge_dim,
    class_names,
    weights_cpu,
    device,
):
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
    target_far = float(h.get("TARGET_FAR", 0.005))

    if len(base_train_seqs) < seq_len or len(base_val_seqs) < 1 or len(base_test_seqs) < 1:
        print(
            f"Skip {group_tag}: SEQ_LEN={seq_len} too long for available sequences "
            f"(train={len(base_train_seqs)}, val={len(base_val_seqs)}, test={len(base_test_seqs)})",
            flush=True,
        )
        return

    val_seqs = _pad_seq_for_last_frame_coverage(base_val_seqs, seq_len)
    test_seqs = _pad_seq_for_last_frame_coverage(base_test_seqs, seq_len)

    train_dataset = TemporalGraphDataset(base_train_seqs, seq_len=seq_len)
    val_dataset = TemporalGraphDataset(val_seqs, seq_len=seq_len)
    test_dataset = TemporalGraphDataset(test_seqs, seq_len=seq_len)

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print(
            f"Skip {group_tag}: empty dataset windows (train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)})",
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
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=temporal_collate_fn, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate_fn, **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate_fn, **loader_kwargs
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
        num_classes=len(class_names),
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

    criterion = nn.CrossEntropyLoss(weight=weights_cpu.to(device))

    os.makedirs("models/ids2017_full", exist_ok=True)
    os.makedirs("png/ids2017_full", exist_ok=True)
    kernel_tag = "-".join(str(k) for k in kernels)
    time_str = datetime.datetime.now().strftime("%m%d_%H%M")
    run_tag = f"{group_tag}_seq{seq_len}_h{hidden}_k{kernel_tag}_cls{len(class_names)}_{time_str}"
    best_model_path = f"models/ids2017_full/best_model_{run_tag}.pth"

    print(f"Best model will be saved to: {best_model_path}", flush=True)

    print("🔥 Start Training...", flush=True)
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

            edge_masks = getattr(model, "_last_edge_masks", None)
            if edge_masks is not None and len(edge_masks) > 0 and edge_masks[-1] is not None:
                last_frame_labels = batched_seq[-1].edge_labels[edge_masks[-1]]
            else:
                last_frame_labels = batched_seq[-1].edge_labels

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
            val_acc,
            val_prec,
            val_rec,
            val_f1_macro,
            val_f1_weighted,
            val_auc,
            val_asa,
            val_far,
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
        val_f1 = val_f1_macro

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{group_tag}] Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val F1(macro): {val_f1:.4f} | Val F1(weighted): {val_f1_weighted:.4f} | "
            f"ASA: {val_asa:.4f} | FPR: {val_far:.4f} | CL Loss: {avg_cl_loss:.4f} | LR: {current_lr:.6f}",
            flush=True,
        )

        metric_value = val_f1 if early_stop_metric == "val_f1" else val_asa
        metric_display = "Val F1(macro)" if early_stop_metric == "val_f1" else "Val ASA"

        if metric_value > best_metric + min_delta:
            best_metric = metric_value
            no_improve_epochs = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "seq_len": seq_len,
                    "num_classes": len(class_names),
                    "class_names": class_names,
                    "edge_dim": edge_dim,
                },
                best_model_path,
            )
            print(f"[{group_tag}] New Best Model Saved! ({metric_display}: {best_metric:.4f})", flush=True)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"[{group_tag}] ⏹️ Early Stopping at Epoch {epoch+1}", flush=True)
                break

    print(f"\n[{group_tag}] Loading Best Model for Final Testing...", flush=True)
    if os.path.exists(best_model_path):
        try:
            ckpt = torch.load(best_model_path, map_location=device)
            if isinstance(ckpt, dict) and "state_dict" in ckpt:
                ckpt_seq_len = ckpt.get("seq_len", None)
                ckpt_num_classes = ckpt.get("num_classes", None)
                ckpt_edge_dim = ckpt.get("edge_dim", None)

                if ckpt_seq_len != seq_len or ckpt_num_classes != len(class_names) or ckpt_edge_dim != edge_dim:
                    print(
                        f"[{group_tag}] ⚠️ Checkpoint config mismatch, skip loading. "
                        f"(ckpt seq_len={ckpt_seq_len}, num_classes={ckpt_num_classes}, edge_dim={ckpt_edge_dim}) "
                        f"vs (current seq_len={seq_len}, num_classes={len(class_names)}, edge_dim={edge_dim})",
                        flush=True,
                    )
                else:
                    model.load_state_dict(ckpt["state_dict"], strict=True)
            else:
                model.load_state_dict(ckpt, strict=True)
        except Exception as e:
            print(f"[{group_tag}] ⚠️ Failed to load best model: {e}", flush=True)

    print(f"\n[{group_tag}] === Post-Training Threshold Optimization ===", flush=True)
    y_true_val, y_score_val = _collect_attack_scores(model, val_loader, device, class_names)
    best_thresh, val_f2, val_far, val_asa = _attack_best_threshold(y_true_val, y_score_val, max_far=target_far)
    print(
        f"[{group_tag}] Best Threshold found on VAL -> th={best_thresh:.4f}, Val F2={val_f2:.4f}, Val FAR={val_far:.4f}, Val ASA={val_asa:.4f}",
        flush=True,
    )

    print(f"\n[{group_tag}] === Final Evaluation on Test Set ===", flush=True)
    (
        final_acc,
        final_prec,
        final_rec,
        final_f1_macro,
        final_f1_weighted,
        final_auc,
        final_asa,
        final_far,
        final_labels,
        final_preds,
    ) = evaluate_comprehensive_with_threshold_v2(model, test_loader, device, class_names, threshold=best_thresh, average="weighted")
    final_f1 = final_f1_macro

    present = np.unique(np.asarray(final_labels, dtype=np.int64))
    missing = sorted(list(set(range(len(class_names))) - set(present.tolist())))
    counts = np.bincount(np.asarray(final_labels, dtype=np.int64), minlength=len(class_names))
    print(f"[{group_tag}] Final Labels Present IDs: {present.tolist()}", flush=True)
    if len(missing) > 0:
        missing_names = [class_names[i] for i in missing if i < len(class_names)]
        print(f"[{group_tag}] ⚠️ Final Labels Missing IDs: {missing} ({missing_names})", flush=True)
    nonzero_pairs = []
    for i, c in enumerate(counts.tolist()):
        if c > 0:
            nonzero_pairs.append(f"{class_names[i]}({i}):{c}")
    print(f"[{group_tag}] Final Labels Counts -> " + ", ".join(nonzero_pairs), flush=True)
    print(
        f"[{group_tag}] Final Test -> ACC: {final_acc:.4f}, PREC: {final_prec:.4f}, Rec: {final_rec:.4f}, "
        f"F1(macro): {final_f1:.4f}, F1(weighted): {final_f1_weighted:.4f}, "
        f"AUC: {final_auc:.4f}, ASA: {final_asa:.4f}, FAR: {final_far:.4f}",
        flush=True,
    )

    # 1. 补算测试集的 F1_Weighted
    try:
        final_f1_weighted = float(f1_score(final_labels, final_preds, average="weighted", zero_division=0))
    except Exception:
        final_f1_weighted = 0.0

    # 2. 写入包含 8 大核心指标的终极 CSV
    log_file = "experiment_results.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", encoding="utf-8") as f:
        if not file_exists:
            # 统一为论文要求的指标顺序
            f.write("Dataset,Group,SeqLen,Threshold,ACC,PRE,REC,F1_Macro,F1_Weighted,AUC,ASA,FAR\n")

        # 注意：下面的 DATASET_NAME 请根据脚本自行改为 NB15, Darknet2020 或 IDS2017
        DATASET_NAME = "IDS2017"
        f.write(
            f"{DATASET_NAME},{group_tag},{seq_len},{best_thresh:.6f},"
            f"{final_acc:.4f},{final_prec:.4f},{final_rec:.4f},"
            f"{final_f1:.4f},{final_f1_weighted:.4f},{final_auc:.4f},"
            f"{final_asa:.4f},{final_far:.4f}\n"
        )

    try:
        labels_idx = list(range(len(class_names)))
        cm = confusion_matrix(final_labels, final_preds, labels=labels_idx)
        cm = np.asarray(cm, dtype=np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0

        import seaborn as sns
        
        plt.figure(figsize=(12, 10)) # 保持稍大的画布
        
        # === 动态文本矩阵 ===
        annot = np.empty_like(cm_pct, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if cm[i, j] == 0:
                    annot[i, j] = "0"  # 核心改动：0值只保留一个0
                else:
                    annot[i, j] = f"{int(cm[i, j])}\n({cm_pct[i, j]:.1f}%)"

        ax = sns.heatmap(
            cm_pct,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            vmin=0.0,
            vmax=100.0,
            linewidths=0.5,           # 保留细微的白色网格线
            annot_kws={"size": 9}     # 字体大小适中
        )
        
        plt.xticks(rotation=45, ha='right', fontsize=10) 
        plt.yticks(rotation=0, fontsize=10)
        
        plt.title(f"{group_tag} Confusion Matrix (Threshold={best_thresh:.4f})", fontsize=14, pad=15)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"png/ids2017_full/FINAL_CM_{run_tag}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[{group_tag}] Confusion Matrix Saved (Optimized UI).", flush=True)
    except Exception as e:
        print(f"[{group_tag}] Plotting failed: {e}", flush=True)

    print(f"[{group_tag}] Total Time: {time.time() - start_time:.2f}s", flush=True)

# ==========================================
# 3. 主训练流程
# ==========================================
def main():
    set_seed(int(os.getenv("SEED", "42")))
    # --- 配置 ---
    # 指向存放 IDS2017 所有 CSV 的文件夹
    DATA_DIR = os.getenv("DATA_DIR", "data/2017/TrafficLabelling_")
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {DEVICE}")

    # --- 1. 批量加载与拼接 CSV ---
    print(f"Scanning CSV files in {DATA_DIR}...")
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}. Please check the path.")
        return
    
    data_frames = []
    for file_path in tqdm(csv_files, desc="Loading CSVs"):
        try:
            df = pd.read_csv(file_path, encoding="latin1", low_memory=False)
            df.columns = df.columns.str.strip()
            
            rename_map = {
                "Source IP": "Src IP", "Destination IP": "Dst IP",
                "Source Port": "Src Port", "Destination Port": "Dst Port",
                " Timestamp": "Timestamp"
            }
            df = df.rename(columns=rename_map)
            
            if "Timestamp" not in df.columns: continue
            data_frames.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")

    if not data_frames: return

    print("Concatenating DataFrames...")
    data = pd.concat(data_frames, ignore_index=True)
    del data_frames

    # --- 2. 数据清洗 ---
    print("Cleaning Data...")
    data["Label"] = data["Label"].astype(str).str.strip()
    data = data[data["Label"].notna()]
    data = data[data["Label"] != ""]
    data = data[~data["Label"].str.lower().isin(["nan", "none"])]

    label_merge = str(os.getenv("LABEL_MERGE", "none")).strip().lower()
    if label_merge in {"b", "web_attack", "webattack"}:
        data["Label"] = data["Label"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        web_mask = data["Label"].str.lower().str.startswith("web attack")
        data.loc[web_mask, "Label"] = "Web Attack"
    
    # 编码 Label
    label_encoder = LabelEncoder()
    data["Label"] = label_encoder.fit_transform(data["Label"])
    class_names = list(label_encoder.classes_)
    print(f"🏷️ Classes: {class_names}")
    vc_all = data["Label"].value_counts().sort_index()
    pairs_all = []
    for label_id, cnt in vc_all.items():
        label_name = class_names[int(label_id)] if int(label_id) < len(class_names) else str(label_id)
        pairs_all.append(f"{label_name}({int(label_id)}):{int(cnt)}")
    print("Overall Label Counts -> " + ", ".join(pairs_all), flush=True)

    # 时间处理
    print("Parsing Timestamps & Sorting...")
    # IDS2017 常见格式: "dd/MM/yyyy h:mm" 或 "dd/MM/yyyy hh:mm:ss a"
    data["Timestamp"] = _parse_timestamp_series(data["Timestamp"])
    data.dropna(subset=["Timestamp", "Src IP", "Dst IP"], inplace=True)
    
    # 全局按时间排序 (这是时序学习的关键)
    data = data.sort_values("Timestamp").reset_index(drop=True)
    data["time_idx"] = data["Timestamp"].dt.floor("min")

    split_mode = str(os.getenv("SPLIT_MODE", "stratified_time")).strip().lower()
    train_ratio = float(os.getenv("TRAIN_RATIO", "0.8"))
    val_ratio = float(os.getenv("VAL_RATIO", "0.1"))
    train_ratio = float(max(0.05, min(0.95, train_ratio)))
    val_ratio = float(max(0.01, min(0.5, val_ratio)))
    if train_ratio + val_ratio >= 0.99:
        val_ratio = max(0.01, 0.99 - train_ratio)

    unique_times = data["time_idx"].drop_duplicates().values
    total_len = len(unique_times)
    if total_len < 3:
        raise ValueError(f"Not enough unique time points for splitting: total_len={total_len}")

    if split_mode in {"stratified_time", "stratified", "balanced"}:
        print("Performing Snapshot-Level Stratified Split by time_idx...", flush=True)
        from sklearn.model_selection import train_test_split

        counts_by_time = (
            data.groupby(["time_idx", "Label"], sort=False)
            .size()
            .unstack(fill_value=0)
            .reindex(unique_times, fill_value=0)
        )
        unique_times = counts_by_time.index.values
        test_ratio = 1.0 - train_ratio - val_ratio
        split_seed = int(os.getenv("SPLIT_SEED", os.getenv("SEED", "42")))
        test_ratio = float(max(0.0, test_ratio))

        normal_col = int(counts_by_time.sum(axis=0).idxmax())
        cols = [int(c) for c in counts_by_time.columns.tolist()]
        attack_cols = [c for c in cols if int(c) != int(normal_col)]

        dominant_labels = []
        for t in unique_times:
            row = counts_by_time.loc[t]
            if len(attack_cols) > 0 and float(row[attack_cols].sum()) > 0.0:
                dominant_labels.append(int(row[attack_cols].idxmax()))
            else:
                dominant_labels.append(int(normal_col))
        dominant_labels = np.asarray(dominant_labels, dtype=np.int64)

        try:
            train_times, temp_times, _, temp_labels = train_test_split(
                unique_times,
                dominant_labels,
                test_size=(val_ratio + test_ratio),
                stratify=dominant_labels,
                random_state=split_seed,
            )
            val_times, test_times = train_test_split(
                temp_times,
                test_size=(test_ratio / (val_ratio + test_ratio)),
                stratify=temp_labels,
                random_state=split_seed,
            )
        except ValueError:
            print("⚠️ Warning: falling back to random snapshot split...", flush=True)
            train_times, temp_times = train_test_split(
                unique_times,
                test_size=(val_ratio + test_ratio),
                random_state=split_seed,
            )
            val_times, test_times = train_test_split(
                temp_times,
                test_size=(test_ratio / (val_ratio + test_ratio)),
                random_state=split_seed,
            )

        train_df = data[data["time_idx"].isin(train_times)].sort_values("Timestamp").copy()
        val_df = data[data["time_idx"].isin(val_times)].sort_values("Timestamp").copy()
        test_df = data[data["time_idx"].isin(test_times)].sort_values("Timestamp").copy()
        del data
    else:
        print("Performing Temporal Split (8:1:1)...")
        train_idx = int(total_len * train_ratio)
        val_idx = int(total_len * (train_ratio + val_ratio))
        train_idx = max(1, min(total_len - 2, train_idx))
        val_idx = max(train_idx + 1, min(total_len - 1, val_idx))

        split_time_train = unique_times[train_idx]
        split_time_val = unique_times[val_idx]
        print(f"Splitting: Train < {split_time_train} <= Val < {split_time_val} <= Test", flush=True)

        train_df = data[data["time_idx"] < split_time_train].copy()
        val_df = data[(data["time_idx"] >= split_time_train) & (data["time_idx"] < split_time_val)].copy()
        test_df = data[data["time_idx"] >= split_time_val].copy()
        del data

    print(f"Final Split -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}", flush=True)
    print(f"Train Set Classes: {np.unique(train_df['Label'].values)}", flush=True)
    print(f"Val Set Classes: {np.unique(val_df['Label'].values)}", flush=True)
    print(f"Test Set Classes: {np.unique(test_df['Label'].values)}", flush=True)

    def _print_label_counts(df, split_name):
        vc = df["Label"].value_counts().sort_index()
        pairs = []
        for label_id, cnt in vc.items():
            label_name = class_names[int(label_id)] if int(label_id) < len(class_names) else str(label_id)
            pairs.append(f"{label_name}({int(label_id)}):{int(cnt)}")
        print(f"{split_name} Label Counts -> " + ", ".join(pairs), flush=True)

    _print_label_counts(train_df, "Train")
    _print_label_counts(val_df, "Val")
    _print_label_counts(test_df, "Test")

    # --- 4. 归一化 ---
    print("Pre-processing & Normalization...")
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {"Label", "Timestamp", "Src IP", "Dst IP", "Flow ID", "Src Port", "Dst Port", "time_idx"}
    feat_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    # 清洗 Inf/NaN 并填充 0
    for df in [train_df, val_df, test_df]:
        df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Log1p
    for col in feat_cols:
        if train_df[col].max() > 100:
            train_df[col] = np.log1p(train_df[col].abs())
            val_df[col] = np.log1p(val_df[col].abs())
            test_df[col] = np.log1p(test_df[col].abs())
            
    scaler = StandardScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    val_df[feat_cols] = scaler.transform(val_df[feat_cols])
    test_df[feat_cols] = scaler.transform(test_df[feat_cols])

    print("Building Subnet Map (From Train Set Only)...")
    train_df['Src IP'] = train_df['Src IP'].astype(str).str.strip()
    train_df['Dst IP'] = train_df['Dst IP'].astype(str).str.strip()

    train_ips = pd.concat([train_df['Src IP'], train_df['Dst IP']]).unique()
    subnet_to_idx = {'<UNK>': UNK_SUBNET_ID}

    for ip in train_ips:
        key = _subnet_key(ip)
        if key not in subnet_to_idx:
            subnet_to_idx[key] = len(subnet_to_idx)

    print(f"Train Subnets: {len(subnet_to_idx)}", flush=True)

    print("🏗️ Constructing Train Graphs...")
    
    train_seqs = []
    for name, group in tqdm(train_df.groupby('time_idx', sort=True)):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: train_seqs.append(g)

    print("🏗️ Constructing Val Graphs...")
    val_seqs = []
    for name, group in tqdm(val_df.groupby('time_idx', sort=True)):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: val_seqs.append(g)

    print("🏗️ Constructing Test Graphs...")
    test_seqs = []
    for name, group in tqdm(test_df.groupby('time_idx', sort=True)):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: test_seqs.append(g)

    edge_dim = train_seqs[0].edge_attr.shape[1] if train_seqs else 1
    print(f"Edge Dim: {edge_dim}", flush=True)

    label_counts = train_df["Label"].value_counts().sort_index()
    full_counts = np.zeros(len(class_names))
    for i, count in label_counts.items():
        idx = int(i)
        if idx < len(full_counts):
            full_counts[idx] = count
    weights_cpu = 1.0 / (torch.sqrt(torch.tensor(full_counts, dtype=torch.float)) + 1.0)

    max_weight_limit = weights_cpu.min() * 20.0
    weights_cpu = torch.clamp(weights_cpu, max=max_weight_limit)

    weights_cpu = weights_cpu / weights_cpu.sum() * len(class_names)

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
            h = resolve_hparams(g, env=os.environ, dataset="ids2017")
            run_one_experiment(
                g,
                h,
                train_seqs,
                val_seqs,
                test_seqs,
                edge_dim,
                class_names,
                weights_cpu,
                DEVICE,
            )
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

# set -e

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# HP_GROUPS=DK_EXP1_BASE,DK_EXP2_DRIFT1,DK_EXP3_DRIFT2,DK_EXP4_NOEDGE \
# python -u MILAN/a2_4.py > logs_2020.txt 2>&1

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# NUM_WORKERS=0
# BATCH_SIZE=256
# HP_GROUPS=IDS_EXP1_BASE,IDS_EXP2_MAXCL,IDS_EXP3_SMALLH,IDS_EXP4_NOCL,IDS_EXP5_DROP \
# python -u MILAN/a3_1.py > logs_2017.txt 2>&1

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# NUM_WORKERS=0
# BATCH_SIZE=256
# HP_GROUPS=NB_EXP1_BASE,NB_EXP2_LONGK,NB_EXP3_STRONG,NB_EXP4_NOATT,NB_EXP5_LONGSEQ \
# python -u MILAN/a1_2.py > logs_nb15.txt 2>&1


# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# NUM_WORKERS=0
# BATCH_SIZE=256
# MIN_VAL_ATTACK_EDGES=50 \
# HP_GROUPS=IS_EXP1_BASE \
# python -u MILAN/train_2012.py > logs_2012.txt 2>&1

# tail -f logs_2017.txt
# tail -f logs_nb15.txt
