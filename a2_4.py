import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import hashlib
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import warnings
import random
import datetime
import gc
from tqdm import tqdm
from model import MILAN
from hparams_a3 import resolve_hparams
# === 关键修复：统一在顶部导入所有评估函数 ===
from analys import (
    _attack_best_threshold,
    _collect_attack_scores,
    FocalLoss,
    evaluate_comprehensive,
    evaluate_comprehensive_with_threshold,
    evaluate_with_threshold,
)
# from ROEN_Final import ROEN_Final

# ==========================================
# 辅助函数：哈希与子网键生成
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
    # 使用 MD5 生成确定性 Hash ID (int64)
    hash_obj = hashlib.md5(str(ip_str).encode())
    return int(hash_obj.hexdigest()[:15], 16)


def _parse_timestamp_series(ts):
    ts = ts.astype(str)
    sample = ts.replace({"nan": np.nan, "None": np.nan}).dropna().head(2000)
    if len(sample) == 0:
        return pd.to_datetime(ts, errors="coerce")

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

UNK_SUBNET_ID = 0

def get_subnet_id_safe(ip_str, subnet_map):
    key = _subnet_key(ip_str)
    return subnet_map.get(key, UNK_SUBNET_ID)

# ==========================================
# 1. 稀疏图构建函数 (Inductive & Robust)
# ==========================================
def create_graph_data_inductive(time_slice, subnet_map, label_encoder, time_window):
    time_slice = time_slice.copy()
    
    # 确保 IP 为字符串
    time_slice['Src IP'] = time_slice['Src IP'].astype(str).str.strip()
    time_slice['Dst IP'] = time_slice['Dst IP'].astype(str).str.strip()

    # 使用 Hash ID 替代 Global Map
    src_ids = time_slice['Src IP'].apply(get_ip_id_hash).values.astype(np.int64)
    dst_ids = time_slice['Dst IP'].apply(get_ip_id_hash).values.astype(np.int64)

    # 标签处理
    if label_encoder:
        try:
            labels = label_encoder.transform(time_slice['Label'].astype(str))
        except:
            labels = np.zeros(len(time_slice), dtype=int)
    else:
        labels = time_slice['Label'].values.astype(int)

    # 构建局部图索引
    all_nodes_in_slice = np.concatenate([src_ids, dst_ids])
    unique_nodes, inverse_indices = np.unique(all_nodes_in_slice, return_inverse=True)
    
    n_nodes = len(unique_nodes)
    src_local = inverse_indices[:len(src_ids)]
    dst_local = inverse_indices[len(src_ids):]
    
    # [优化] 转为 numpy array 以避免 PyTorch 警告
    edge_index = torch.tensor(np.array([src_local, dst_local]), dtype=torch.long)
    n_id = torch.tensor(unique_nodes, dtype=torch.long)
    
    # --- 节点特征工程 (4维) ---
    ones = torch.ones(edge_index.size(1), dtype=torch.float)
    in_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees.scatter_add_(0, edge_index[0], ones)
    in_degrees.scatter_add_(0, edge_index[1], ones)

    # 特征 3: 特权端口使用率
    src_port_col = 'Src Port' if 'Src Port' in time_slice.columns else 'Source Port'
    src_port = pd.to_numeric(time_slice.get(src_port_col, 0), errors='coerce').fillna(0).values
    is_priv_src = (src_port < 1024).astype(np.float32)
    
    priv_port_count = torch.zeros(n_nodes, dtype=torch.float)
    priv_port_count.scatter_add_(0, edge_index[0], torch.tensor(is_priv_src, dtype=torch.float))
    priv_ratio = priv_port_count / (out_degrees + 1e-6)

    # 特征 4: 流量聚合
    pkt_col = None
    for cand in ['Total Fwd Packets', 'Total Fwd Packet', 'Tot Fwd Pkts', 'Fwd Packets']:
        if cand in time_slice.columns:
            pkt_col = cand
            break
            
    if pkt_col is None:
        fwd_pkts = torch.zeros(edge_index.size(1), dtype=torch.float)
    else:
        # 已经是归一化后的数据
        fwd_pkts = torch.tensor(
            pd.to_numeric(time_slice[pkt_col], errors='coerce').fillna(0).values,
            dtype=torch.float,
        )
    
    node_pkt_sum = torch.zeros(n_nodes, dtype=torch.float)
    node_pkt_sum.scatter_add_(0, edge_index[0], fwd_pkts)

    is_hub = ((in_degrees + out_degrees) > 50).to(torch.float)
    
    x = torch.stack(
        [torch.log1p(in_degrees), torch.log1p(out_degrees), priv_ratio, node_pkt_sum, is_hub],
        dim=-1,
    ).float()

    # --- 子网 ID ---
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
    edge_attr_vals = (
        time_slice.drop(columns=drop_cols, errors="ignore")
        .select_dtypes(include=[np.number])
        .values
    )
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
    run_tag_prefix="",
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
    max_cl_edges = int(h.get("MAX_CL_EDGES", 10000))
    patience = int(h["PATIENCE"])
    min_delta = float(h["MIN_DELTA"])
    early_stop_metric = str(h["EARLY_STOP_METRIC"])
    cl_loss_weight = float(h["CL_LOSS_WEIGHT"])
    accum_steps = max(1, int(h.get("ACCUM_STEPS", 1)))
    drop_path = float(h.get("DROP_PATH", 0.1))
    dropedge_p = float(h.get("DROPEDGE_P", 0.2))
    cl_view1_dropedge_p = float(h.get("CL_VIEW1_DROPEDGE_P", 0.1))
    cl_view2_dropedge_p = float(h.get("CL_VIEW2_DROPEDGE_P", 0.2))
    warmup_epochs = max(0, int(h.get("WARMUP_EPOCHS", 5)))
    cosine_t0 = max(1, int(h.get("COSINE_T0", 10)))
    cosine_tmult = max(1, int(h.get("COSINE_TMULT", 1)))
    eta_min_ratio = float(h.get("ETA_MIN_RATIO", 0.01))
    target_far = float(h.get("TARGET_FAR", 0.01))

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
        f"KERNELS={kernels}, MAX_CL_EDGES={max_cl_edges}, NUM_EPOCHS={num_epochs}, PATIENCE={patience}, MIN_DELTA={min_delta}, "
        f"EARLY_STOP_METRIC={early_stop_metric}, CL_LOSS_WEIGHT={cl_loss_weight}, "
        f"ACCUM_STEPS={accum_steps}, DROP_PATH={drop_path}, DROPEDGE_P={dropedge_p}, "
        f"CL_VIEW1_DROPEDGE_P={cl_view1_dropedge_p}, CL_VIEW2_DROPEDGE_P={cl_view2_dropedge_p}, "
        f"WARMUP_EPOCHS={warmup_epochs}, COSINE_T0={cosine_t0}, COSINE_TMULT={cosine_tmult}, ETA_MIN_RATIO={eta_min_ratio}",
        flush=True,
    )

    model = MILAN(
        node_in=5,
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

    save_dir = "models/2020"
    os.makedirs(save_dir, exist_ok=True)
    kernel_tag = "-".join(str(k) for k in kernels)
    time_str = datetime.datetime.now().strftime("%m%d_%H%M")
    prefix = str(run_tag_prefix).strip()
    if prefix:
        run_tag = f"{prefix}_{group_tag}_seq{seq_len}_h{hidden}_k{kernel_tag}_cls{len(class_names)}_{time_str}"
    else:
        run_tag = f"{group_tag}_seq{seq_len}_h{hidden}_k{kernel_tag}_cls{len(class_names)}_{time_str}"
    best_model_path = os.path.join(save_dir, f"roen_final_best_{run_tag}.pth")
    print(f"Best model will be saved to: {best_model_path}", flush=True)

    print(f"Start Training on {device}...", flush=True)
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

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_opt_step += 1

            if torch.is_tensor(cl_loss):
                loop.set_postfix(loss=float(full_loss.detach().item()), cl_loss=float(cl_loss.detach().item()))
            else:
                loop.set_postfix(loss=float(full_loss.detach().item()))

        avg_train_loss = total_loss / max(1, len(train_loader))
        avg_cl_loss = total_cl_loss / max(1, cl_loss_steps)

        (
            avg_val_loss,
            val_acc,
            val_prec,
            val_rec,
            val_f1,
            val_far,
            val_auc,
            val_asa,
            y_true_val,
            y_pred_val,
            _y_probs_val,
        ) = evaluate_comprehensive(
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
            val_f1_weighted = float(f1_score(y_true_val, y_pred_val, average="weighted", zero_division=0))
        except Exception:
            val_f1_weighted = 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{group_tag}] Epoch {epoch+1:03d} | "
            f"Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val F1(macro): {val_f1:.4f} | "
            f"Val F1(weighted): {val_f1_weighted:.4f} | "
            f"ASA: {val_asa:.4f} | "
            f"Rec: {val_rec:.4f} | "
            f"FPR: {val_far:.4f} | "
            f"CL Loss: {avg_cl_loss:.4f} | "
            f"LR: {current_lr:.6f}",
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
            print(f"[{group_tag}] >>> ⭐ New Best Model Saved! ({metric_display}: {best_metric:.4f})", flush=True)
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"[{group_tag}] >>> Early Stopping ({metric_display} did not improve for {patience} epochs)", flush=True)
                break

    print(f"\n[{group_tag}] Loading Best Model from {best_model_path} for Final Testing...", flush=True)
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
                    model.load_state_dict(ckpt["state_dict"])
            else:
                model.load_state_dict(ckpt)
        except RuntimeError as e:
            print(f"[{group_tag}] ⚠️ Failed to load checkpoint. Using current weights. Error: {e}", flush=True)
    else:
        print(f"[{group_tag}] Warning: No best model file found! Using last epoch weights.", flush=True)

    final_acc, final_prec, final_rec, final_f1, final_far, final_auc, final_asa = evaluate_comprehensive(
        model, test_loader, device, class_names, average="macro"
    )
    print(
        f"[{group_tag}] Final Test -> ACC: {final_acc:.4f}, PREC: {final_prec:.4f}, F1(macro): {final_f1:.4f}, "
        f"Rec: {final_rec:.4f}, FAR: {final_far:.4f}, AUC: {final_auc:.4f}, ASA: {final_asa:.4f}",
        flush=True,
    )

    print(f"\n[{group_tag}] === Post-Training Threshold Optimization ===", flush=True)
    y_true_val, y_score_val = _collect_attack_scores(model, val_loader, device, class_names)
    optimal_thresh, val_f1, val_far, val_asa = _attack_best_threshold(y_true_val, y_score_val, max_far=target_far)
    print(
        f"[{group_tag}] Best Threshold found on VAL -> th={optimal_thresh:.4f}, Val F1={val_f1:.4f}, Val FAR={val_far:.4f}, Val ASA={val_asa:.4f}",
        flush=True,
    )

    print(f"\n[{group_tag}] === Final Evaluation on Test Set ===", flush=True)
    opt_acc, opt_prec, opt_rec, opt_f1, opt_far, opt_auc, opt_asa, final_labels, final_preds = evaluate_comprehensive_with_threshold(
        model, test_loader, device, class_names, threshold=optimal_thresh, average="macro"
    )
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
        f"[{group_tag}] Optimal Threshold Test -> ACC: {opt_acc:.4f}, PREC: {opt_prec:.4f}, "
        f"F1(macro): {opt_f1:.4f}, Rec: {opt_rec:.4f}, FAR: {opt_far:.4f}, "
        f"AUC: {opt_auc:.4f}, ASA: {opt_asa:.4f}",
        flush=True,
    )

    log_file = "experiment_results.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("Dataset,Group,SeqLen,Hidden,Heads,DropEdge,Threshold,F1,ASA,FAR,AUC\n")
        f.write(
            f"Darknet2020,{group_tag},{seq_len},{hidden},{heads},{dropedge_p},{optimal_thresh:.6f},"
            f"{opt_f1:.4f},{opt_asa:.4f},{opt_far:.4f},{opt_auc:.4f}\n"
        )

    labels_idx = list(range(len(class_names)))
    cm = confusion_matrix(final_labels, final_preds, labels=labels_idx)
    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0

    os.makedirs("png/2020", exist_ok=True)
    save_path = f"png/2020/FINAL_BEST_CM_{run_tag}_Thresh{optimal_thresh}.png"
    try:
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        annot = np.empty_like(cm_pct, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{int(cm[i, j])}\n({cm_pct[i, j]:.1f}%)"
        sns.heatmap(
            cm_pct,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            vmin=0.0,
            vmax=100.0,
        )
        plt.title(f"{group_tag} Confusion Matrix (Threshold={optimal_thresh})")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[{group_tag}] Final Confusion Matrix saved to {save_path}", flush=True)
    except Exception as e:
        print(f"[{group_tag}] Plotting failed: {e}", flush=True)

    print(f"[{group_tag}] Total Time: {time.time() - start_time:.2f}s", flush=True)
    return {
        "group": group_tag,
        "seq_len": seq_len,
        "hidden": hidden,
        "heads": heads,
        "dropedge_p": dropedge_p,
        "optimal_thresh": float(optimal_thresh),
        "opt_f1": float(opt_f1),
        "opt_asa": float(opt_asa),
        "opt_far": float(opt_far),
        "opt_auc": float(opt_auc),
        "final_f1": float(final_f1),
        "final_asa": float(final_asa),
        "final_far": float(final_far),
        "final_auc": float(final_auc),
    }

# ==========================================
# 3. 主训练流程
# ==========================================
def main():
    set_seed(int(os.getenv("SEED", "42")))
    # --- 配置 ---
    CSV_PATH = os.getenv("CSV_PATH", "data/CIC-Darknet2020/Darknet.csv")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {DEVICE}")

    # --- 1. 数据加载 ---
    print("Loading Data (CIC-Darknet2020)...")
    data = pd.read_csv(CSV_PATH) 
    
    data.drop(columns=['Label.1'], inplace=True, errors='ignore')
    data = data.dropna(subset=['Label', 'Timestamp']).copy()
    data["Label"] = data["Label"].astype(str).str.strip()
    data = data[data["Label"] != ""]
    data = data[~data["Label"].str.lower().isin(["nan", "none"])]
    
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'].astype(str))
    class_names = list(label_encoder.classes_)
    print(f"Classes: {class_names}")

    # 时间处理
    print("Processing Time...")
    data['Timestamp'] = _parse_timestamp_series(data['Timestamp'])
    data = data.dropna(subset=['Timestamp'])
    data = data.sort_values('Timestamp')
    data['time_idx'] = data['Timestamp'].dt.floor('min')

    blockwise = int(os.getenv("BLOCKWISE_SPLIT", "1"))
    if blockwise != 0:
        num_blocks = max(1, int(os.getenv("NUM_BLOCKS", "10")))
        block_train_ratio = float(os.getenv("BLOCK_TRAIN_RATIO", "0.8"))
        block_val_ratio = float(os.getenv("BLOCK_VAL_RATIO", "0.1"))
        min_block_times = max(3, int(os.getenv("BLOCK_MIN_TIMES", "30")))
        block_build_mode = str(os.getenv("BLOCK_BUILD_MODE", "adaptive")).strip().lower()
        require_all_classes = int(os.getenv("BLOCK_REQUIRE_ALL_CLASSES", "0")) != 0
        ensure_split_classes = int(os.getenv("BLOCK_ENSURE_SPLIT_CLASSES", "0")) != 0
        block_inner_split = str(os.getenv("BLOCK_INNER_SPLIT", "stratified_time")).strip().lower()

        min_val_classes = int(os.getenv("MIN_VAL_CLASSES", str(len(class_names))))
        min_test_classes = int(os.getenv("MIN_TEST_CLASSES", str(len(class_names))))
        min_val_per_class = int(os.getenv("MIN_VAL_PER_CLASS", "20"))
        min_test_per_class = int(os.getenv("MIN_TEST_PER_CLASS", "20"))

        print(
            f"Performing Block-wise Temporal Split (N={num_blocks}, mode={block_build_mode}, inner={block_inner_split})...",
            flush=True,
        )
        unique_times = data["time_idx"].drop_duplicates().values
        total_len = len(unique_times)
        if total_len < 3:
            raise ValueError(f"Not enough unique time points for splitting: total_len={total_len}")

        if block_build_mode == "adaptive":
            counts_by_time_all = (
                data.groupby(["time_idx", "Label"], sort=False)
                .size()
                .unstack(fill_value=0)
                .reindex(unique_times, fill_value=0)
            )
            counts_mat_all = counts_by_time_all.to_numpy(dtype=np.int64, copy=False)

            idx_splits = []
            start = 0
            current_counts = np.zeros(counts_mat_all.shape[1], dtype=np.int64)
            for i in range(total_len):
                current_counts += counts_mat_all[i]
                cur_len = i - start + 1

                if len(idx_splits) >= (num_blocks - 1):
                    continue

                has_all = bool(np.all(current_counts > 0))
                if cur_len >= min_block_times and ((not require_all_classes) or has_all):
                    idx_splits.append(np.arange(start, i + 1, dtype=np.int64))
                    start = i + 1
                    current_counts = np.zeros(counts_mat_all.shape[1], dtype=np.int64)

            if start < total_len:
                idx_splits.append(np.arange(start, total_len, dtype=np.int64))
            if len(idx_splits) == 0:
                idx_splits = [np.arange(0, total_len, dtype=np.int64)]
        else:
            idx_splits = np.array_split(np.arange(total_len, dtype=np.int64), num_blocks)

        time_to_block = {}
        for b, idxs in enumerate(idx_splits):
            for t in unique_times[idxs]:
                time_to_block[t] = b
        data["block_id"] = data["time_idx"].map(time_to_block).astype(np.int16)
        block_indices = data.groupby("block_id", sort=False).indices
        block_label_counts = data.groupby(["block_id", "Label"], sort=False).size().unstack(fill_value=0)
        block_label_counts = block_label_counts.reindex(range(num_blocks), fill_value=0)

        raw_groups = os.getenv("HP_GROUPS", "").strip()
        if raw_groups:
            raw = raw_groups.replace(";", ",").replace("\n", ",").replace(" ", ",")
            groups = [g.strip() for g in raw.split(",") if g.strip()]
        else:
            groups = [os.getenv("HP_GROUP", "").strip()]
        groups = [g for g in groups if str(g).strip() != ""]
        if len(groups) == 0:
            groups = ["BEST"]

        def _print_label_counts(df, split_name):
            vc = df['Label'].value_counts().sort_index()
            pairs = []
            for label_id, cnt in vc.items():
                label_name = class_names[int(label_id)] if int(label_id) < len(class_names) else str(label_id)
                pairs.append(f"{label_name}({int(label_id)}):{int(cnt)}")
            print(f"{split_name} Label Counts -> " + ", ".join(pairs), flush=True)

        results_by_group = {str(g).strip().upper() or "CUSTOM": [] for g in groups}
        overall_start = time.time()
        effective_blocks = 0

        for b, idxs in enumerate(idx_splits):
            print(f"[Block {b:02d}] Preparing...", flush=True)
            block_times = unique_times[idxs]
            if len(block_times) < min_block_times:
                print(f"[Block {b:02d}] Skip: too few minutes (len={len(block_times)})", flush=True)
                continue

            if require_all_classes:
                row = block_label_counts.loc[b].to_numpy(dtype=np.int64, copy=False)
                present = np.flatnonzero(row > 0).tolist()
                if len(present) < len(class_names):
                    missing = sorted(list(set(range(len(class_names))) - set(present)))
                    missing_names = [class_names[i] for i in missing if i < len(class_names)]
                    print(f"[Block {b:02d}] Skip: missing classes {missing} ({missing_names})", flush=True)
                    continue

            pos = block_indices.get(b, None)
            if pos is None or len(pos) == 0:
                print(f"[Block {b:02d}] Skip: empty block", flush=True)
                continue
            block_df = data.iloc[pos].copy()

            block_len = len(block_times)
            base_train_idx = int(block_len * block_train_ratio)
            base_val_idx = int(block_len * (block_train_ratio + block_val_ratio))
            base_train_idx = max(1, min(block_len - 2, base_train_idx))
            base_val_idx = max(base_train_idx + 1, min(block_len - 1, base_val_idx))

            counts_by_time = (
                block_df.groupby(["time_idx", "Label"], sort=False)
                .size()
                .unstack(fill_value=0)
                .reindex(block_times, fill_value=0)
            )
            counts_mat = counts_by_time.to_numpy(dtype=np.int64, copy=False)
            prefix = np.cumsum(counts_mat, axis=0, dtype=np.int64)
            prefix0 = np.vstack([np.zeros((1, prefix.shape[1]), dtype=np.int64), prefix])
            total_counts = prefix0[-1]

            def _interval_ok(start, end, min_classes, min_per_class):
                if end <= start:
                    return False
                counts = prefix0[end] - prefix0[start]
                if int(min_per_class) > 0:
                    eligible = counts >= int(min_per_class)
                else:
                    eligible = counts > 0
                return int(np.sum(eligible)) >= int(min_classes)

            def _test_ok(start, min_classes, min_per_class):
                if start >= block_len:
                    return False
                counts = total_counts - prefix0[start]
                if int(min_per_class) > 0:
                    eligible = counts >= int(min_per_class)
                else:
                    eligible = counts > 0
                return int(np.sum(eligible)) >= int(min_classes)

            train_df = None
            val_df = None
            test_df = None

            if block_inner_split in {"stratified_time", "stratified", "balanced"}:
                n_train = int(base_train_idx)
                n_val = int(max(1, base_val_idx - base_train_idx))
                n_test = int(max(1, block_len - base_val_idx))
                n_total = int(n_train + n_val + n_test)
                if n_total > block_len:
                    overflow = n_total - block_len
                    n_train = max(1, n_train - overflow)

                totals = total_counts.astype(np.float64, copy=False)
                rarity = 1.0 / (totals + 1.0)
                time_scores = counts_mat @ rarity
                order = np.argsort(-time_scores, kind="mergesort")

                caps = {"train": int(n_train), "val": int(n_val), "test": int(n_test)}
                chosen = {"train": [], "val": [], "test": []}
                set_counts = {
                    "train": np.zeros(counts_mat.shape[1], dtype=np.int64),
                    "val": np.zeros(counts_mat.shape[1], dtype=np.int64),
                    "test": np.zeros(counts_mat.shape[1], dtype=np.int64),
                }
                assigned = set()

                def _eligible(counts, min_classes, min_per_class):
                    if int(min_per_class) > 0:
                        ok = counts >= int(min_per_class)
                    else:
                        ok = counts > 0
                    return int(np.sum(ok)) >= int(min_classes)

                def _gain(name, vec, min_classes, min_per_class):
                    counts = set_counts[name]
                    if int(min_per_class) > 0:
                        need = np.maximum(0, int(min_per_class) - counts)
                        add = np.minimum(vec, need)
                        gain = float(np.sum(rarity * (add > 0)))
                    else:
                        missing = (counts == 0) & (vec > 0)
                        gain = float(np.sum(rarity[missing]))
                    fill = len(chosen[name]) / float(max(1, caps[name]))
                    return gain - 0.05 * fill

                for name, min_classes, min_per in (
                    ("val", min_val_classes, min_val_per_class),
                    ("test", min_test_classes, min_test_per_class),
                ):
                    if caps[name] <= 0:
                        continue
                    while (not _eligible(set_counts[name], min_classes, min_per)) and (len(chosen[name]) < caps[name]):
                        best_idx = None
                        best_gain = -1e9
                        for idx in order.tolist():
                            if idx in assigned:
                                continue
                            vec = counts_mat[idx]
                            if vec.sum() <= 0:
                                continue
                            g = _gain(name, vec, min_classes, min_per)
                            if g > best_gain:
                                best_gain = g
                                best_idx = idx
                        if best_idx is None or best_gain <= 0:
                            break
                        chosen[name].append(best_idx)
                        assigned.add(best_idx)
                        set_counts[name] = set_counts[name] + counts_mat[best_idx]

                for idx in order.tolist():
                    if idx in assigned:
                        continue
                    candidates = [k for k in ("train", "val", "test") if len(chosen[k]) < caps[k]]
                    if not candidates:
                        break
                    vec = counts_mat[idx]
                    if vec.sum() <= 0:
                        best = min(candidates, key=lambda k: len(chosen[k]) / float(max(1, caps[k])))
                    else:
                        best = max(
                            candidates,
                            key=lambda k: _gain(
                                k,
                                vec,
                                min_val_classes if k == "val" else (min_test_classes if k == "test" else 0),
                                min_val_per_class if k == "val" else (min_test_per_class if k == "test" else 0),
                            ),
                        )
                    chosen[best].append(idx)
                    assigned.add(idx)
                    set_counts[best] = set_counts[best] + vec

                remaining = [i for i in range(block_len) if i not in assigned]
                for name in ("train", "val", "test"):
                    need = caps[name] - len(chosen[name])
                    if need > 0 and remaining:
                        chosen[name].extend(remaining[:need])
                        for j in remaining[:need]:
                            set_counts[name] = set_counts[name] + counts_mat[j]
                        remaining = remaining[need:]

                if ensure_split_classes:
                    ok_val = _eligible(set_counts["val"], min_val_classes, min_val_per_class)
                    ok_test = _eligible(set_counts["test"], min_test_classes, min_test_per_class)
                    if (not ok_val) or (not ok_test):
                        print(
                            f"[Block {b:02d}] Stratified inner split cannot satisfy label coverage; fallback to temporal split.",
                            flush=True,
                        )
                        block_inner_split = "temporal"
                if block_inner_split in {"stratified_time", "stratified", "balanced"}:
                    train_times = block_times[np.array(sorted(chosen["train"]), dtype=np.int64)]
                    val_times = block_times[np.array(sorted(chosen["val"]), dtype=np.int64)]
                    test_times = block_times[np.array(sorted(chosen["test"]), dtype=np.int64)]
                    train_df = block_df[block_df["time_idx"].isin(train_times)].copy()
                    val_df = block_df[block_df["time_idx"].isin(val_times)].copy()
                    test_df = block_df[block_df["time_idx"].isin(test_times)].copy()

            if train_df is None or val_df is None or test_df is None:
                train_idx = int(base_train_idx)
                val_idx = int(base_val_idx)
                adjusted = False
                if ensure_split_classes:
                    step = max(1, int(block_len * 0.02))
                    found = False
                    train_idx_candidate = int(base_train_idx)
                    while train_idx_candidate >= 1 and not found:
                        val_idx_candidate = int(min(block_len - 1, max(train_idx_candidate + 1, base_val_idx)))
                        while val_idx_candidate > train_idx_candidate + 1 and not found:
                            if _interval_ok(train_idx_candidate, val_idx_candidate, min_val_classes, min_val_per_class) and _test_ok(
                                val_idx_candidate, min_test_classes, min_test_per_class
                            ):
                                train_idx = int(train_idx_candidate)
                                val_idx = int(val_idx_candidate)
                                found = True
                                break
                            val_idx_candidate = int(max(train_idx_candidate + 1, val_idx_candidate - step))
                        if found:
                            break
                        train_idx_candidate = int(max(1, train_idx_candidate - step))

                    if not found:
                        print(
                            f"[Block {b:02d}] Skip: cannot satisfy split label coverage "
                            f"(MIN_VAL_CLASSES={min_val_classes}, MIN_TEST_CLASSES={min_test_classes}, "
                            f"MIN_VAL_PER_CLASS={min_val_per_class}, MIN_TEST_PER_CLASS={min_test_per_class})",
                            flush=True,
                        )
                        del block_df
                        gc.collect()
                        continue

                    adjusted = (train_idx != base_train_idx) or (val_idx != base_val_idx)

                split_time_train = block_times[train_idx]
                split_time_val = block_times[val_idx]
                train_df = block_df[block_df["time_idx"] < split_time_train].copy()
                val_df = block_df[(block_df["time_idx"] >= split_time_train) & (block_df["time_idx"] < split_time_val)].copy()
                test_df = block_df[block_df["time_idx"] >= split_time_val].copy()

                if adjusted:
                    print(
                        f"[Block {b:02d}] Adjusted intra-block split for label coverage: "
                        f"MIN_VAL_CLASSES={min_val_classes}, MIN_TEST_CLASSES={min_test_classes}, "
                        f"MIN_VAL_PER_CLASS={min_val_per_class}, MIN_TEST_PER_CLASS={min_test_per_class}",
                        flush=True,
                    )
                print(
                    f"\n[Block {b:02d}] Splitting: Train < {split_time_train} <= Val < {split_time_val} <= Test",
                    flush=True,
                )
            else:
                print(
                    f"\n[Block {b:02d}] Splitting: stratified_time on time_idx (minutes, disjoint)",
                    flush=True,
                )

            del block_df
            print(
                f"[Block {b:02d}] Split Sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}",
                flush=True,
            )
            _print_label_counts(train_df, f"[Block {b:02d}] Train")
            _print_label_counts(val_df, f"[Block {b:02d}] Val")
            _print_label_counts(test_df, f"[Block {b:02d}] Test")

            if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
                print(f"[Block {b:02d}] Skip: empty split", flush=True)
                del train_df, val_df, test_df
                gc.collect()
                continue

            print(f"[Block {b:02d}] Performing Normalization (Inductive)...", flush=True)
            numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['Label', 'Timestamp', 'Src IP', 'Dst IP', 'Flow ID', 'Src Port', 'Dst Port', 'time_idx', 'block_id']
            feat_cols = [c for c in numeric_cols if c not in exclude_cols]

            for col in feat_cols:
                train_col = pd.to_numeric(train_df[col], errors="coerce")
                finite_mask = np.isfinite(train_col.to_numpy(dtype=np.float64, copy=False))
                if finite_mask.any():
                    finite_vals = train_col.to_numpy(dtype=np.float64, copy=False)[finite_mask]
                    finite_max = float(np.max(finite_vals))
                    finite_min = float(np.min(finite_vals))
                else:
                    finite_max = 0.0
                    finite_min = 0.0

                train_df[col] = train_df[col].replace([np.inf], finite_max).replace([-np.inf], finite_min)
                val_df[col] = val_df[col].replace([np.inf], finite_max).replace([-np.inf], finite_min)
                test_df[col] = test_df[col].replace([np.inf], finite_max).replace([-np.inf], finite_min)

            train_df[feat_cols] = train_df[feat_cols].fillna(0)
            val_df[feat_cols] = val_df[feat_cols].fillna(0)
            test_df[feat_cols] = test_df[feat_cols].fillna(0)

            for col in feat_cols:
                if train_df[col].max() > 100:
                    train_df[col] = np.log1p(train_df[col].abs())
                    val_df[col] = np.log1p(val_df[col].abs())
                    test_df[col] = np.log1p(test_df[col].abs())

            scaler = StandardScaler()
            train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
            val_df[feat_cols] = scaler.transform(val_df[feat_cols])
            test_df[feat_cols] = scaler.transform(test_df[feat_cols])
            print(f"[Block {b:02d}] Normalization Done.", flush=True)

            print(f"[Block {b:02d}] Building Subnet Map (From Train Set Only)...", flush=True)
            train_df['Src IP'] = train_df['Src IP'].astype(str).str.strip()
            train_df['Dst IP'] = train_df['Dst IP'].astype(str).str.strip()
            train_ips = pd.concat([train_df['Src IP'], train_df['Dst IP']]).unique()
            subnet_to_idx = {'<UNK>': UNK_SUBNET_ID}
            for ip in train_ips:
                key = _subnet_key(ip)
                if key not in subnet_to_idx:
                    subnet_to_idx[key] = len(subnet_to_idx)

            print(f"[Block {b:02d}] Constructing Train Graphs...", flush=True)
            train_grouped = train_df.groupby('time_idx', sort=True)
            train_seqs = []
            for name, group in tqdm(train_grouped, leave=False):
                g = create_graph_data_inductive(group, subnet_to_idx, None, name)
                if g:
                    train_seqs.append(g)

            print(f"[Block {b:02d}] Constructing Val Graphs...", flush=True)
            val_grouped = val_df.groupby('time_idx', sort=True)
            val_seqs = []
            for name, group in tqdm(val_grouped, leave=False):
                g = create_graph_data_inductive(group, subnet_to_idx, None, name)
                if g:
                    val_seqs.append(g)

            print(f"[Block {b:02d}] Constructing Test Graphs...", flush=True)
            test_grouped = test_df.groupby('time_idx', sort=True)
            test_seqs = []
            for name, group in tqdm(test_grouped, leave=False):
                g = create_graph_data_inductive(group, subnet_to_idx, None, name)
                if g:
                    test_seqs.append(g)

            if len(train_seqs) > 0:
                edge_dim = train_seqs[0].edge_attr.shape[1]
            elif len(test_seqs) > 0:
                edge_dim = test_seqs[0].edge_attr.shape[1]
            else:
                print(f"[Block {b:02d}] Skip: no graphs constructed", flush=True)
                del train_df, val_df, test_df, train_seqs, val_seqs, test_seqs
                gc.collect()
                continue

            label_counts = train_df["Label"].value_counts().sort_index()
            full_counts = np.zeros(len(class_names))
            for i, count in label_counts.items():
                idx = int(i)
                if idx < len(full_counts):
                    full_counts[idx] = count
            weights_cpu = 1.0 / (torch.sqrt(torch.tensor(full_counts, dtype=torch.float)) + 1.0)
            weights_cpu = weights_cpu / weights_cpu.sum() * len(class_names)

            block_prefix = f"b{b:02d}"
            for g in groups:
                group_tag = str(g).strip().upper() or "CUSTOM"
                try:
                    h = resolve_hparams(g, env=os.environ, dataset="darknet2020")
                    res = run_one_experiment(
                        g,
                        h,
                        train_seqs,
                        val_seqs,
                        test_seqs,
                        edge_dim,
                        class_names,
                        weights_cpu,
                        DEVICE,
                        run_tag_prefix=block_prefix,
                    )
                    if isinstance(res, dict):
                        results_by_group[group_tag].append(res)
                except torch.OutOfMemoryError as e:
                    print(f"[{block_prefix}][{group_tag}] CUDA OOM, skip this group. Error: {e}", flush=True)
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"[{block_prefix}][{group_tag}] CUDA OOM, skip this group. Error: {e}", flush=True)
                    else:
                        raise
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

            effective_blocks += 1
            del train_df, val_df, test_df, train_seqs, val_seqs, test_seqs
            gc.collect()

        print(f"\nBlock-wise Done. Effective Blocks: {effective_blocks}", flush=True)
        for group_tag, items in results_by_group.items():
            if not items:
                print(f"[{group_tag}] No valid blocks", flush=True)
                continue
            opt_f1s = [float(x.get("opt_f1", 0.0)) for x in items]
            opt_asas = [float(x.get("opt_asa", 0.0)) for x in items]
            opt_fars = [float(x.get("opt_far", 0.0)) for x in items]
            opt_aucs = [float(x.get("opt_auc", 0.0)) for x in items]
            print(
                f"[{group_tag}] Block-Avg (OptThresh on Val) -> "
                f"F1(macro): {float(np.mean(opt_f1s)):.4f}, "
                f"ASA: {float(np.mean(opt_asas)):.4f}, "
                f"FPR: {float(np.mean(opt_fars)):.4f}, "
                f"AUC: {float(np.mean(opt_aucs)):.4f} | "
                f"Blocks: {len(items)}",
                flush=True,
            )

        print(f"\nAll Experiments Done. Total Time: {time.time() - overall_start:.2f}s", flush=True)
        return

    print("Performing Temporal Split (8:1:1)...")
    unique_times = data["time_idx"].drop_duplicates().values
    total_len = len(unique_times)
    train_idx = int(total_len * 0.8)
    val_idx = int(total_len * 0.9)
    train_idx = max(1, min(total_len - 2, train_idx))
    val_idx = max(train_idx + 1, min(total_len - 1, val_idx))

    min_val_classes = int(os.getenv("MIN_VAL_CLASSES", "3"))
    min_test_classes = int(os.getenv("MIN_TEST_CLASSES", "3"))
    min_val_per_class = int(os.getenv("MIN_VAL_PER_CLASS", "20"))
    min_test_per_class = int(os.getenv("MIN_TEST_PER_CLASS", "20"))
    step = max(1, int(total_len * 0.01))
    adjusted = False

    counts_by_time = (
        data.groupby(["time_idx", "Label"], sort=False)
        .size()
        .unstack(fill_value=0)
        .reindex(unique_times, fill_value=0)
    )
    counts_mat = counts_by_time.to_numpy(dtype=np.int64, copy=False)
    prefix = np.cumsum(counts_mat, axis=0, dtype=np.int64)
    prefix0 = np.vstack([np.zeros((1, prefix.shape[1]), dtype=np.int64), prefix])
    total_counts = prefix0[-1]

    def _interval_ok(start, end, min_classes, min_per_class):
        if end <= start:
            return False
        counts = prefix0[end] - prefix0[start]
        if int(min_per_class) > 0:
            eligible = counts >= int(min_per_class)
        else:
            eligible = counts > 0
        return int(np.sum(eligible)) >= int(min_classes)

    def _test_ok(start, min_classes, min_per_class):
        if start >= total_len:
            return False
        counts = total_counts - prefix0[start]
        if int(min_per_class) > 0:
            eligible = counts >= int(min_per_class)
        else:
            eligible = counts > 0
        return int(np.sum(eligible)) >= int(min_classes)

    base_train_idx = int(train_idx)
    base_val_idx = int(val_idx)
    found = False
    train_idx_candidate = int(train_idx)
    while train_idx_candidate >= 1 and not found:
        val_idx_candidate = int(min(total_len - 1, max(train_idx_candidate + 1, base_val_idx)))
        while val_idx_candidate > train_idx_candidate + 1 and not found:
            if _interval_ok(train_idx_candidate, val_idx_candidate, min_val_classes, min_val_per_class) and _test_ok(
                val_idx_candidate, min_test_classes, min_test_per_class
            ):
                train_idx = int(train_idx_candidate)
                val_idx = int(val_idx_candidate)
                found = True
                break
            val_idx_candidate = int(max(train_idx_candidate + 1, val_idx_candidate - step))
        if found:
            break
        train_idx_candidate = int(max(1, train_idx_candidate - step))

    adjusted = (train_idx != base_train_idx) or (val_idx != base_val_idx)

    split_time_train = unique_times[train_idx]
    split_time_val = unique_times[val_idx]
    if adjusted:
        print(
            "Adjusted split to ensure label coverage: "
            f"MIN_VAL_CLASSES={min_val_classes}, MIN_TEST_CLASSES={min_test_classes}, "
            f"MIN_VAL_PER_CLASS={min_val_per_class}, MIN_TEST_PER_CLASS={min_test_per_class}",
            flush=True,
        )
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
        vc = df['Label'].value_counts().sort_index()
        pairs = []
        for label_id, cnt in vc.items():
            label_name = class_names[int(label_id)] if int(label_id) < len(class_names) else str(label_id)
            pairs.append(f"{label_name}({int(label_id)}):{int(cnt)}")
        print(f"{split_name} Label Counts -> " + ", ".join(pairs), flush=True)

    _print_label_counts(train_df, "Train")
    _print_label_counts(val_df, "Val")
    _print_label_counts(test_df, "Test")

    # === [归一化] ===
    print("Performing Normalization (Inductive)...")
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['Label', 'Timestamp', 'Src IP', 'Dst IP', 'Flow ID', 'Src Port', 'Dst Port', 'time_idx']
    feat_cols = [c for c in numeric_cols if c not in exclude_cols]
    
    for col in feat_cols:
        train_col = pd.to_numeric(train_df[col], errors="coerce")
        finite_mask = np.isfinite(train_col.to_numpy(dtype=np.float64, copy=False))
        if finite_mask.any():
            finite_vals = train_col.to_numpy(dtype=np.float64, copy=False)[finite_mask]
            finite_max = float(np.max(finite_vals))
            finite_min = float(np.min(finite_vals))
        else:
            finite_max = 0.0
            finite_min = 0.0

        train_df[col] = train_df[col].replace([np.inf], finite_max).replace([-np.inf], finite_min)
        val_df[col] = val_df[col].replace([np.inf], finite_max).replace([-np.inf], finite_min)
        test_df[col] = test_df[col].replace([np.inf], finite_max).replace([-np.inf], finite_min)

    train_df[feat_cols] = train_df[feat_cols].fillna(0)
    val_df[feat_cols] = val_df[feat_cols].fillna(0)
    test_df[feat_cols] = test_df[feat_cols].fillna(0)
 
    # Log1p
    for col in feat_cols:
        if train_df[col].max() > 100:
            train_df[col] = np.log1p(train_df[col].abs())
            val_df[col] = np.log1p(val_df[col].abs())
            test_df[col] = np.log1p(test_df[col].abs())
    
    # Fit on Train, Transform Val/Test
    scaler = StandardScaler()
    try:
        train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
        val_df[feat_cols] = scaler.transform(val_df[feat_cols])
        test_df[feat_cols] = scaler.transform(test_df[feat_cols])
    except ValueError as e:
        print("Error during scaling. Check for inf values.")
        raise e
         
    print("Normalization Done.")

    # --- 2. 构建 Subnet Map (Train Only) ---
    print("Building Subnet Map (From Train Set Only)...")
    train_df['Src IP'] = train_df['Src IP'].astype(str).str.strip()
    train_df['Dst IP'] = train_df['Dst IP'].astype(str).str.strip()
    
    train_ips = pd.concat([train_df['Src IP'], train_df['Dst IP']]).unique()
    subnet_to_idx = {'<UNK>': UNK_SUBNET_ID}
    
    for ip in train_ips:
        key = _subnet_key(ip)
        if key not in subnet_to_idx:
            subnet_to_idx[key] = len(subnet_to_idx)
            
    num_subnets = len(subnet_to_idx)
    print(f"Train Subnets: {num_subnets}")

    # --- 3. 构建 Graphs ---
    print("Constructing Train Graphs...")
    train_grouped = train_df.groupby('time_idx', sort=True)
    train_seqs = []
    for name, group in tqdm(train_grouped):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: train_seqs.append(g)

    print("Constructing Val Graphs...")
    val_grouped = val_df.groupby('time_idx', sort=True)
    val_seqs = []
    for name, group in tqdm(val_grouped):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: val_seqs.append(g)

    print("Constructing Test Graphs...")
    test_grouped = test_df.groupby('time_idx', sort=True)
    test_seqs = []
    for name, group in tqdm(test_grouped):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: test_seqs.append(g)

    print(f"Total Train Graphs: {len(train_seqs)}, Val Graphs: {len(val_seqs)}, Test Graphs: {len(test_seqs)}")

    if len(train_seqs) > 0:
        edge_dim = train_seqs[0].edge_attr.shape[1]
    elif len(test_seqs) > 0:
        edge_dim = test_seqs[0].edge_attr.shape[1]
    else:
        edge_dim = 1
    print(f"Edge Dim: {edge_dim}", flush=True)

    label_counts = train_df["Label"].value_counts().sort_index()
    full_counts = np.zeros(len(class_names))
    for i, count in label_counts.items():
        idx = int(i)
        if idx < len(full_counts):
            full_counts[idx] = count
    weights_cpu = 1.0 / (torch.sqrt(torch.tensor(full_counts, dtype=torch.float)) + 1.0)
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
            h = resolve_hparams(g, env=os.environ, dataset="darknet2020")
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
    os.makedirs('models', exist_ok=True)
    os.makedirs('models/2020', exist_ok=True)
    os.makedirs('png/2020', exist_ok=True)
    main()
# HP_GROUP=E9 python a2_4.py
# HP_GROUP=E4 python a2_4.py
# HP_GROUP=E14 python a2_4.py
# HP_GROUP=E15 python a2_4.py
# HP_GROUP=E16 python a2_4.py
# HP_GROUP=E17 python a2_4.py

# HP_GROUP=E1 python a3_1.py
# HP_GROUP=E2 python a3_1.py
# HP_GROUP=E3 python a3_1.py
# HP_GROUP=E4 python a3_1.py
# HP_GROUP=E8 python a3_1.py
# HP_GROUP=E1 python a1_2.py
# HP_GROUP=E2 python a1_2.py
# HP_GROUP=E3 python a1_2.py
# HP_GROUP=E4 python a1_2.py
# HP_GROUP=E8 python a1_2.py
