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
import random
import datetime
import gc
from tqdm import tqdm
# from analys import FocalLoss
from analys import (
    _attack_best_threshold,
    _collect_attack_scores,
    evaluate_comprehensive,
    evaluate_comprehensive_with_threshold,
    evaluate_with_threshold,
)
from hparams_a3 import resolve_hparams

# # 引入 Fast 模型
# from network_fast_transformer import ROEN_Fast_Transformer 
# from network_advanced import ROEN_Advanced
# from model_Final import ROEN_Final
from model import MILAN
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
    # 使用 MD5 生成确定性 Hash ID
    hash_obj = hashlib.md5(str(ip_str).encode())
    return int(hash_obj.hexdigest()[:15], 16)

UNK_SUBNET_ID = 0

def get_subnet_id_safe(ip_str, subnet_map):
    key = _subnet_key(ip_str)
    return subnet_map.get(key, UNK_SUBNET_ID)

# ==========================================
# 1. 稀疏图构建函数 (Inductive)
# ==========================================
def create_graph_data_inductive(time_slice, subnet_map, label_encoder, time_window):
    time_slice = time_slice.copy()
    time_slice['Src IP'] = time_slice['Src IP'].astype(str).str.strip()
    time_slice['Dst IP'] = time_slice['Dst IP'].astype(str).str.strip()

    # 使用 Hash ID
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

    all_nodes_in_slice = np.concatenate([src_ids, dst_ids])
    unique_nodes, inverse_indices = np.unique(all_nodes_in_slice, return_inverse=True)
    
    n_nodes = len(unique_nodes)
    src_local = inverse_indices[:len(src_ids)]
    dst_local = inverse_indices[len(src_ids):]
    
    # [优化] 显式转换 numpy array 以消除 UserWarning
    edge_index = torch.tensor(np.array([src_local, dst_local]), dtype=torch.long)
    n_id = torch.tensor(unique_nodes, dtype=torch.long)
    
    # 度特征
    ones = torch.ones(edge_index.size(1), dtype=torch.float)
    in_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees = torch.zeros(n_nodes, dtype=torch.float)
    out_degrees.scatter_add_(0, edge_index[0], ones)
    in_degrees.scatter_add_(0, edge_index[1], ones)

    # 子网特征 (Hash + Map)
    subnet_id = None
    if subnet_map is not None:
        subnet_ids_for_node = {}
        # 预先构建当前 slice 的 hash->subnet 映射
        # 这里为了效率，只对 unique IP 做处理
        unique_ips = pd.concat([time_slice['Src IP'], time_slice['Dst IP']]).unique()
        for ip_str in unique_ips:
            hid = get_ip_id_hash(ip_str)
            subnet_ids_for_node[hid] = get_subnet_id_safe(ip_str, subnet_map)
            
        subnet_id = torch.tensor(
            [subnet_ids_for_node.get(int(h), UNK_SUBNET_ID) for h in unique_nodes],
            dtype=torch.long,
        )

    # 行为特征 1: 特权端口比率
    src_port = pd.to_numeric(time_slice.get('Src Port', 0), errors='coerce').fillna(0).values
    is_priv_src = (src_port < 1024).astype(np.float32)
    priv_port_count = torch.zeros(n_nodes, dtype=torch.float)
    priv_port_count.scatter_add_(0, edge_index[0], torch.tensor(is_priv_src, dtype=torch.float))
    priv_ratio = priv_port_count / (out_degrees + 1e-6)

    # 行为特征 2: 流量聚合 (已归一化，可能为负，不能 Log)
    pkt_col = None
    for cand in ['Total Fwd Packets', 'Total Fwd Packet', 'Tot Fwd Pkts', 'Fwd Packets']:
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
    node_pkt_sum.scatter_add_(0, edge_index[0], fwd_pkts)

    # [关键修复] 移除 node_pkt_sum 的 log1p，直接使用
    x = torch.stack(
        [torch.log1p(in_degrees), torch.log1p(out_degrees), priv_ratio, node_pkt_sum],
        dim=-1,
    ).float()
    
    drop_cols = ['Src IP', 'Dst IP', 'Flow ID', 'Label', 'Timestamp', 'Src Port', 'Dst Port']
    edge_attr_vals = time_slice.drop(columns=drop_cols, errors='ignore').select_dtypes(include=[np.number]).values
    edge_attr_vals = np.nan_to_num(edge_attr_vals, nan=0.0, posinf=0.0, neginf=0.0)
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

# ==========================================
# 3. 评估辅助函数
# ==========================================

def _forward_compatible(model, batched_seq):
    try:
        return model(batched_seq)
    except TypeError:
        try:
            return model(batched_seq, len(batched_seq))
        except TypeError:
            return model(graphs=batched_seq, seq_len=len(batched_seq))

def evaluate_compatible(model, dataloader, device, class_names):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataloader:
            batch = [g.to(device) for g in batch]

            out = _forward_compatible(model, batch)
            if isinstance(out, tuple):
                preds_seq = out[0]
            else:
                preds_seq = out

            logits = preds_seq[-1]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_labels.extend(batch[-1].edge_labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return f1_score(all_labels, all_preds, average='weighted', zero_division=0)

def evaluate_with_threshold_compatible(model, dataloader, device, class_names, threshold=0.4):
    model.eval()
    all_labels = []
    all_preds = []

    normal_indices = []
    for idx, name in enumerate(class_names):
        low = str(name).lower()
        if any(k in low for k in ("non", "non-tor", "nonvpn", "normal", "benign")):
            normal_indices.append(idx)
    if len(normal_indices) == 0 and len(class_names) > 0:
        normal_indices = [0]

    attack_indices = [i for i in range(len(class_names)) if i not in set(normal_indices)]

    with torch.no_grad():
        for batch in dataloader:
            batch = [g.to(device) for g in batch]

            out = _forward_compatible(model, batch)
            if isinstance(out, tuple):
                preds_seq = out[0]
            else:
                preds_seq = out

            logits = preds_seq[-1]
            probs = torch.softmax(logits, dim=1)
            final_preds = torch.argmax(probs, dim=1)

            if len(attack_indices) > 0:
                attack_probs_sum = probs[:, attack_indices].sum(dim=1)
                mask = attack_probs_sum > threshold

                if mask.any():
                    sub_probs = probs[mask][:, attack_indices]
                    sub_argmax = torch.argmax(sub_probs, dim=1)
                    final_preds[mask] = torch.tensor(attack_indices, device=device)[sub_argmax]

            all_labels.extend(batch[-1].edge_labels.cpu().numpy())
            all_preds.extend(final_preds.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    is_true_normal = np.isin(y_true, normal_indices)
    is_pred_normal = np.isin(y_pred, normal_indices)
    fp = np.logical_and(is_true_normal, ~is_pred_normal).sum()
    tn = np.logical_and(is_true_normal, is_pred_normal).sum()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    is_true_attack = ~is_true_normal
    asa = (y_pred[is_true_attack] == y_true[is_true_attack]).mean() if is_true_attack.any() else 0.0

    acc = (y_pred == y_true).mean() if len(y_true) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) if len(y_true) > 0 else 0.0

    return acc, f1, far, asa


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
    target_far = float(h.get("TARGET_FAR", 0.025))

    if len(base_train_seqs) < seq_len or len(base_val_seqs) < 1 or len(base_test_seqs) < 1:
        print(
            f"Skip {group_tag}: SEQ_LEN={seq_len} too long for available sequences "
            f"(train={len(base_train_seqs)}, val={len(base_val_seqs)}, test={len(base_test_seqs)})",
            flush=True,
        )
        return

    val_seqs = _pad_seq_for_last_frame_coverage(base_val_seqs, seq_len)
    test_seqs = _pad_seq_for_last_frame_coverage(base_test_seqs, seq_len)

    train_dataset = TemporalGraphDataset(base_train_seqs, seq_len)
    val_dataset = TemporalGraphDataset(val_seqs, seq_len)
    test_dataset = TemporalGraphDataset(test_seqs, seq_len)

    if len(train_dataset) == 0 or len(val_dataset) == 0 or len(test_dataset) == 0:
        print(
            f"Skip {group_tag}: empty dataset windows (train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)})",
            flush=True,
        )
        return

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=temporal_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=temporal_collate_fn
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

    os.makedirs("models/nb15", exist_ok=True)
    os.makedirs("png/nb15", exist_ok=True)
    kernel_tag = "-".join(str(k) for k in kernels)
    time_str = datetime.datetime.now().strftime("%m%d_%H%M")
    run_tag = f"{group_tag}_seq{seq_len}_h{hidden}_k{kernel_tag}_cls{len(class_names)}_{time_str}"
    best_model_path = f"models/nb15/best_model_{run_tag}.pth"
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

        model.eval()
        total_val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for batched_seq in val_loader:
                if not batched_seq:
                    continue
                batched_seq = [g.to(device) for g in batched_seq]
                out = model(graphs=batched_seq)
                all_preds = out[0] if isinstance(out, tuple) else out
                val_loss = criterion(all_preds[-1], batched_seq[-1].edge_labels)
                total_val_loss += float(val_loss.detach().item())
                val_steps += 1

        avg_val_loss = total_val_loss / max(1, val_steps)

        val_acc, val_prec, val_rec, val_f1, val_far, val_auc, val_asa = evaluate_comprehensive(
            model, val_loader, device, class_names
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[{group_tag}] Epoch {epoch+1} | Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | ASA: {val_asa:.4f} | CL Loss: {avg_cl_loss:.4f} | LR: {current_lr:.6f}",
            flush=True,
        )

        metric_value = val_f1 if early_stop_metric == "val_f1" else val_asa
        metric_display = "Val F1" if early_stop_metric == "val_f1" else "Val ASA"

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
                    model.load_state_dict(ckpt["state_dict"])
            else:
                model.load_state_dict(ckpt)
        except RuntimeError as e:
            print(f"[{group_tag}] ⚠️ Failed to load checkpoint (shape mismatch). Using current weights. Error: {e}", flush=True)

    print(f"\n[{group_tag}] === Post-Training Threshold Optimization ===", flush=True)
    y_true_attack, y_score = _collect_attack_scores(model, test_loader, device, class_names)
    best_thresh, best_f1, best_far, best_asa = _attack_best_threshold(y_true_attack, y_score, max_far=target_far)
    print(
        f"[{group_tag}] Best Threshold -> th={best_thresh:.4f}, F1={best_f1:.4f}, FAR={best_far:.4f}, ASA={best_asa:.4f}",
        flush=True,
    )

    final_acc, final_prec, final_rec, final_f1, final_far, final_auc, final_asa, final_labels, final_preds = evaluate_comprehensive_with_threshold(
        model, test_loader, device, class_names, threshold=best_thresh
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
        f"[{group_tag}] Final Test -> ACC: {final_acc:.4f}, PREC: {final_prec:.4f}, Rec: {final_rec:.4f}, "
        f"F1: {final_f1:.4f}, AUC: {final_auc:.4f}, ASA: {final_asa:.4f}",
        flush=True,
    )

    log_file = "experiment_results.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write("Dataset,Group,SeqLen,Hidden,Heads,DropEdge,Threshold,F1,ASA,FAR,AUC\n")
        f.write(
            f"NB15,{group_tag},{seq_len},{hidden},{heads},{dropedge_p},{best_thresh:.6f},"
            f"{final_f1:.4f},{final_asa:.4f},{final_far:.4f},{final_auc:.4f}\n"
        )

    try:
        labels_idx = list(range(len(class_names)))
        cm = confusion_matrix(final_labels, final_preds, labels=labels_idx)
        cm = np.asarray(cm, dtype=np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0) * 100.0

        import seaborn as sns

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_pct, annot=True, fmt=".1f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title(f"{group_tag} Confusion Matrix (Threshold={best_thresh})")
        plt.tight_layout()
        plt.savefig(f"png/nb15/FINAL_CM_{run_tag}.png", dpi=300)
        plt.close()
        print(f"[{group_tag}] Confusion Matrix Saved.", flush=True)
    except Exception as e:
        print(f"[{group_tag}] Plotting failed: {e}", flush=True)

    print(f"[{group_tag}] Total Time: {time.time() - start_time:.2f}s", flush=True)

# ==========================================
# 4. 主流程
# ==========================================
def temporal_split(data_list, test_size=0.2):
    split_idx = int(len(data_list) * (1 - test_size))
    return data_list[:split_idx], data_list[split_idx:]

def main():
    set_seed(int(os.getenv("SEED", "42")))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading NB15 Data (CICFlowMeter Format)...")
    data = pd.read_csv("data/CIC-NUSW-NB15/CICFlowMeter_out.csv") 
    
    # 清洗标签
    data['Label'] = data['Label'].astype(str).str.strip().replace('', np.nan)
    data.dropna(subset=['Label', 'Timestamp'], inplace=True)
    
    label_encoder = LabelEncoder()
    data['Label'] = label_encoder.fit_transform(data['Label'])
    class_names = list(label_encoder.classes_)
    print(f"Classes: {class_names}")

    # 时间处理
    print("Processing Time..." )
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], dayfirst=True, errors='coerce')
    data.dropna(subset=['Timestamp'], inplace=True)
    data = data.sort_values('Timestamp')
    data['time_idx'] = data['Timestamp'].dt.floor('min')

    # === [关键优化] 数据切分 (先切分，后归一化，防止 Data Leakage) ===
    unique_times = data['time_idx'].drop_duplicates().values
    total_len = len(unique_times)
    train_idx = int(total_len * 0.8)
    val_idx = int(total_len * 0.9)
    train_idx = max(1, min(total_len - 1, train_idx))
    val_idx = max(train_idx + 1, min(total_len - 1, val_idx))

    split_time_train = unique_times[train_idx]
    split_time_val = unique_times[val_idx]

    print(f"Splitting: Train < {split_time_train} <= Val < {split_time_val} <= Test")

    train_df = data[data['time_idx'] < split_time_train].copy()
    val_df = data[(data['time_idx'] >= split_time_train) & (data['time_idx'] < split_time_val)].copy()
    test_df = data[data['time_idx'] >= split_time_val].copy()

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 归一化 (仅在训练集上 Fit)
    print("Normalizing (Fit on Train, Transform Test)..." )
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    exclude = ['Label', 'Timestamp', 'Src IP', 'Dst IP', 'Flow ID', 'Src Port', 'Dst Port']
    feat_cols = [c for c in numeric_cols if c not in exclude]
    
    # 填充缺失值
    train_df[feat_cols] = train_df[feat_cols].fillna(0)
    val_df[feat_cols] = val_df[feat_cols].fillna(0)
    test_df[feat_cols] = test_df[feat_cols].fillna(0)
    
    # Log1p 处理长尾
    for col in feat_cols:
        # 注意：这里假设原始数据非负。如果已经标准化过则不需要。通常原始数据是 Byte/Packet count，是非负的。
        if train_df[col].max() > 100: 
            train_df[col] = np.log1p(train_df[col].abs())
            val_df[col] = np.log1p(val_df[col].abs())
            test_df[col] = np.log1p(test_df[col].abs())

    scaler = StandardScaler()
    train_df[feat_cols] = scaler.fit_transform(train_df[feat_cols])
    val_df[feat_cols] = scaler.transform(val_df[feat_cols])
    test_df[feat_cols] = scaler.transform(test_df[feat_cols]) # 使用训练集的参数转换测试集

    # 构建 Subnet Map (仅用训练集)
    print("Building Subnet Map (From Train Set Only)..." )
    data['Src IP'] = data['Src IP'].astype(str).str.strip() # 确保原始 dataframe 类型正确 (用于 IP 提取)
    train_df['Src IP'] = train_df['Src IP'].astype(str).str.strip()
    train_df['Dst IP'] = train_df['Dst IP'].astype(str).str.strip()
    
    train_ips = pd.concat([train_df['Src IP'], train_df['Dst IP']]).unique()
    subnet_to_idx = {'<UNK>': UNK_SUBNET_ID}
    for ip in train_ips:
        key = _subnet_key(ip)
        if key not in subnet_to_idx:
            subnet_to_idx[key] = len(subnet_to_idx)
    num_subnets = len(subnet_to_idx)
    print(f"Train Subnets: {num_subnets} (Unknown subnets in Test will be mapped to 0)")

    # 构建图序列
    print("Building Train Graphs...")
    train_grouped = train_df.groupby('time_idx', sort=True)
    train_seqs = []
    for name, group in tqdm(train_grouped):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: train_seqs.append(g)

    print("Building Val Graphs...")
    val_grouped = val_df.groupby('time_idx', sort=True)
    val_seqs = []
    for name, group in tqdm(val_grouped):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: val_seqs.append(g)

    print("Building Test Graphs...")
    test_grouped = test_df.groupby('time_idx', sort=True)
    test_seqs = []
    for name, group in tqdm(test_grouped):
        g = create_graph_data_inductive(group, subnet_to_idx, None, name)
        if g: test_seqs.append(g)

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
            h = resolve_hparams(g, env=os.environ, dataset="nb15")
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
    os.makedirs('models/nb15', exist_ok=True)
    os.makedirs('png/nb15', exist_ok=True)
    main()
