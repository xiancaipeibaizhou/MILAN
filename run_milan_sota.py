import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import joblib
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, fbeta_score
)
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from datetime import datetime
from tqdm import tqdm
# 导入超参解析和你的模型组
from hparams_a3 import resolve_hparams
from model import MILAN
from ablation_models import (
    MILAN_WoGlobal, MILAN_WoLocal, MILAN_WoGating, 
    MILAN_WoEdgeAug, MILAN_StandardTransformer
)

# ==========================================
# 1. 核心时序数据加载逻辑 (100% 还原你的设计)
# ==========================================
class TemporalGraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data_seq, seq_len=10):
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

def get_normal_indices(class_names):
    """智能识别所有属于正常的类别索引 (兼容 Darknet2020 的 Non-Tor/NonVPN)"""
    if class_names is None: return [0]
    normals = []
    for idx, name in enumerate(class_names):
        name_lower = str(name).lower().replace('-', '').replace('_', '').replace(' ', '')
        if any(k in name_lower for k in ['benign', 'normal', 'nonvpn', 'nontor']):
            normals.append(idx)
    # 如果找不到，保底返回 0
    return normals if len(normals) > 0 else [0]
# ==========================================
# 2. 动态 F2 阈值与统一评估模块
# ==========================================
def find_best_macro_f1_threshold_and_predict(y_true_val, y_prob_val, y_prob_test, normal_indices):
    """多正常类兼容版的 Macro F1 阈值搜寻"""
    # 攻击概率 = 1 - 所有正常类概率之和
    normal_probs_val = np.sum(y_prob_val[:, normal_indices], axis=1)
    attack_probs_val = 1.0 - normal_probs_val
    y_true_val_bin = (~np.isin(y_true_val, normal_indices)).astype(int)
    
    candidates = np.unique(np.quantile(attack_probs_val, np.linspace(0.0, 1.0, 101)))
    best_th, best_macro_f1, best_far = 0.5, -1.0, 1.0
    
    for th in candidates:
        y_pred_val_sim = np.argmax(y_prob_val, axis=-1)
        
        # 1. 攻击概率低于阈值，强判为概率最大的那个正常类
        best_normal_class_val = np.array(normal_indices)[np.argmax(y_prob_val[:, normal_indices], axis=1)]
        y_pred_val_sim[attack_probs_val < th] = best_normal_class_val[attack_probs_val < th]
        
        # 2. 攻击概率高于阈值，但原本预测为正常的，强判为概率最大的攻击类
        mask = (attack_probs_val >= th) & np.isin(y_pred_val_sim, normal_indices)
        if mask.any():
            probs_copy = y_prob_val.copy()
            probs_copy[:, normal_indices] = -1.0  # 将正常类概率降维打击
            y_pred_val_sim[mask] = np.argmax(probs_copy[mask], axis=-1)
            
        y_pred_val_bin = (~np.isin(y_pred_val_sim, normal_indices)).astype(int)
        fp = np.logical_and(y_true_val_bin == 0, y_pred_val_bin == 1).sum()
        tn = np.logical_and(y_true_val_bin == 0, y_pred_val_bin == 0).sum()
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        macro_f1 = f1_score(y_true_val, y_pred_val_sim, average='macro', zero_division=0)
        
        if macro_f1 > best_macro_f1 or (macro_f1 == best_macro_f1 and far < best_far):
            best_macro_f1, best_th, best_far = macro_f1, th, far

    # 应用到测试集
    test_preds = np.argmax(y_prob_test, axis=-1)
    normal_probs_test = np.sum(y_prob_test[:, normal_indices], axis=1)
    attack_probs_test = 1.0 - normal_probs_test
    
    best_normal_class_test = np.array(normal_indices)[np.argmax(y_prob_test[:, normal_indices], axis=1)]
    test_preds[attack_probs_test < best_th] = best_normal_class_test[attack_probs_test < best_th]
    
    mask_to_attack = (attack_probs_test >= best_th) & np.isin(test_preds, normal_indices)
    if mask_to_attack.any():
        probs_copy = y_prob_test.copy()
        probs_copy[:, normal_indices] = -1.0 
        test_preds[mask_to_attack] = np.argmax(probs_copy[mask_to_attack], axis=-1)
        
    return test_preds, best_th, best_macro_f1, best_far

def compute_all_metrics(y_true, y_pred, y_prob=None, class_names=None, normal_indices=None):
    metrics = {}
    metrics['ACC'] = accuracy_score(y_true, y_pred)
    metrics['APR'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['RE'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['F1 (Weighted)'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['F1 (Macro)'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # ---------------- 安全版 AUC 计算 ----------------
    if y_prob is not None and y_prob.ndim == 2:
        present_classes = np.unique(y_true)
        if len(present_classes) < 2:
            metrics['AUC'] = float('nan')
        elif len(present_classes) == 2:
            pos_class = present_classes[1]
            y_true_bin = (y_true == pos_class).astype(int)
            try: metrics['AUC'] = roc_auc_score(y_true_bin, y_prob[:, pos_class])
            except: metrics['AUC'] = float('nan')
        else:
            aucs = []
            for c in present_classes:
                y_true_bin = (y_true == c).astype(int)
                if len(np.unique(y_true_bin)) == 2:
                    try: aucs.append(roc_auc_score(y_true_bin, y_prob[:, c]))
                    except: pass
            metrics['AUC'] = np.mean(aucs) if len(aucs) > 0 else float('nan')
    else: 
        metrics['AUC'] = float('nan')

    # ---------------- 精准计算 FAR 和 ASA (支持多正常类) ----------------
    num_classes = len(class_names) if class_names is not None else int(np.max(y_true)) + 1
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    
    if normal_indices is None: normal_indices = [0]
    
    is_true_normal = np.isin(y_true, normal_indices)
    is_pred_normal = np.isin(y_pred, normal_indices)
    
    # 误报：原本是正常类，被预测成了非正常类
    fp = np.logical_and(is_true_normal, ~is_pred_normal).sum()
    # 真阴：原本是正常类，预测也是正常类
    tn = np.logical_and(is_true_normal, is_pred_normal).sum()
    metrics['FAR'] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    
    # 攻击拦截率：原本是非正常类
    attack_mask = ~is_true_normal
    attack_total = attack_mask.sum()
    # 并且预测正确的非正常类 (必须是预测对了具体的攻击类别才算拦截成功)
    attack_correct = np.logical_and(attack_mask, y_true == y_pred).sum()
    metrics['ASA'] = float(attack_correct / attack_total) if attack_total > 0 else 0.0

    return metrics, cm

def plot_and_save_confusion_matrix(cm, target_names, save_path):
    clean_target_names = [str(name).replace('\x96', '-').replace('\u2013', '-') for name in target_names]
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) 
    
    num_classes = len(clean_target_names)
    fig_width = max(10, num_classes * 1.0)
    fig_height = max(8, num_classes * 0.8)
    
    plt.figure(figsize=(fig_width, fig_height))
    sns.set(font_scale=1.0) 
    
    annot = np.empty_like(cm_norm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{int(cm[i, j])}\n({cm_norm[i, j]*100:.1f}%)" if cm[i, j] > 0 else ""

    sns.heatmap(cm_norm, annot=annot, fmt="", cmap='Blues', cbar=True,
                xticklabels=clean_target_names, yticklabels=clean_target_names, vmin=0.0, vmax=1.0)
    
    plt.title('Normalized Confusion Matrix', pad=20, fontsize=14)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# 3. 评估获取 Logits
# ==========================================
@torch.no_grad()
def get_eval_predictions(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for batched_seq in loader:
        batched_seq = [g.to(device) for g in batched_seq]
        out = model(batched_seq)
        all_preds, _ = out if isinstance(out, tuple) else (out, None)
        
        logits = all_preds[-1]
        probs = torch.softmax(logits, dim=-1)
        
        # 同步应用你的 DropEdge mask 保障维度对齐
        edge_masks = getattr(model, "_last_edge_masks", None)
        if edge_masks is not None and len(edge_masks) > 0 and edge_masks[-1] is not None:
            labels = batched_seq[-1].edge_labels[edge_masks[-1]]
        else:
            labels = batched_seq[-1].edge_labels
            
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
    return np.concatenate(all_labels), np.concatenate(all_probs)

# ==========================================
# 4. 主流程
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='unsw_nb15')
    parser.add_argument('--data_dir', type=str, default='../processed_data')
    parser.add_argument('--variant', type=str, default='MILAN', 
                        choices=['MILAN', 'WoGlobal', 'WoLocal', 'WoGating', 'WoEdgeAug', 'StandardTransformer'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取 HP_GROUPS 超参 (完美对接你的 hparams_a3.py)
    group_str = os.getenv("HP_GROUPS", "NB_EXP1_BASE").split(",")[0].strip()
    h = resolve_hparams(group_str, env=os.environ, dataset=args.dataset)
    
    seq_len = int(h["SEQ_LEN"])
    batch_size = int(h["BATCH_SIZE"])
    num_epochs = int(h["NUM_EPOCHS"])
    lr = float(h["LR"])
    hidden = int(h["HIDDEN"])
    heads = int(h["HEADS"])
    kernels = list(h["KERNELS"])
    max_cl_edges = int(h.get("MAX_CL_EDGES", 8192))
    patience = int(h["PATIENCE"])
    cl_loss_weight = float(h["CL_LOSS_WEIGHT"])
    accum_steps = max(1, int(h.get("ACCUM_STEPS", 1)))
    drop_path = float(h.get("DROP_PATH", 0.1))
    dropedge_p = float(h.get("DROPEDGE_P", 0.2))
    warmup_epochs = max(0, int(h.get("WARMUP_EPOCHS", 5)))

    # 规范输出路径
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{args.variant}_{group_str}_dim{hidden}_seq{seq_len}"
    save_dir = os.path.join("results", args.dataset, exp_name, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    print(f"📁 Outputs will be saved to: {save_dir}")

    # 读取干净的图数据
    dataset_path = os.path.join(args.data_dir, args.dataset)
    train_graphs = torch.load(os.path.join(dataset_path, "train_graphs.pt"), weights_only=False)
    val_graphs = torch.load(os.path.join(dataset_path, "val_graphs.pt"), weights_only=False)
    test_graphs = torch.load(os.path.join(dataset_path, "test_graphs.pt"), weights_only=False)

    # 还原你的 Smooth Category Weights 逻辑
    counts = np.zeros(100) # 预设足够大的空间
    for g in train_graphs:
        counts += np.bincount(g.edge_labels.numpy(), minlength=100)
    num_classes = int(np.max(np.nonzero(counts))) + 1
    counts = counts[:num_classes]
    
    weights_cpu = 1.0 / (np.sqrt(counts) + 1.0)
    weights_cpu = torch.tensor(weights_cpu / weights_cpu.sum() * num_classes, dtype=torch.float32)

    # 获取名字和维度
    label_enc_path = os.path.join(dataset_path, "label_encoder.pkl")
    class_names = joblib.load(label_enc_path).classes_ if os.path.exists(label_enc_path) else [f"Class_{i}" for i in range(num_classes)]
    node_dim, edge_dim = train_graphs[0].x.shape[1], train_graphs[0].edge_attr.shape[1]

    # 时序 Dataloader
    train_loader = DataLoader(TemporalGraphDataset(train_graphs, seq_len), batch_size=batch_size, shuffle=True, collate_fn=temporal_collate_fn)
    val_loader = DataLoader(TemporalGraphDataset(val_graphs, seq_len), batch_size=batch_size, shuffle=False, collate_fn=temporal_collate_fn)
    test_loader = DataLoader(TemporalGraphDataset(test_graphs, seq_len), batch_size=batch_size, shuffle=False, collate_fn=temporal_collate_fn)

    # 动态模型分发
    model_kwargs = {
        "node_in": node_dim, "edge_in": edge_dim, "hidden": hidden, "num_classes": num_classes,
        "seq_len": seq_len, "heads": heads, "dropout": 0.3, "max_cl_edges": max_cl_edges,
        "kernels": kernels, "drop_path": drop_path, "dropedge_p": dropedge_p,
    }
    
    print(f"🚀 Initializing Model Variant: {args.variant}")
    if args.variant == "MILAN": model = MILAN(**model_kwargs).to(device)
    elif args.variant == "WoGlobal": model = MILAN_WoGlobal(**model_kwargs).to(device)
    elif args.variant == "WoLocal": model = MILAN_WoLocal(**model_kwargs).to(device)
    elif args.variant == "WoGating": model = MILAN_WoGating(**model_kwargs).to(device)
    elif args.variant == "WoEdgeAug": model = MILAN_WoEdgeAug(**model_kwargs).to(device)
    elif args.variant == "StandardTransformer": model = MILAN_StandardTransformer(**model_kwargs).to(device)

    # 还原你的 Optimizer 与 Scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=h.get("COSINE_T0", 10), T_mult=h.get("COSINE_TMULT", 1), eta_min=lr*0.01)
    criterion = nn.CrossEntropyLoss(weight=weights_cpu.to(device))

    # 训练循环控制参数
    num_train_steps = max(1, len(train_loader))
    warmup_total_steps = int(warmup_epochs) * ((num_train_steps + accum_steps - 1) // accum_steps)
    global_opt_step = 0
    best_val_f1 = -1.0
    patience_cnt = 0
    training_log = []

    print("🔥 Start Training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss, total_cl_loss, cl_loss_steps = 0.0, 0.0, 0
        optimizer.zero_grad(set_to_none=True)
        
        for step, batched_seq in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)):
            batched_seq = [g.to(device) for g in batched_seq]
            
            # 还原你的 CL Loss 逻辑
            out = model(batched_seq)
            all_preds, cl_loss = out if isinstance(out, tuple) else (out, None)
            
            edge_masks = getattr(model, "_last_edge_masks", None)
            if edge_masks is not None and len(edge_masks) > 0 and edge_masks[-1] is not None:
                last_frame_labels = batched_seq[-1].edge_labels[edge_masks[-1]]
            else:
                last_frame_labels = batched_seq[-1].edge_labels
                
            main_loss = criterion(all_preds[-1], last_frame_labels)
            full_loss = main_loss + cl_loss_weight * cl_loss if torch.is_tensor(cl_loss) else main_loss
            loss = full_loss / float(accum_steps)
            loss.backward()
            
            total_loss += full_loss.item()
            if torch.is_tensor(cl_loss):
                total_cl_loss += cl_loss.item()
                cl_loss_steps += 1
                
            if ((step + 1) % accum_steps == 0) or ((step + 1) == num_train_steps):
                if warmup_total_steps > 0 and global_opt_step < warmup_total_steps:
                    warm_lr = lr * float(global_opt_step + 1) / float(warmup_total_steps)
                    for pg in optimizer.param_groups: pg["lr"] = warm_lr
                else:
                    progress = float(epoch) + float(step + 1) / float(num_train_steps)
                    scheduler.step(progress - float(warmup_epochs))
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_opt_step += 1

        # 验证评估
        val_true, val_probs = get_eval_predictions(model, val_loader, device)
        val_preds_raw = np.argmax(val_probs, axis=-1)
        val_f1_macro = f1_score(val_true, val_preds_raw, average='macro', zero_division=0)
        
        log_line = f"Epoch {epoch+1:03d} | Loss: {total_loss/num_train_steps:.4f} | CL: {total_cl_loss/max(1, cl_loss_steps):.4f} | Val F1: {val_f1_macro:.4f}"
        print(log_line)
        training_log.append(log_line)
        
        if val_f1_macro > best_val_f1 + h.get("MIN_DELTA", 0.0):
            best_val_f1 = val_f1_macro
            patience_cnt = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
        else:
            patience_cnt += 1
            if patience_cnt >= patience: break

    # ====================
    # 测试与动态阈值应用
    # ====================
    print("\n[Testing] Evaluating Best Model...")
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth")))
    
    val_true, val_prob = get_eval_predictions(model, val_loader, device)
    test_true, test_prob = get_eval_predictions(model, test_loader, device)
    
    # 1. 先获取该数据集的真实正常类索引
    normal_indices = get_normal_indices(class_names)
        
    # 2. 把 normal_indices 传给这两个函数
    test_pred, best_th, val_macro, val_far = find_best_macro_f1_threshold_and_predict(val_true, val_prob, test_prob, normal_indices)
    metrics, cm = compute_all_metrics(test_true, test_pred, test_prob, class_names, normal_indices)
    plot_and_save_confusion_matrix(cm, class_names, os.path.join(save_dir, f"cm_thresh_{best_th:.2f}.png"))
    
    # ====================
    # 结果写入与保存
    # ====================
    # 1. 保存当前实验的详细指标
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        f.write(f"=== {exp_name} (Thresh: {best_th:.2f}) ===\n")
        for k, v in metrics.items(): 
            f.write(f"{k}: {v:.4f}\n")
            
    # 2. 保存训练过程的 Loss 和 F1 变化 (用于画图)
    with open(os.path.join(save_dir, "training_history.log"), "w") as f:
        for log_line in training_log:
            f.write(log_line + "\n")
            
    # 3. 追加写入全局 CSV 记录表 (可用于论文画表)
    # ！！就是这里，必须先定义 csv_file 变量 ！！
    csv_file = "milan_ablations_results.csv" 
    
    # 如果文件不存在，先写入表头
    if not os.path.isfile(csv_file):
        with open(csv_file, "w") as f:
            f.write("Dataset,Variant,Group,Threshold,ACC,APR,RE,F1_Macro,F1_Weighted,AUC,ASA,FAR\n")
            
    # 追加写入当前实验的具体数值
    with open(csv_file, "a") as f:
        f.write(f"{args.dataset},{args.variant},{group_str},{best_th:.4f},"
                f"{metrics['ACC']:.4f},{metrics['APR']:.4f},{metrics['RE']:.4f},"
                f"{metrics['F1 (Macro)']:.4f},{metrics['F1 (Weighted)']:.4f},{metrics['AUC']:.4f},"
                f"{metrics['ASA']:.4f},{metrics['FAR']:.4f}\n")

if __name__ == "__main__":
    main()