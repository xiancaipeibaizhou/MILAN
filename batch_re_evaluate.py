import os
import re
import glob
import argparse
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# 导入你现有的超参和模型组件
from hparams_a3 import resolve_hparams
from model import MILAN
from ablation_models import (
    MILAN_WoGlobal, MILAN_WoLocal, MILAN_WoGating, 
    MILAN_WoEdgeAug, MILAN_StandardTransformer
)

# ==========================================
# 1. 混淆矩阵绘图函数
# ==========================================
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
# 2. 数据加载逻辑 (带缓存)
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
        
        edge_masks = getattr(model, "_last_edge_masks", None)
        if edge_masks is not None and len(edge_masks) > 0 and edge_masks[-1] is not None:
            labels = batched_seq[-1].edge_labels[edge_masks[-1]]
        else:
            labels = batched_seq[-1].edge_labels
            
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)

# ==========================================
# 3. 纯净版阈值搜寻与评估
# ==========================================
from run_milan_sota import find_best_macro_f1_threshold_and_predict,compute_all_metrics,get_normal_indices
# ==========================================
# 4. 批量处理与数据缓存逻辑
# ==========================================
DATASET_CACHE = {}

def get_dataset_info(data_dir, dataset_name, seq_len):
    cache_key = f"{dataset_name}_{seq_len}"
    if cache_key in DATASET_CACHE:
        return DATASET_CACHE[cache_key]

    dataset_path = os.path.join(data_dir, dataset_name)
    print(f"  [Disk Load] Loading {dataset_name} for the first time...")
    
    train_graphs = torch.load(os.path.join(dataset_path, "train_graphs.pt"), weights_only=False)
    val_graphs = torch.load(os.path.join(dataset_path, "val_graphs.pt"), weights_only=False)
    test_graphs = torch.load(os.path.join(dataset_path, "test_graphs.pt"), weights_only=False)

    counts = np.zeros(100) 
    for g in train_graphs: counts += np.bincount(g.edge_labels.numpy(), minlength=100)
    num_classes = int(np.max(np.nonzero(counts))) + 1

    label_enc_path = os.path.join(dataset_path, "label_encoder.pkl")
    class_names = joblib.load(label_enc_path).classes_ if os.path.exists(label_enc_path) else [f"Class_{i}" for i in range(num_classes)]
    node_dim, edge_dim = train_graphs[0].x.shape[1], train_graphs[0].edge_attr.shape[1]

    val_loader = DataLoader(TemporalGraphDataset(val_graphs, seq_len), batch_size=32, shuffle=False, collate_fn=temporal_collate_fn)
    test_loader = DataLoader(TemporalGraphDataset(test_graphs, seq_len), batch_size=32, shuffle=False, collate_fn=temporal_collate_fn)

    DATASET_CACHE[cache_key] = (val_loader, test_loader, num_classes, class_names, node_dim, edge_dim)
    del train_graphs
    return DATASET_CACHE[cache_key]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results', help="Root directory containing model results")
    parser.add_argument('--data_dir', type=str, default='../processed_data', help="Directory of processed datasets")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    search_pattern = os.path.join(args.results_dir, "**", "best_model.pth")
    model_paths = glob.glob(search_pattern, recursive=True)
    
    if not model_paths:
        print(f"❌ No models found in '{args.results_dir}'.")
        return
        
    print(f"🎯 Found {len(model_paths)} models to re-evaluate.\n")

    csv_file = "batch_re_evaluate_results.csv"
    if not os.path.isfile(csv_file):
        with open(csv_file, "w") as f:
            f.write("Dataset,Variant,Group,Hidden,Seq,Threshold,ACC,APR,RE,F1_Macro,F1_Weighted,AUC,ASA,FAR\n")

    for idx, model_path in enumerate(model_paths, 1):
        print("="*60)
        print(f"[{idx}/{len(model_paths)}] Processing: {model_path}")
        
        parts = model_path.split(os.sep)
        if len(parts) < 4:
            continue
            
        dataset_name = parts[-4]
        exp_name = parts[-3]
        model_dir = os.path.dirname(model_path)
        
        match = re.match(r"^(.*?)_(.*)_dim(\d+)_seq(\d+)$", exp_name)
        if not match:
            print(f"  ⚠️ Cannot parse exp_name '{exp_name}', skipping...")
            continue
            
        variant = match.group(1)
        group_str = match.group(2)
        hidden = int(match.group(3))
        seq_len = int(match.group(4))
        
        h = resolve_hparams(group_str, env=os.environ, dataset=dataset_name)
        heads = int(h["HEADS"])
        kernels = list(h["KERNELS"])
        max_cl_edges = int(h.get("MAX_CL_EDGES", 8192))
        drop_path = float(h.get("DROP_PATH", 0.1))
        dropedge_p = float(h.get("DROPEDGE_P", 0.2))

        try:
            val_loader, test_loader, num_classes, class_names, node_dim, edge_dim = get_dataset_info(args.data_dir, dataset_name, seq_len)
        except Exception as e:
            print(f"  ❌ Error loading dataset {dataset_name}: {e}")
            continue

        model_kwargs = {
            "node_in": node_dim, "edge_in": edge_dim, "hidden": hidden, "num_classes": num_classes,
            "seq_len": seq_len, "heads": heads, "dropout": 0.3, "max_cl_edges": max_cl_edges,
            "kernels": kernels, "drop_path": drop_path, "dropedge_p": dropedge_p,
        }
        
        if variant == "MILAN": model = MILAN(**model_kwargs).to(device)
        elif variant == "WoGlobal": model = MILAN_WoGlobal(**model_kwargs).to(device)
        elif variant == "WoLocal": model = MILAN_WoLocal(**model_kwargs).to(device)
        elif variant == "WoGating": model = MILAN_WoGating(**model_kwargs).to(device)
        elif variant == "WoEdgeAug": model = MILAN_WoEdgeAug(**model_kwargs).to(device)
        elif variant == "StandardTransformer": model = MILAN_StandardTransformer(**model_kwargs).to(device)
        else: continue

        model.load_state_dict(torch.load(model_path, map_location=device))
        
        val_true, val_prob = get_eval_predictions(model, val_loader, device)
        test_true, test_prob = get_eval_predictions(model, test_loader, device)
        
        # 1. 先获取该数据集的真实正常类索引
        normal_indices = get_normal_indices(class_names)
        
        # 2. 把 normal_indices 传给这两个函数
        test_pred, best_th, val_macro, val_far = find_best_macro_f1_threshold_and_predict(val_true, val_prob, test_prob, normal_indices)
        metrics, cm = compute_all_metrics(test_true, test_pred, test_prob, class_names, normal_indices)
        
        print(f"  ✅ Done! Test Macro F1: {metrics['F1 (Macro)']:.4f}, ASA: {metrics['ASA']:.4f}, FAR: {metrics['FAR']:.4f}")

        # ！！核心修改在这里！！
        # 1. 直接覆盖/重写对应模型目录下的 metrics.txt
        with open(os.path.join(model_dir, "metrics.txt"), "w") as f:
            f.write(f"=== {exp_name} (Thresh: {best_th:.2f}) ===\n")
            for k, v in metrics.items(): 
                f.write(f"{k}: {v:.4f}\n")
                
        # 2. 重新绘制混淆矩阵并存放在模型目录下 (带有新的阈值名字)
        plot_and_save_confusion_matrix(cm, class_names, os.path.join(model_dir, f"cm_thresh_{best_th:.2f}.png"))
                
        # 3. 追加到全局大表
        with open(csv_file, "a") as f:
            f.write(f"{dataset_name},{variant},{group_str},{hidden},{seq_len},{best_th:.4f},"
                    f"{metrics['ACC']:.4f},{metrics['APR']:.4f},{metrics['RE']:.4f},"
                    f"{metrics['F1 (Macro)']:.4f},{metrics['F1 (Weighted)']:.4f},{metrics['AUC']:.4f},"
                    f"{metrics['ASA']:.4f},{metrics['FAR']:.4f}\n")

if __name__ == "__main__":
    main()