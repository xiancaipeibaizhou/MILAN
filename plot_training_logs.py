import os
import re
import glob
import matplotlib.pyplot as plt

def plot_training_history(log_file_path, save_path, title_name="Training History"):
    """
    读取 training_history.log 并绘制 Loss 与 Val F1 随时间变化的曲线图
    """
    epochs = []
    losses = []
    cl_losses = []
    val_f1s = []

    # 1. 解析日志文件
    with open(log_file_path, 'r') as f:
        for line in f:
            # 正则匹配日志行，例如: "Epoch 001 | Loss: 0.3061 | CL: 0.4765 | Val F1: 0.5051"
            match = re.search(r"Epoch\s+(\d+)\s*\|\s*Loss:\s*([\d.]+)\s*\|\s*CL:\s*([\d.]+)\s*\|\s*Val F1:\s*([\d.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                cl_losses.append(float(match.group(3)))
                val_f1s.append(float(match.group(4)))

    if not epochs:
        print(f"  ⚠️ No valid log data found in {log_file_path}")
        return

    # 2. 准备绘图 (使用双 Y 轴，因为 Loss 和 F1 的刻度范围不同)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # --- 左 Y 轴：Loss 曲线 ---
    color1 = 'tab:red'
    color2 = 'tab:orange'
    ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', color=color1, fontsize=12, fontweight='bold')
    
    # 画 Main Loss 和 CL Loss
    l1 = ax1.plot(epochs, losses, color=color1, label='Main Loss', linewidth=2, marker='o', markersize=4)
    l2 = ax1.plot(epochs, cl_losses, color=color2, label='Contrastive Loss', linewidth=2, linestyle='--', marker='x', markersize=4)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # --- 右 Y 轴：Val F1 曲线 ---
    ax2 = ax1.twinx()  
    color3 = 'tab:blue'
    ax2.set_ylabel('Validation Macro F1', color=color3, fontsize=12, fontweight='bold')
    
    # 画 F1
    l3 = ax2.plot(epochs, val_f1s, color=color3, label='Val F1', linewidth=2, marker='s', markersize=4)
    ax2.tick_params(axis='y', labelcolor=color3)
    
    # 动态设置右侧 Y 轴范围，让 F1 曲线看起来更饱满
    min_f1, max_f1 = min(val_f1s), max(val_f1s)
    ax2.set_ylim([max(0, min_f1 - 0.1), min(1.0, max_f1 + 0.1)])

    # --- 合并图例 ---
    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    # 将图例放在图表右下角，避免遮挡早期的 Loss 曲线
    ax1.legend(lines, labels, loc='lower right', fontsize=10, framealpha=0.9)

    plt.title(f'{title_name}\nLearning Curve', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    
    # 3. 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved learning curve to: {save_path}")

def main():
    # 在 results 文件夹下递归搜索所有的 training_history.log
    search_pattern = os.path.join("results", "**", "training_history.log")
    log_files = glob.glob(search_pattern, recursive=True)

    if not log_files:
        print("❌ No training_history.log files found in 'results/' directory.")
        return

    print(f"🎯 Found {len(log_files)} training logs. Generating plots...\n")

    for idx, log_file in enumerate(log_files, 1):
        print(f"[{idx}/{len(log_files)}] Processing: {log_file}")
        
        # 解析文件夹名字作为图表标题
        model_dir = os.path.dirname(log_file)
        parts = model_dir.split(os.sep)
        
        if len(parts) >= 3:
            dataset_name = parts[-3]
            exp_name = parts[-2]
            title = f"{dataset_name} | {exp_name}"
        else:
            title = "Training History"

        # 生成图片保存路径
        save_path = os.path.join(model_dir, "learning_curve.png")
        
        # 调用绘图函数
        plot_training_history(log_file, save_path, title_name=title)

if __name__ == "__main__":
    main()