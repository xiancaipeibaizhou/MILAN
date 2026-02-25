# MILAN: Entropy-Aware Adaptive Graph Learning with Dual-Stream Temporal Dynamics for High-Speed Intrusion Detection

<p align="left">
  <img src="https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=flat-square&logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/PyG-2.4.0-3C2179?style=flat-square&logo=python" alt="PyG">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License">
</p>

**MILAN** is an advanced Spatio-Temporal Graph Neural Network (ST-GNN) designed for high-speed Network Intrusion Detection Systems (NIDS). It is specifically architected to tackle critical real-world challenges, including **Extreme Class Imbalance**, **Temporal Concept Drift**, and **Encrypted Traffic Obfuscation**.

## ✨ Key Highlights

1. **Dual-Stream Temporal Dynamics**:
   * **Local Stream (Stream A)**: Utilizes multi-scale Inception convolutions ($\mathcal{K}=\{1, 3, 5, 7\}$) to precisely capture short-term micro-burst attacks (e.g., DDoS, Brute Force).
   * **Global Stream (Stream B)**: Employs Linear Global Attention to mine the long-range statistical dependencies of encrypted C2 tunnels (e.g., Darknet, VPN).
2. **Entropy-Aware Adaptive Gating**: Breaks the "static pipeline fallacy" by dynamically balancing local features and global context based on the current traffic graph's topological entropy, effectively preventing feature interference.
3. **Jitter-Based Contrastive Learning ($\mathcal{L}_{CL}$)**: Introduces contrastive learning as a regularization anchor in the latent space. It explicitly forces the representations of minority attacks away from benign background clusters, significantly enhancing robustness against temporal concept drift.
4. **Dynamic Operational Thresholding**: Abandons the rigid `0.5` softmax threshold. The framework autonomously calibrates the optimal boundary via validation Macro-F1 across multi-normal classes. This achieves a massive surge in Anomaly-Set Accuracy (ASA) while maintaining a strict False Alarm Rate (FAR $\le 1\%$).

---

## 🗄️ Data Preprocessing

To maintain the purity and cleanliness of the core algorithmic framework, the preprocessing pipeline (converting raw PCAP/CSV traffic logs into Spatio-Temporal PyG Graphs) is maintained independently. 

If you need to generate the datasets (`train_graphs.pt`, `val_graphs.pt`, `test_graphs.pt`) from scratch, please visit our dedicated data processing repository:

👉 **[Generate Data Repository](https://github.com/xiancaipeibaizhou/generate_data.git)**

---

## 📂 Repository Structure

```text
├── model.py                 # Core MILAN architecture (Dual-stream, Gating, Edge-Augmented Attention)
├── ablation_models.py       # Model variants for ablation studies (WoGlobal, WoLocal, WoGating, etc.)
├── hparams_a3.py            # Centralized hyperparameter configuration dictionary for diverse datasets
├── run_milan_sota.py        # Core training/validation script (CL loss, dynamic weighting, early stopping)
├── batch_re_evaluate.py     # Automated batch evaluation (Threshold searching, CM plotting, metric exporting)
├── plot_training_logs.py    # Visualization tool (Plots dual-axis learning curves for training dynamics)
├── run_all_experiments.sh   # Automated pipeline script to execute all baselines and ablation groups sequentially
└── README.md                # Project documentation
