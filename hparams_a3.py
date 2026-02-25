import os

def _parse_int_list(text, fallback):
    if text is None:
        return list(fallback)
    s = str(text).strip()
    if not s:
        return list(fallback)
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    return [int(p) for p in parts]

def _canonical_dataset_name(name):
    s = str(name or "").strip().lower()
    if not s:
        return ""
    if s in {"nb15", "unsw-nb15", "unsw_nb15", "unswnb15"}: return "nb15"
    if s in {"darknet2020", "cic-darknet2020", "cic_darknet2020", "darknet"}: return "darknet2020"
    if s in {"ids2017", "cicids2017", "cic-ids2017", "cic_ids2017"}: return "ids2017"
    if s in {"iscx2012", "iscx-ids2012", "iscx_ids2012", "2012", "ids2012"}: return "iscx2012"
    return s

# ==========================================
# 1. 核心控制变量实验组 (用于超参数搜索与敏感性分析)
# ==========================================
GROUPS = {
    # --- 维度 1: 时序序列长度测试 (Sequence Length) ---
    "EXP_SEQ_5":  {"SEQ_LEN": 5,  "BATCH_SIZE": 64},
    "EXP_SEQ_10": {"SEQ_LEN": 10, "BATCH_SIZE": 32}, # 通常的 Base
    "EXP_SEQ_20": {"SEQ_LEN": 20, "BATCH_SIZE": 16},
    "EXP_SEQ_30": {"SEQ_LEN": 30, "BATCH_SIZE": 8},

    # --- 维度 2: 局部流多尺度感受野 (Inception Kernels) ---
    "EXP_KER_SHORT": {"KERNELS": [1, 3]},
    "EXP_KER_BASE":  {"KERNELS": [1, 3, 5, 7]},
    "EXP_KER_LONG":  {"KERNELS": [1, 3, 5, 7, 9, 11]},
    "EXP_KER_SINGLE":{"KERNELS": [3]},

    # --- 维度 3: 模型容量与表征维度 (Capacity) ---
    "EXP_CAP_TINY":  {"HIDDEN": 32,  "HEADS": 4},
    "EXP_CAP_SMALL": {"HIDDEN": 64,  "HEADS": 4},
    "EXP_CAP_BASE":  {"HIDDEN": 128, "HEADS": 8},
    "EXP_CAP_LARGE": {"HIDDEN": 256, "HEADS": 8},

    # --- 维度 4: 图结构鲁棒性 (DropEdge) ---
    "EXP_DROP_0":  {"DROPEDGE_P": 0.0}, # 不使用 DropEdge (退化为标准静态图)
    "EXP_DROP_1":  {"DROPEDGE_P": 0.1},
    "EXP_DROP_2":  {"DROPEDGE_P": 0.2},
    "EXP_DROP_4":  {"DROPEDGE_P": 0.4},

    # --- 维度 5: 对比学习强度 (Contrastive Loss Weight) ---
    "EXP_CL_0":   {"CL_LOSS_WEIGHT": 0.0},  # 相当于无对比学习 (NOCL)
    "EXP_CL_005": {"CL_LOSS_WEIGHT": 0.05},
    "EXP_CL_01":  {"CL_LOSS_WEIGHT": 0.1},
    "EXP_CL_03":  {"CL_LOSS_WEIGHT": 0.3},
}

# ==========================================
# 2. 数据集特定最优 Base 锚点
# ==========================================
DATASET_BEST = {
    # Darknet 流量高度加密，需要更强的结构扰动和对比学习来学习鲁棒表征
    "darknet2020": {"HIDDEN": 128, "HEADS": 8, "DROPEDGE_P": 0.3, "CL_LOSS_WEIGHT": 0.5, "LR": 0.0005, "MAX_CL_EDGES": 4096},
    
    # NB15 攻击种类多且隐蔽，需要均衡的超参和较高的学习率
    "nb15":        {"HIDDEN": 128, "HEADS": 8, "DROPEDGE_P": 0.2, "CL_LOSS_WEIGHT": 0.2, "LR": 0.002, "MAX_CL_EDGES": 8192},
    
    # IDS2017 数据量大，攻击特征相对明显，降低扰动概率加快收敛
    "ids2017":     {"HIDDEN": 128, "HEADS": 8, "DROPEDGE_P": 0.1, "CL_LOSS_WEIGHT": 0.1, "LR": 0.002, "MAX_CL_EDGES": 8192},
    
    # ISCX2012 数据集较老，特征简单，使用较小的 Hidden 防止过拟合
    "iscx2012":    {"HIDDEN": 64,  "HEADS": 4, "DROPEDGE_P": 0.1, "CL_LOSS_WEIGHT": 0.1, "LR": 0.002, "MAX_CL_EDGES": 8192},
}

def resolve_hparams(group, env=None, dataset=None):
    if env is None:
        env = os.environ

    cl_env = env.get("CL_LOSS_WEIGHT", env.get("CL_WEIGHT", None))

    # 全局默认保底参数
    h = {
        "SEQ_LEN": int(env.get("SEQ_LEN", "10")),
        "BATCH_SIZE": int(env.get("BATCH_SIZE", "32")),
        "ACCUM_STEPS": int(env.get("ACCUM_STEPS", "1")),
        "NUM_EPOCHS": int(env.get("NUM_EPOCHS", "150")),
        "LR": float(env.get("LR", "0.002")),
        "HIDDEN": int(env.get("HIDDEN", "128")),
        "HEADS": int(env.get("HEADS", "8")),
        "KERNELS": _parse_int_list(env.get("KERNELS", ""), fallback=[1, 3, 5, 7]),
        "MAX_CL_EDGES": int(env.get("MAX_CL_EDGES", "8192")),
        "PATIENCE": int(env.get("PATIENCE", "10")),
        "MIN_DELTA": float(env.get("MIN_DELTA", "0.0")),
        "CL_LOSS_WEIGHT": float(cl_env if cl_env is not None else "0.1"),
        "DROP_PATH": float(env.get("DROP_PATH", "0.1")),
        "DROPEDGE_P": float(env.get("DROPEDGE_P", "0.2")),
        "WARMUP_EPOCHS": int(env.get("WARMUP_EPOCHS", "5")),
        "COSINE_T0": int(env.get("COSINE_T0", "10")),
        "COSINE_TMULT": int(env.get("COSINE_TMULT", "1")),
    }

    # 1. 注入 Dataset 最优 Base 设定
    dataset_name = _canonical_dataset_name(dataset if dataset is not None else env.get("DATASET", None))
    best = DATASET_BEST.get(dataset_name, None)
    if isinstance(best, dict):
        for k, v in best.items():
            if k == "CL_LOSS_WEIGHT" and cl_env is None:
                h[k] = v
            elif k not in env:
                h[k] = v

    # 2. 注入特定的实验组设定 (覆盖 Base)
    group_norm = (group or "").strip().upper()
    if group_norm not in {"", "BEST", "AUTO", "DEFAULT"} and group_norm in GROUPS:
        for k, v in GROUPS[group_norm].items():
            if k == "CL_LOSS_WEIGHT" and cl_env is None:
                h[k] = v
            elif k not in env:
                h[k] = v

    return h