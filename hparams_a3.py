import os


def _parse_int_list(text, fallback):
    if text is None:
        return list(fallback)
    s = str(text).strip()
    if not s:
        return list(fallback)
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    out = []
    for p in parts:
        out.append(int(p))
    return out


GROUPS = {
    "A5": {"SEQ_LEN": 5, "HIDDEN": 64, "KERNELS": [1, 3, 5, 7], "CL_LOSS_WEIGHT": 0.1},
    "A10": {"SEQ_LEN": 10, "HIDDEN": 64, "KERNELS": [1, 3, 5, 7], "CL_LOSS_WEIGHT": 0.1},
    "A20": {"SEQ_LEN": 20, "HIDDEN": 64, "KERNELS": [1, 3, 5, 7], "CL_LOSS_WEIGHT": 0.1},
    "A30": {"SEQ_LEN": 30, "HIDDEN": 64, "KERNELS": [1, 3, 5, 7], "CL_LOSS_WEIGHT": 0.1},
    "A40": {"SEQ_LEN": 40, "HIDDEN": 64, "KERNELS": [1, 3, 5, 7], "CL_LOSS_WEIGHT": 0.1},
    "A50": {"SEQ_LEN": 50, "HIDDEN": 64, "KERNELS": [1, 3, 5, 7], "CL_LOSS_WEIGHT": 0.1},
    "B_MS": {"SEQ_LEN": 10, "HIDDEN": 64, "KERNELS": [1, 3, 5, 7]},
    "B_S3": {"SEQ_LEN": 10, "HIDDEN": 64, "KERNELS": [3]},
    "B_S5": {"SEQ_LEN": 10, "HIDDEN": 64, "KERNELS": [5]},
    "B_S7": {"SEQ_LEN": 10, "HIDDEN": 64, "KERNELS": [7]},
    "B_357": {"SEQ_LEN": 10, "HIDDEN": 64, "KERNELS": [3, 5, 7]},
    "B_3579": {"SEQ_LEN": 10, "HIDDEN": 64, "KERNELS": [3, 5, 7, 9]},
    "B_1357": {"SEQ_LEN": 10, "HIDDEN": 64, "KERNELS": [1, 3, 5, 7]},
    "B_13579": {"SEQ_LEN": 10, "HIDDEN": 64, "KERNELS": [1, 3, 5, 7, 9]},
    "B_35711": {"SEQ_LEN": 10, "HIDDEN": 64, "KERNELS": [3, 5, 7, 11]},
    "B_357911": {"SEQ_LEN": 10, "HIDDEN": 64, "KERNELS": [3, 5, 7, 9, 11]},
    "C0": {"CL_LOSS_WEIGHT": 0.0},
    "C001": {"CL_LOSS_WEIGHT": 0.01},
    "C005": {"CL_LOSS_WEIGHT": 0.05},
    "C01": {"CL_LOSS_WEIGHT": 0.1},
    "C02": {"CL_LOSS_WEIGHT": 0.2},
    "D1": {"HIDDEN": 32, "HEADS": 4},
    "D2": {"HIDDEN": 64, "HEADS": 4},    
    "D4": {"HIDDEN": 64, "HEADS": 8},
    "D3": {"HIDDEN": 128, "HEADS": 4},
    "D5": {"HIDDEN": 128, "HEADS": 8},
    "D6": {"BATCH_SIZE": 16},
    "D7": {"PATIENCE": 20,"BATCH_SIZE": 16,"HIDDEN": 32},
    "E1": {"MAX_CL_EDGES": "2048"},
    "E2": {"MAX_CL_EDGES": "4096"},
    "E3": {"MAX_CL_EDGES": "6144"},
    "E4": {"MAX_CL_EDGES": "8192"},
    "E5": {"MAX_CL_EDGES": "10000"},
    "E6": {"MAX_CL_EDGES": "10240"},
    "E7": {"MAX_CL_EDGES": "12288"},
    "E8": {"MAX_CL_EDGES": "16384"},
    "E9": {"MAX_CL_EDGES": "6000"},
    "E10": {"MAX_CL_EDGES": "7000"},
    "E11": {"MAX_CL_EDGES": "8000"},
    "E12": {"MAX_CL_EDGES": "9000"},
    "E13": {"MAX_CL_EDGES": "10000"},
    "E14": {"MAX_CL_EDGES": "7963"},
    "E15": {"MAX_CL_EDGES": "8448"},
    "E16": {"MAX_CL_EDGES": "8960"},
    "E17": {"MAX_CL_EDGES": "8192"},

    "F0": {"CL_LOSS_WEIGHT": 0.0},
    "F1": {"CL_VIEW1_DROPEDGE_P": 0.1, "CL_VIEW2_DROPEDGE_P": 0.2},
    "F2": {"CL_VIEW1_DROPEDGE_P": 0.2, "CL_VIEW2_DROPEDGE_P": 0.4},
    "F3": {"CL_VIEW1_DROPEDGE_P": 0.3, "CL_VIEW2_DROPEDGE_P": 0.5},
    

}


def resolve_hparams(group, env=None):
    if env is None:
        env = os.environ

    cl_env = env.get("CL_LOSS_WEIGHT", None)
    if cl_env is None:
        cl_env = env.get("CL_WEIGHT", None)

    h = {
        "SEQ_LEN": int(env.get("SEQ_LEN", "10")),
        "BATCH_SIZE": int(env.get("BATCH_SIZE", "32")),
        "ACCUM_STEPS": int(env.get("ACCUM_STEPS", "1")),
        "NUM_EPOCHS": int(env.get("NUM_EPOCHS", "150")),
        "LR": float(env.get("LR", "0.001")),
        "HIDDEN": int(env.get("HIDDEN", "128")),
        "HEADS": int(env.get("HEADS", "8")),
        "KERNELS": _parse_int_list(env.get("KERNELS", ""), fallback=[1, 3, 5, 7]),
        "MAX_CL_EDGES": int(env.get("MAX_CL_EDGES", "2048")),
        "PATIENCE": int(env.get("PATIENCE", "10")),
        "MIN_DELTA": float(env.get("MIN_DELTA", "0.0")),
        "EARLY_STOP_METRIC": str(env.get("EARLY_STOP_METRIC", "val_f1")).strip().lower(),
        "CL_LOSS_WEIGHT": float(cl_env if cl_env is not None else "0.1"),
        "DROP_PATH": float(env.get("DROP_PATH", "0.1")),
        "DROPEDGE_P": float(env.get("DROPEDGE_P", "0.2")),
        "CL_VIEW1_DROPEDGE_P": float(env.get("CL_VIEW1_DROPEDGE_P", "0.1")),
        "CL_VIEW2_DROPEDGE_P": float(env.get("CL_VIEW2_DROPEDGE_P", "0.2")),
        "WARMUP_EPOCHS": int(env.get("WARMUP_EPOCHS", "5")),
        "COSINE_T0": int(env.get("COSINE_T0", "10")),
        "COSINE_TMULT": int(env.get("COSINE_TMULT", "1")),
        "ETA_MIN_RATIO": float(env.get("ETA_MIN_RATIO", "0.01")),
        "TARGET_FAR": float(env.get("TARGET_FAR", "0.01")),
    }

    group = (group or "").strip().upper()
    if group in GROUPS:
        for k, v in GROUPS[group].items():
            if k == "CL_LOSS_WEIGHT":
                if ("CL_LOSS_WEIGHT" not in env) and ("CL_WEIGHT" not in env):
                    h[k] = v
                continue
            if k not in env:
                h[k] = v

    if h["EARLY_STOP_METRIC"] in {"f1", "valf1"}:
        h["EARLY_STOP_METRIC"] = "val_f1"
    elif h["EARLY_STOP_METRIC"] in {"asa", "valasa"}:
        h["EARLY_STOP_METRIC"] = "val_asa"
    else:
        h["EARLY_STOP_METRIC"] = "val_f1"

    return h
