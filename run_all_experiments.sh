cd MILAN
# 1. 在 UNSW-NB15 数据集上测试 MILAN 最终版
python run_milan_sota.py --dataset unsw_nb15 --variant MILAN 

# 2. 在 CIC-IDS2017 数据集上测试
python run_milan_sota.py --dataset cic_ids2017 --variant MILAN 

# 3. 在 ISCX-IDS2012 数据集上测试
python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN

# 4. 在 Darknet2020 加密流量数据集上测试
python run_milan_sota.py --dataset darknet2020_block --variant MILAN

# 基准测试 (Baseline): 仅保留最基础的 Transformer 架构
python run_milan_sota.py --dataset unsw_nb15 --variant StandardTransformer

# 变体 A: 去除局部特征提取 (无 Temporal Inception)
python run_milan_sota.py --dataset unsw_nb15 --variant WoLocal

# 变体 B: 去除全局特征提取 (无 Linear Attention)
python run_milan_sota.py --dataset unsw_nb15 --variant WoGlobal

# 变体 C: 去除自适应门控融合 (退化为简单的直接相加或拼接)
python run_milan_sota.py --dataset unsw_nb15 --variant WoGating

# 变体 D: 去除边特征与图拓扑增强 (退化为仅利用节点特征)
python run_milan_sota.py --dataset unsw_nb15 --variant WoEdgeAug

# 完整形态: MILAN
python run_milan_sota.py --dataset unsw_nb15 --variant MILAN

HP_GROUPS=EXP_SEQ_5  python run_milan_sota.py --dataset unsw_nb15 --variant MILAN
HP_GROUPS=EXP_SEQ_10 python run_milan_sota.py --dataset unsw_nb15 --variant MILAN
HP_GROUPS=EXP_SEQ_20 python run_milan_sota.py --dataset unsw_nb15 --variant MILAN
HP_GROUPS=EXP_SEQ_30 python run_milan_sota.py --dataset unsw_nb15 --variant MILAN

HP_GROUPS=EXP_KER_SINGLE python run_milan_sota.py --dataset unsw_nb15 --variant MILAN
HP_GROUPS=EXP_KER_SHORT  python run_milan_sota.py --dataset unsw_nb15 --variant MILAN
HP_GROUPS=EXP_KER_BASE   python run_milan_sota.py --dataset unsw_nb15 --variant MILAN
HP_GROUPS=EXP_KER_LONG   python run_milan_sota.py --dataset unsw_nb15 --variant MILAN

HP_GROUPS=EXP_DROP_0 python run_milan_sota.py --dataset unsw_nb15 --variant MILAN
HP_GROUPS=EXP_DROP_1 python run_milan_sota.py --dataset unsw_nb15 --variant MILAN
HP_GROUPS=EXP_DROP_2 python run_milan_sota.py --dataset unsw_nb15 --variant MILAN
HP_GROUPS=EXP_DROP_4 python run_milan_sota.py --dataset unsw_nb15 --variant MILAN

HP_GROUPS=EXP_CL_0   python run_milan_sota.py --dataset unsw_nb15 --variant MILAN
HP_GROUPS=EXP_CL_005 python run_milan_sota.py --dataset unsw_nb15 --variant MILAN
HP_GROUPS=EXP_CL_01  python run_milan_sota.py --dataset unsw_nb15 --variant MILAN
HP_GROUPS=EXP_CL_03  python run_milan_sota.py --dataset unsw_nb15 --variant MILAN

# 基准测试 (Baseline): 仅保留最基础的 Transformer 架构
python run_milan_sota.py --dataset darknet2020_block --variant StandardTransformer

# 变体 A: 去除局部特征提取 (无 Temporal Inception)
python run_milan_sota.py --dataset darknet2020_block --variant WoLocal

# 变体 B: 去除全局特征提取 (无 Linear Attention)
python run_milan_sota.py --dataset darknet2020_block --variant WoGlobal

# 变体 C: 去除自适应门控融合 (退化为简单的直接相加或拼接)
python run_milan_sota.py --dataset darknet2020_block --variant WoGating

# 变体 D: 去除边特征与图拓扑增强 (退化为仅利用节点特征)
python run_milan_sota.py --dataset darknet2020_block --variant WoEdgeAug

# 完整形态: MILAN
python run_milan_sota.py --dataset darknet2020_block --variant MILAN

HP_GROUPS=EXP_SEQ_5  python run_milan_sota.py --dataset darknet2020_block --variant MILAN
HP_GROUPS=EXP_SEQ_10 python run_milan_sota.py --dataset darknet2020_block --variant MILAN
HP_GROUPS=EXP_SEQ_20 python run_milan_sota.py --dataset darknet2020_block --variant MILAN
HP_GROUPS=EXP_SEQ_30 python run_milan_sota.py --dataset darknet2020_block --variant MILAN

HP_GROUPS=EXP_KER_SINGLE python run_milan_sota.py --dataset darknet2020_block --variant MILAN
HP_GROUPS=EXP_KER_SHORT  python run_milan_sota.py --dataset darknet2020_block --variant MILAN
HP_GROUPS=EXP_KER_BASE   python run_milan_sota.py --dataset darknet2020_block --variant MILAN
HP_GROUPS=EXP_KER_LONG   python run_milan_sota.py --dataset darknet2020_block --variant MILAN

HP_GROUPS=EXP_DROP_0 python run_milan_sota.py --dataset darknet2020_block --variant MILAN
HP_GROUPS=EXP_DROP_1 python run_milan_sota.py --dataset darknet2020_block --variant MILAN
HP_GROUPS=EXP_DROP_2 python run_milan_sota.py --dataset darknet2020_block --variant MILAN
HP_GROUPS=EXP_DROP_4 python run_milan_sota.py --dataset darknet2020_block --variant MILAN

HP_GROUPS=EXP_CL_0   python run_milan_sota.py --dataset darknet2020_block --variant MILAN
HP_GROUPS=EXP_CL_005 python run_milan_sota.py --dataset darknet2020_block --variant MILAN
HP_GROUPS=EXP_CL_01  python run_milan_sota.py --dataset darknet2020_block --variant MILAN
HP_GROUPS=EXP_CL_03  python run_milan_sota.py --dataset darknet2020_block --variant MILAN

# 基准测试 (Baseline): 仅保留最基础的 Transformer 架构
python run_milan_sota.py --dataset iscx_ids2012 --variant StandardTransformer

# 变体 A: 去除局部特征提取 (无 Temporal Inception)
python run_milan_sota.py --dataset iscx_ids2012 --variant WoLocal

# 变体 B: 去除全局特征提取 (无 Linear Attention)
python run_milan_sota.py --dataset iscx_ids2012 --variant WoGlobal

# 变体 C: 去除自适应门控融合 (退化为简单的直接相加或拼接)
python run_milan_sota.py --dataset iscx_ids2012 --variant WoGating

# 变体 D: 去除边特征与图拓扑增强 (退化为仅利用节点特征)
python run_milan_sota.py --dataset iscx_ids2012 --variant WoEdgeAug

# 完整形态: MILAN
python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN

HP_GROUPS=EXP_SEQ_5  python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN
HP_GROUPS=EXP_SEQ_10 python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN
HP_GROUPS=EXP_SEQ_20 python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN
HP_GROUPS=EXP_SEQ_30 python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN

HP_GROUPS=EXP_KER_SINGLE python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN
HP_GROUPS=EXP_KER_SHORT  python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN
HP_GROUPS=EXP_KER_BASE   python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN
HP_GROUPS=EXP_KER_LONG   python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN

HP_GROUPS=EXP_DROP_0 python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN
HP_GROUPS=EXP_DROP_1 python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN
HP_GROUPS=EXP_DROP_2 python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN
HP_GROUPS=EXP_DROP_4 python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN

HP_GROUPS=EXP_CL_0   python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN
HP_GROUPS=EXP_CL_005 python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN
HP_GROUPS=EXP_CL_01  python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN
HP_GROUPS=EXP_CL_03  python run_milan_sota.py --dataset iscx_ids2012 --variant MILAN

# 基准测试 (Baseline): 仅保留最基础的 Transformer 架构
python run_milan_sota.py --dataset cic_ids2017 --variant StandardTransformer

# 变体 A: 去除局部特征提取 (无 Temporal Inception)
python run_milan_sota.py --dataset cic_ids2017 --variant WoLocal

# 变体 B: 去除全局特征提取 (无 Linear Attention)
python run_milan_sota.py --dataset cic_ids2017 --variant WoGlobal

# 变体 C: 去除自适应门控融合 (退化为简单的直接相加或拼接)
python run_milan_sota.py --dataset cic_ids2017 --variant WoGating

# 变体 D: 去除边特征与图拓扑增强 (退化为仅利用节点特征)
python run_milan_sota.py --dataset cic_ids2017 --variant WoEdgeAug

# 完整形态: MILAN
python run_milan_sota.py --dataset cic_ids2017 --variant MILAN

HP_GROUPS=EXP_SEQ_5  python run_milan_sota.py --dataset cic_ids2017 --variant MILAN
HP_GROUPS=EXP_SEQ_10 python run_milan_sota.py --dataset cic_ids2017 --variant MILAN
HP_GROUPS=EXP_SEQ_20 python run_milan_sota.py --dataset cic_ids2017 --variant MILAN
HP_GROUPS=EXP_SEQ_30 python run_milan_sota.py --dataset cic_ids2017 --variant MILAN

HP_GROUPS=EXP_KER_SINGLE python run_milan_sota.py --dataset cic_ids2017 --variant MILAN
HP_GROUPS=EXP_KER_SHORT  python run_milan_sota.py --dataset cic_ids2017 --variant MILAN
HP_GROUPS=EXP_KER_BASE   python run_milan_sota.py --dataset cic_ids2017 --variant MILAN
HP_GROUPS=EXP_KER_LONG   python run_milan_sota.py --dataset cic_ids2017 --variant MILAN

HP_GROUPS=EXP_DROP_0 python run_milan_sota.py --dataset cic_ids2017 --variant MILAN
HP_GROUPS=EXP_DROP_1 python run_milan_sota.py --dataset cic_ids2017 --variant MILAN
HP_GROUPS=EXP_DROP_2 python run_milan_sota.py --dataset cic_ids2017 --variant MILAN
HP_GROUPS=EXP_DROP_4 python run_milan_sota.py --dataset cic_ids2017 --variant MILAN

HP_GROUPS=EXP_CL_0   python run_milan_sota.py --dataset cic_ids2017 --variant MILAN
HP_GROUPS=EXP_CL_005 python run_milan_sota.py --dataset cic_ids2017 --variant MILAN
HP_GROUPS=EXP_CL_01  python run_milan_sota.py --dataset cic_ids2017 --variant MILAN
HP_GROUPS=EXP_CL_03  python run_milan_sota.py --dataset cic_ids2017 --variant MILAN