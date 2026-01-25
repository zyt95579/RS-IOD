# configs/gdino_inc/dior/dior_15+5_phase2.py

_base_ = '../_base_/gdino_inc_distn_coco.py' # 继承基础配置

# 加载第一阶段(0-14类)训练好的权重
load_from = './work_dirs/dota_phase1/epoch_30.pth'

dataset_type = 'DotaIncDataset'
data_root = '/data/zyt/DOTA/'  # 你的 DIOR 数据路径

# 定义增量设置
start = 5
end = 10 # DIOR 总共 20 类

distn_cfg=dict(
    # 指向第一阶段的配置文件，用于获取旧模型配置
    ori_config_file='configs/gdino_inc/dota/dota_phase1.py',
    future_class=False,
    label_distn=dict(
        type='adaptive_pseudo',
        mode='hardlabel',
        loss_ld=dict(type='KnowledgeDistillationKLDivLoss', loss_weight=1.0, T=100),
        sigma=0.4, nms_th=0.5, label_iou_th=0.7,
    ),
    feat_distn=dict(
        type='inter-class',
        subtype = 'opt1',
        img_loss=dict(type='L2Loss', loss_weight=3.0, reduction='mean'),
        text_loss=dict(type='L2Loss', loss_weight=5.0, reduction='mean'),
    ),
    query_distn=dict(
        type='seperate_queryinit',
        num_matching_query=900,
        num_aux_query=900,
    ),
)

model = dict(
    type='GroundingDINO_inc_gcd',
    num_queries=900,
    bbox_head=dict(
        type='GroundingDINOHead_inc_gcd',
        contrastive_cfg=dict(max_text_len=256, log_scale=None, bias=None),
        distn_cfg=distn_cfg,
        trunc_class=[start, end]), # [15, 20]
    dn_cfg=None,
    distn_cfg=distn_cfg
)

# 训练数据加载器
train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # 这里的 json 文件只应该包含 15-19 类别的标注
        ann_file='annotation/train_task_2.json',
        data_prefix=dict(img='image/train/'),
        setting='full_text',
        start=start,
        end=end))

# 验证/测试数据加载器
val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # 验证集通常包含所有类别，用于评估防遗忘能力
        ann_file='annotation/test_task_12.json',
        data_prefix=dict(img='image/val/'),
        setting='full_text',
        start=start,
        end=end))

# 评估器
val_evaluator = dict(
    type='IncCocoMetric', # 可以复用这个 Metric，只要你的 DIOR 数据是 COCO 格式
    ann_file=data_root + 'annotation/test_task_12.json')

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# 优化器和训练策略
optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.00005, weight_decay=0.0001))
max_epochs = 25

custom_hooks = [
    dict(type='Increment_distn_hook')
]