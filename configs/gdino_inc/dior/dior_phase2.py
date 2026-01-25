# configs/gdino_inc/dior/dior_15+5_phase2.py

_base_ = '../_base_/gdino_inc_distn_coco.py' 

load_from = './work_dirs/dior_phase1/epoch_25.pth'
#resume=True
dataset_type = 'DiorIncDataset'
data_root = '/data/zyt/'  
start = 10
end = 20 
distn_cfg=dict(
    ori_config_file='configs/gdino_inc/dior/dior_phase1.py',
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

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='DIOR_in/10/train_task_2.json',
        setting='full_text',
        start=start,
        end=end))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='DIOR_in/10/test_task_12.json',
        setting='full_text',
        start=start,
        end=end))

val_evaluator = dict(
    type='IncCocoMetric', 
    ann_file=data_root + 'DIOR_in/10/test_task_12.json')

test_dataloader = val_dataloader
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.00005, weight_decay=0.0001))
max_epochs = 25

custom_hooks = [
    dict(type='Increment_distn_hook')
]

