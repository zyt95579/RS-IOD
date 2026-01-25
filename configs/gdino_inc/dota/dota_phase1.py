_base_ = '../_base_/gdino_inc_base_coco.py' # 可以复用基础配置，只要下面覆盖了数据集设置
pretrained = '/data/zyt/GCD-main/swin_tiny_patch4_window7_224.pth'

# 1. 修改数据集类型为我们新定义的 DiorIncDataset
dataset_type = 'DotaIncDataset'
data_root = '/data/zyt/DOTA/'

# 2. 修改类别范围：DIOR 第一阶段通常是 0-5 类
start = 0
end = 5
resume =True
model = dict(
    type='GroundingDINO_inc',
    num_queries=900,
    backbone=dict(
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    bbox_head=dict(trunc_class=[start, end]),   # 训练时限制只学习 0-14 类
    )

train_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # 3. 修改标注文件路径：指向包含 0-14 类标注的 COCO 格式 JSON
        ann_file='annotation/train_task_1.json',
        data_prefix=dict(img='image/train/'),
        start=start,
        end=end))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # 验证集可以使用包含所有类别或仅含 0-14 类的 JSON，这里主要验证模型对已学类别的性能
        ann_file='annotation/test_task_1.json',
        data_prefix=dict(img='image/val/'),
        start=start,
        end=end))

# 4. 修改评估器配置
val_evaluator = dict(
    type='IncCocoMetric', # 只要是 COCO 格式 JSON，可以用同一个 Metric
    ann_file=data_root + 'annotation/test_task_1.json')

test_dataloader = val_dataloader
test_evaluator = val_evaluator

optim_wrapper = dict(optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.0001))

# learning policy
max_epochs = 30