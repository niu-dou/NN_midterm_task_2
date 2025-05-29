# 继承基础配置
_base_ = [
    '../_base_/models/sparse-rcnn_r50_fpn_new.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]


dataset_type = 'CocoDataset'
data_root = '/root/autodl-tmp/coco_format/'
img_prefix_2007 = '/root/autodl-tmp/VOCdevkit/VOC2007/JPEGImages/'  
img_prefix_2012 = '/root/autodl-tmp/VOCdevkit/VOC2012/JPEGImages/'  

metainfo = {
    'classes': ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
    # palette is a list of color tuples, which is used for visualization.
    'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192), (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255), (153, 69, 1), (120, 166, 157), (0, 182, 199), (0, 226, 252), (182, 182, 255), (0, 0, 230), (220, 20, 60), (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88)]
}

model = dict(
    roi_head=dict(
        bbox_head=[dict(type='DIIHead', num_classes=20) for _ in range(6)],
        mask_head=dict(
            type='DynamicMaskHead',
            num_classes=20,
            roi_feat_size=14,
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            loss_mask=dict(type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        )
    )
)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='PackDetInputs')
]


train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='instances_train2012.json',
        data_prefix=dict(img=img_prefix_2012),
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='instances_test2012.json', 
        data_prefix=dict(img=img_prefix_2012),
        test_mode=True,
        pipeline=test_pipeline),
)

test_dataloader =  dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file='instances_test2007.json', 
        data_prefix=dict(img=img_prefix_2012),
        test_mode=True,
        pipeline=test_pipeline),
)


val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'instances_test2012.json',
    metric=['bbox', 'segm']
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'instances_test2007.json',
    metric=['bbox', 'segm']
)


optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True, 
        type='AdamW', 
        lr=0.0001, 
        weight_decay=0.0001),
    clip_grad=dict(max_norm=1.0, norm_type=2))


param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]


log_config = dict(
    interval=50,  # 每50个迭代记录一次日志
    hooks=[
        dict(type='TextLoggerHook'),  # 保留文本日志
        dict(type='TensorboardLoggerHook',  # 启用TensorBoard
             log_dir='work_dirs/tensorboard',  # 日志保存路径
             interval=50,  # 与文本日志同步
             reset_flag=False,
             by_epoch=True)  # 按轮次记录
    ])

# 使用预训练的 Sparse R-CNN 模型权重来做初始化
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth'