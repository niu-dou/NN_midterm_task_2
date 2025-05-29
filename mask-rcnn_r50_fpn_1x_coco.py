_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]



model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=20),
        mask_head=dict(num_classes=20)))


data_root = '/root/autodl-tmp/coco_format/'
img_prefix_2007 = '/root/autodl-tmp/VOCdevkit/VOC2007/JPEGImages/'  
img_prefix_2012 = '/root/autodl-tmp/VOCdevkit/VOC2012/JPEGImages/'  

metainfo = {
    'classes': ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
    
    'palette': [(106, 0, 228), (119, 11, 32), (165, 42, 42), (0, 0, 192), (197, 226, 255), (0, 60, 100), (0, 0, 142), (255, 77, 255), (153, 69, 1), (120, 166, 157), (0, 182, 199), (0, 226, 252), (182, 182, 255), (0, 0, 230), (220, 20, 60), (163, 255, 0), (0, 82, 0), (3, 95, 161), (0, 80, 100), (183, 130, 88)]
}
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=8, val_interval=1)
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='instances_train2012.json',
        data_prefix=dict(img=img_prefix_2012))
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='instances_test2012.json',
        data_prefix=dict(img=img_prefix_2012),
        test_mode=True),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='instances_test2007.json',
        data_prefix=dict(img=img_prefix_2007),
        test_mode=True),
)


val_evaluator = dict(
    ann_file=data_root + 'instances_test2012.json',
    metric=['bbox', 'segm']
)
test_evaluator = dict(
    ann_file=data_root + 'instances_test2007.json',
    metric=['bbox', 'segm']
)

log_config = dict(
    interval=50, 
    hooks=[
        dict(type='TextLoggerHook'),  
        dict(type='TensorboardLoggerHook',  # TensorBoard
             log_dir='work_dirs/tensorboard',  # 保存路径
             interval=50,  
             reset_flag=False,
             by_epoch=False)  
    ])

# 使用预训练的 Mask R-CNN 模型权重来做初始化
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'