
# 基于VOC数据集的Sparse R-CNN与Mask R-CNN训练
本项目使用MMDetection框架，在转换为COCO格式的VOC2012数据集上训练Sparse R-CNN和Mask R-CNN实例分割模型。

## 环境准备
1. **数据集下载**：
   ```bash
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   tar -xvf *.tar
   ```

2. **MMDetection安装**：
   ```bash
   pip install mmcv-full
   pip install mmdet
   # 或从源码安装：
   git clone https://github.com/open-mmlab/mmdetection.git
   cd mmdetection
   pip install -v -e .
   ```

3. **脚本部署**：
   - 将提供的配置文件放入对应目录：
     ```
     configs/
     ├── mask_rcnn/
     │   └── mask_rcnn_r50_fpn_1x_coco.py
     ├── sparse_rcnn/
     │   └── sparse_rcnn_r50_fpn_1x_coco.py
     └── _base_/models/
         └── sparse_rcnn_r50_fpn.py
     ```

## 核心脚本说明
| 脚本文件                          | 功能描述                                                                 | 存放位置                     |
|-----------------------------------|--------------------------------------------------------------------------|------------------------------|
| `convert.py`                      | 将PASCAL VOC格式数据集转换为COCO格式                                    | 项目根目录                   |
| `mask_rcnn_r50_fpn_1x_coco.py`    | Mask R-CNN训练配置（Backbone/学习率/数据增强等）                         | `configs/mask_rcnn/`         |
| `sparse_rcnn_r50_fpn_1x_coco.py`  | Sparse R-CNN训练配置                                                    | `configs/sparse_rcnn/`       |
| `sparse_rcnn_r50_fpn.py`          | Sparse R-CNN模型架构定义                                                | `configs/_base_/models/`     |
| `visualize.py`                    | 可视化脚本（支持实例分割/目标检测结果可视化）                           | 项目根目录                   |

## 操作指南

### 1. 数据集转换
修改`convert.py`中的路径变量后执行：
```bash
# 需修改的变量示例：
# VOC_ROOT = '/path/to/VOCdevkit'
# OUT_DIR = '/path/to/output/coco_style'
python convert.py
```
生成文件结构：
```
coco_style/
├── annotations
│   ├── instances_train2017.json
│   └── instances_val2017.json
└── images
    ├── train2017
    └── val2017
```

### 2. 模型训练
```bash
# 训练Sparse R-CNN (修改--work-dir指定输出目录)
python tools/train.py configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py \
    --work-dir /path/to/work_dir \
    --cfg-options data.samples_per_gpu=2 data.workers_per_gpu=2

# 训练Mask R-CNN
python tools/train.py configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py \
    --work-dir /path/to/work_dir
```
> **注意**：首次运行会自动下载ImageNet预训练权重

### 3. 结果可视化
修改`visualize.py`中以下参数后执行：
```python
config_file = 'configs/sparse_rcnn/sparse_rcnn_r50_fpn_1x_coco.py'
checkpoint_file = '/path/to/latest.pth'
img_path = 'test.jpg'  # 待可视化图像路径
python visualize.py
```

### 4. 训练日志分析
```bash
# 生成关键指标曲线图 (修改.json路径)
python tools/analysis_tools/analyze_logs.py plot_curve \
    /path/to/scalars.json \
    --keys loss bbox_mAP segm_mAP \
    --out metrics_curve.png
```

## 性能建议
1. **显存优化**：在`sparse_rcnn_r50_fpn.py`中调整：
   ```python
   model = dict(
       rpn_head=dict(num_proposals=100),  # 减少proposal数量
       test_cfg=dict(nms=dict(iou_threshold=0.6))  # 调整NMS阈值
   )
   ```
2. **数据增强**：在配置文件中添加增强策略提升鲁棒性：
   ```python
   train_pipeline = [
       dict(type='RandomFlip', flip_ratio=0.5),
       dict(type='RandomBrightnessContrast', brightness_limit=0.2, contrast_limit=0.2),
   ]
   ```

## 许可信息
本项目采用 [MIT 许可证](LICENSE)。模型权重和配置文件的使用请遵循其原始许可协议：
- Mask R-CNN: [Apache 2.0](https://github.com/open-mmlab/mmdetection/blob/master/LICENSE)
- Sparse R-CNN: [原始实现许可](https://github.com/PeizeSun/SparseR-CNN/blob/main/LICENSE)

