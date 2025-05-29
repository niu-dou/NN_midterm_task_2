
import os
import mmcv
import cv2
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS

# 配置参数
# config_mask = '/root/mask-rcnn_r50_fpn_1x_coco.py'
# checkpoint_mask = '/root/epoch_8.pth'
config_sparse = '/root/sparse-rcnn_r50_fpn_1x_voc2012_new.py'
checkpoint_sparse = '/root/epoch_1.pth'
testset_dir = '/root/autodl-tmp/VOCdevkit/VOC2007/JPEGImages'  # VOC2007测试集路径
output_dir = '/root/project/output/dunjiang'
os.makedirs(output_dir, exist_ok=True)

# 初始化模型
# model_mask = init_detector(config_mask, checkpoint_mask, device='cuda:0')
model_sparse = init_detector(config_sparse, checkpoint_sparse, device='cuda:0')


# 初始化可视化工具（关键修正）
# visualizer_mask = VISUALIZERS.build(model_mask.cfg.visualizer)
# visualizer_mask.dataset_meta = model_mask.dataset_meta

visualizer_sparse = VISUALIZERS.build(model_sparse.cfg.visualizer)
visualizer_sparse.dataset_meta = model_sparse.dataset_meta

# def draw_proposals(img, proposals, color=(0, 255, 0)):
#     """绘制第一阶段proposal boxes"""
#     for box in proposals[:50]:  # 取前50个proposals
#         x1, y1, x2, y2 = map(int, box[:4])
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
#     return img

# valid_exts = ('.jpg', '.png', '.jpeg')
# # 处理4张测试图像
# for img_name in sorted(os.listdir(testset_dir))[:4]:
#     img_path = os.path.join(testset_dir, img_name)
#     if img_name.startswith('.') or not os.path.isfile(img_path):
#         continue
#     # 仅处理图像文件
#     if img_path.lower().endswith(valid_exts):
#         img = mmcv.imread(img_path)
        
#         # # Mask R-CNN Proposals
#         # result_mask = inference_detector(model_mask, img)
#         # proposals = result_mask.pred_instances.bboxes.cpu().numpy()
#         # img_proposals = draw_proposals(img.copy(), proposals)
#         # mmcv.imwrite(img_proposals, os.path.join(output_dir, f'mask_proposals_{img_name}'))
        
#         # # Mask R-CNN最终结果
#         # visualizer_mask.add_datasample('result', img, result_mask, draw_gt=False)
#         # mmcv.imwrite(visualizer_mask.get_image(), os.path.join(output_dir, f'mask_final_{img_name}'))
    
#         # Sparse R-CNN结果
#         result_sparse = inference_detector(model_sparse, img)
#         visualizer_sparse.add_datasample('result', img, result_sparse, draw_gt=False)
#         mmcv.imwrite(visualizer_sparse.get_image(), os.path.join(output_dir, f'sparse_final_{img_name}'))

# 三张外部图像路径（替换为你的实际路径）
external_images = [
    '/root/1.jpg',
    '/root/2.jpg',
    '/root/3.jpg',
]

def visualize_results(model, visualizer, img, img_name, model_name):
    """执行推理并可视化结果"""
    result = inference_detector(model, img)
    
    # 可视化最终结果（带mask、bbox、标签、得分）
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        show=False,
        pred_score_thr=0.5  # 只显示置信度>0.5的结果
    )
    vis_image = visualizer.get_image()  # 正确获取图像[1,6](@ref)
    
    # 保存结果
    output_path = os.path.join(output_dir, f'{model_name}_{os.path.basename(img_name)}')
    mmcv.imwrite(vis_image, output_path)
    return output_path

# 处理三张外部图像
for img_path in external_images:
    if not os.path.exists(img_path):
        print(f"图像不存在: {img_path}")
        continue
    
    img = mmcv.imread(img_path)
    img_name = os.path.basename(img_path)
    
    print(f"处理图像: {img_name}")
    
    # # Mask R-CNN结果
    # mask_result_path = visualize_results(model_mask, visualizer_mask, img, img_name, 'mask_rcnn')
    # print(f"Mask R-CNN结果保存至: {mask_result_path}")
    
    # Sparse R-CNN结果
    sparse_result_path = visualize_results(model_sparse, visualizer_sparse, img, img_name, 'sparse_rcnn')
    print(f"Sparse R-CNN结果保存至: {sparse_result_path}")
    
    print("-" * 50)

print("所有图像处理完成！结果保存在:", output_dir)