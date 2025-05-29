#!/usr/bin/env python3
"""
将VOC数据集转换为COCO格式以支持实例分割
"""

import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from datetime import datetime
from pathlib import Path

# VOC类别名称
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class VOC2COCO:
    def __init__(self, voc_root, output_dir):
        self.voc_root = Path(voc_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # COCO格式数据结构
        self.coco_data = {
            'info': {
                'description': 'VOC Dataset converted to COCO format',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'VOC2COCO Converter',
                'date_created': datetime.now().isoformat()
            },
            'licenses': [{
                'id': 1,
                'name': 'Unknown',
                'url': ''
            }],
            'categories': [],
            'images': [],
            'annotations': []
        }
        
        # 创建类别信息
        for i, class_name in enumerate(VOC_CLASSES):
            self.coco_data['categories'].append({
                'id': i + 1,
                'name': class_name,
                'supercategory': 'object'
            })
        
        self.class_to_id = {class_name: i + 1 for i, class_name in enumerate(VOC_CLASSES)}
        self.image_id = 1
        self.annotation_id = 1
    
    def parse_xml_annotation(self, xml_file):
        """
        解析VOC XML标注文件
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 获取图片信息
        filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        # 获取所有目标
        objects = []
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in self.class_to_id:
                continue
            
            # 获取边界框
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # 计算COCO格式的bbox [x, y, width, height]
            coco_bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
            area = (xmax - xmin) * (ymax - ymin)
            
            objects.append({
                'class_name': class_name,
                'class_id': self.class_to_id[class_name],
                'bbox': coco_bbox,
                'area': area,
                'iscrowd': 0
            })
        
        return {
            'filename': filename,
            'width': width,
            'height': height,
            'objects': objects
        }
    
    def create_dummy_segmentation(self, bbox):
        """
        为目标检测创建虚拟的分割掩码（使用边界框）
        """
        x, y, w, h = bbox
        # 创建矩形分割掩码
        segmentation = [[
            x, y,
            x + w, y,
            x + w, y + h,
            x, y + h
        ]]
        return segmentation
    
    def process_split(self, split_name, image_sets_file, year):
        """
        处理数据集分割（train/val/test）
        """
        print(f"处理 {split_name} 分割...")
        voc_year_dir = self.voc_root / f'VOC{year}'
        # 读取图片列表
        with open(image_sets_file, 'r') as f:
            image_names = [line.strip() for line in f.readlines()]
        
        processed_count = 0
        for image_name in image_names:
            # 检查XML文件是否存在
            xml_file = voc_year_dir / 'Annotations' / f'{image_name}.xml'
            if not xml_file.exists():
                print(f"警告: XML文件不存在 {xml_file}")
                continue
            
            # 检查图片文件是否存在
            img_file = voc_year_dir / 'JPEGImages' / f'{image_name}.jpg'
            if not img_file.exists():
                print(f"警告: 图片文件不存在 {img_file}")
                continue
            
            # 解析标注
            try:
                annotation_data = self.parse_xml_annotation(xml_file)
            except Exception as e:
                print(f"错误: 解析XML文件失败 {xml_file}: {e}")
                continue
            
            # 添加图片信息
            image_info = {
                'id': self.image_id,
                'file_name': annotation_data['filename'],
                'width': annotation_data['width'],
                'height': annotation_data['height'],
                'license': 1
            }
            self.coco_data['images'].append(image_info)
            
            # 添加标注信息
            for obj in annotation_data['objects']:
                annotation = {
                    'id': self.annotation_id,
                    'image_id': self.image_id,
                    'category_id': obj['class_id'],
                    'bbox': obj['bbox'],
                    'area': obj['area'],
                    'iscrowd': obj['iscrowd'],
                    'segmentation': self.create_dummy_segmentation(obj['bbox'])
                }
                self.coco_data['annotations'].append(annotation)
                self.annotation_id += 1
            
            self.image_id += 1
            processed_count += 1
            
            if processed_count % 100 == 0:
                print(f"已处理 {processed_count} 张图片...")
        
        print(f"{split_name} 分割处理完成，共处理 {processed_count} 张图片")
    
    def convert_dataset(self, year='2007'):
        """
        转换整个数据集
        """
        voc_year_dir = self.voc_root / f'VOC{year}'
        
        if not voc_year_dir.exists():
            print(f"错误: VOC{year} 目录不存在 {voc_year_dir}")
            return
        
        # 处理训练集
        train_file = voc_year_dir / 'ImageSets' / 'Main' / 'train.txt'
        if train_file.exists():
            self.process_split('train', train_file, year)
        
        # 保存训练集COCO格式文件
        train_output = self.output_dir / f'instances_train{year}.json'
        with open(train_output, 'w') as f:
            json.dump(self.coco_data, f, indent=2)
        print(f"训练集COCO格式文件保存到: {train_output}")
        
        # 重置数据结构用于测试集
        test_coco_data = {
            'info': self.coco_data['info'].copy(),
            'licenses': self.coco_data['licenses'].copy(),
            'categories': self.coco_data['categories'].copy(),
            'images': [],
            'annotations': []
        }
        
        # 临时保存当前数据
        temp_images = self.coco_data['images'].copy()
        temp_annotations = self.coco_data['annotations'].copy()
        temp_image_id = self.image_id
        temp_annotation_id = self.annotation_id
        
        # 重置为测试集
        self.coco_data['images'] = []
        self.coco_data['annotations'] = []
        self.image_id = 1
        self.annotation_id = 1
        
        # 处理测试集
        if year == '2007':
            test_file = voc_year_dir / 'ImageSets' / 'Main' / 'test.txt'
        elif year == '2012':
            test_file = voc_year_dir / 'ImageSets' / 'Main' / 'val.txt'
        if test_file.exists():
            self.process_split('test', test_file, year)
        
        # 保存测试集COCO格式文件
        test_output = self.output_dir / f'instances_test{year}.json'
        with open(test_output, 'w') as f:
            json.dump(self.coco_data, f, indent=2)
        print(f"测试集COCO格式文件保存到: {test_output}")
        
        print(f"\nVOC{year} 数据集转换完成！")
        print(f"训练集: {len(temp_images)} 张图片, {len(temp_annotations)} 个标注")
        print(f"测试集: {len(self.coco_data['images'])} 张图片, {len(self.coco_data['annotations'])} 个标注")

def main():
    print("=" * 60)
    print("VOC数据集转换为COCO格式")
    print("=" * 60)
    
    # 数据集路径
    voc_root = '/root/autodl-tmp/VOCdevkit'
    output_dir = '/root/autodl-tmp/coco_format'
    
    # 检查VOC数据集是否存在
    if not os.path.exists(voc_root):
        print(f"错误: VOC数据集目录不存在 {voc_root}")
        return
    
    # 转换VOC2007
    if os.path.exists(os.path.join(voc_root, 'VOC2007')):
        print("\n转换VOC2007数据集...")
        # 创建转换器
        converter = VOC2COCO(voc_root, output_dir)
        converter.convert_dataset('2007')
    
    # 转换VOC2012（如果存在）
    if os.path.exists(os.path.join(voc_root, 'VOC2012')):
        print("\n转换VOC2012数据集...")
        # 创建转换器
        converter = VOC2COCO(voc_root, output_dir)
        converter.convert_dataset('2012')
    
    print("\n" + "=" * 60)
    print("数据集转换完成！")
    print(f"COCO格式文件保存在: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()