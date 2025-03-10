# 随机可视化n张图片 
# python vis_oral_json.py --json_dir /path/to/json/files --img_dir /path/to/images --output_dir /path/to/output --mode random --num_samples 5
# python vis_oral_json.py --json_dir ./json_2/ --img_dir ./MM-Oral-OPG-images/ --output_dir ./visulization --mode random --num_samples 5

# 可视化所有图片
# python vis_oral_json.py --json_dir /path/to/json/files --img_dir /path/to/images --output_dir /path/to/output --mode all

import os
import json
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import argparse
from matplotlib.path import Path

def load_json_data(json_file):
    """加载单个JSON文件并返回数据"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def save_image_with_annotations(data, img_dir, output_path, img_id=None, file_name=None):
    """可视化单个图像及其标注，并保存结果"""
    if img_id is None and file_name is None:
        img_id = data["image_id"]
        file_name = data["file_name"]
    
    img_path = os.path.join(img_dir, file_name)
    
    # 检查文件是否存在
    if not os.path.exists(img_path):
        print(f"错误: 图像文件不存在 {img_path}")
        return False
    
    # 创建图形
    plt.figure(figsize=(12, 10))
    
    # 尝试加载图像
    try:
        img = plt.imread(img_path)
        plt.imshow(img)
    except:
        # 如果图像无法加载，创建一个空白画布，使用图像尺寸
        img_width = data.get("image_width", 1000)
        img_height = data.get("image_height", 1000)
        plt.xlim(0, img_width)
        plt.ylim(img_height, 0)  # 反转Y轴以匹配图像坐标系
        
    plt.axis('off')
    plt.title(f"图像ID: {data['image_id']}, 文件名: {file_name}")
    
    # 获取图像的标注 - 牙齿
    teeth = data.get("properties", {}).get("Teeth", [])
    
    # 绘制牙齿标注
    if teeth:
        # 使用不同颜色绘制每个标注
        for i, tooth in enumerate(teeth):
            color = plt.cm.tab10(i % 10)
            
            # 绘制边界框
            if 'bbox' in tooth:
                bbox = tooth['bbox']
                rect = patches.Rectangle(
                    (bbox[0], bbox[1]), bbox[2], bbox[3],
                    fill=False, edgecolor=color, linewidth=2
                )
                plt.gca().add_patch(rect)
            
            # 显示牙齿ID和得分
            if 'tooth_id' in tooth and 'bbox' in tooth:
                plt.text(
                    bbox[0], bbox[1] - 5, 
                    f"ID:{tooth['tooth_id']} ({tooth.get('score', 0):.2f})",
                    bbox=dict(facecolor=color, alpha=0.5)
                )
            
            # 绘制疾病条件
            conditions = tooth.get("conditions", {})
            for cond_name, cond_data in conditions.items():
                if cond_data.get("present", False) and "bbox" in cond_data:
                    # 使用不同的颜色表示疾病
                    cond_color = 'red'
                    
                    # 绘制疾病边界框
                    cond_bbox = cond_data["bbox"]
                    cond_rect = patches.Rectangle(
                        (cond_bbox[0], cond_bbox[1]), cond_bbox[2], cond_bbox[3],
                        fill=False, edgecolor=cond_color, linewidth=2, linestyle='--'
                    )
                    plt.gca().add_patch(cond_rect)
                    
                    # 显示疾病名称
                    plt.text(
                        cond_bbox[0], cond_bbox[1] - 5, 
                        f"{cond_name} ({cond_data.get('score', 0):.2f})",
                        bbox=dict(facecolor=cond_color, alpha=0.3)
                    )
                    
                    # 如果有分割数据，绘制分割
                    if "segmentation" in cond_data:
                        for seg in cond_data["segmentation"]:
                            # 将分割点转换为适合 Polygon 的格式
                            if len(seg) >= 6:  # 确保至少有3个点
                                poly = np.array(seg).reshape((len(seg)//2, 2))
                                plt.fill(poly[:, 0], poly[:, 1], alpha=0.2, color=cond_color)
                                plt.plot(poly[:, 0], poly[:, 1], color=cond_color, linewidth=1)
        
        print(f"图像 {file_name} 有 {len(teeth)} 个牙齿标注")
    else:
        print(f"图像 {file_name} 没有牙齿标注")
    
    # 获取并绘制象限标注
    quadrants = data.get("properties", {}).get("Quadrants", [])
    if quadrants:
        for quadrant in quadrants:
            if quadrant.get("present", False) and "bbox" in quadrant:
                q_bbox = quadrant["bbox"]
                q_rect = patches.Rectangle(
                    (q_bbox[0], q_bbox[1]), q_bbox[2], q_bbox[3],
                    fill=False, edgecolor='red', linewidth=1, linestyle=':'
                )
                plt.gca().add_patch(q_rect)
                plt.text(
                    q_bbox[0], q_bbox[1], 
                    f"{quadrant.get('quadrant', '')} ({quadrant.get('score', 0):.2f})",
                    color='red', fontsize=8
                )
    
    # 保存图像
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close()  # 关闭图形以释放内存
    
    return True

def main():
    parser = argparse.ArgumentParser(description='可视化牙齿JSON数据')
    parser.add_argument('--json_dir', type=str, required=True, help='包含JSON文件的目录')
    parser.add_argument('--img_dir', type=str, required=True, help='包含图像文件的目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出可视化结果的目录')
    parser.add_argument('--mode', type=str, choices=['random', 'all'], default='random', help='可视化模式: random (随机选择) 或 all (所有图像)')
    parser.add_argument('--num_samples', type=int, default=5, help='随机模式下要可视化的图像数量')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = glob.glob(os.path.join(args.json_dir, "*.json"))
    if not json_files:
        print(f"错误: 在 {args.json_dir} 中没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    if args.mode == 'random':
        # 随机选择指定数量的JSON文件
        sample_size = min(args.num_samples, len(json_files))
        selected_files = random.sample(json_files, sample_size)
        print(f"随机选择了 {sample_size} 个文件进行可视化")
    else:
        # 使用所有JSON文件
        selected_files = json_files
        print(f"将可视化所有 {len(selected_files)} 个文件")
    
    # 处理选中的文件
    successful_saves = 0
    for i, json_file in enumerate(selected_files):
        try:
            # 加载JSON数据
            data = load_json_data(json_file)
            
            # 生成输出文件名
            base_name = os.path.basename(json_file)
            output_filename = f"vis_{i+1}_{base_name.replace('.json', '.png')}"
            output_path = os.path.join(args.output_dir, output_filename)
            
            # 可视化并保存
            if save_image_with_annotations(data, args.img_dir, output_path):
                successful_saves += 1
                print(f"成功保存可视化结果到: {output_path}")
            
        except Exception as e:
            print(f"处理文件 {json_file} 时出错: {str(e)}")
    
    print(f"\n总共成功保存了 {successful_saves}/{len(selected_files)} 张可视化图像")
    print(f"可视化结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()
