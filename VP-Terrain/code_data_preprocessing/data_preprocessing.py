
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from collections import Counter
from datetime import datetime
import os


#路径
# ============================================
DATA_PATH = "D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\VP-Terrain\data\_train_7_new.hdf5"
OUTPUT_DIR = "D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\VP-Terrain\data_analysis_report\_figures_report_train"
FIGURES_DIR = "D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\VP-Terrain\data_analysis_report\_figures_report_train\_figures"
REPORT_FILE = "D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\VP-Terrain\data_analysis_report\_figures_report_train/VP_Terrain_train_data_Analysis_Report.md"
# ============================================


try:
    plt.style.use('seaborn-v0_8-whitegrid')
except (OSError, IOError):
    try:
        plt.style.use('seaborn-whitegrid')
    except (OSError, IOError):
        plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# 地形类别
TERRAIN_CLASSES = {
    0: 'Asphalt (沥青)',
    1: 'Grass (草地)',
    2: 'Concrete (水泥)',
    3: 'Brick (砖路)',
    4: 'Board (地板)',
    5: 'Synthetic (塑胶)',
    6: 'Sand (沙地)'
}
TERRAIN_NAMES_EN = ['Asphalt', 'Grass', 'Concrete', 'Brick', 'Board', 'Synthetic', 'Sand']
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']
SIGNAL_CHANNELS = ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyro_X', 'Gyro_Y', 'Gyro_Z', 'Wheel_L', 'Wheel_R']


class VPTerrainReportGenerator:
    
    def __init__(self, hdf5_path, output_dir, figures_dir, report_file):
        self.hdf5_path = hdf5_path
        self.output_dir = output_dir
        self.figures_dir = figures_dir
        self.report_file = report_file
        
        # 创建目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        
        # 数据
        self.images = None
        self.labels = None
        self.signals = None
        self.timestamps = None
        self.images_reshaped = None
        self.signals_reshaped = None
        
        # 报告内容
        self.report_content = []
        
    def load_data(self):
        print("正在加载数据...")
        
        with h5py.File(self.hdf5_path, 'r') as f:
            self.images = f['images/images'][:]
            self.labels = f['labels/labels'][:]
            self.signals = f['signals/signals'][:]
            self.timestamps = f['timeStamps/timeStamps'][:]
        
        # Reshape
        self.images_reshaped = self.images.reshape(-1, 224, 224, 3)
        self.signals_reshaped = self.signals.reshape(-1, 100, 8)
        
        print(f"  数据加载完成: {len(self.labels)} 个样本")
        
    def add_to_report(self, content):
        self.report_content.append(content)
        
    def generate_header(self):
        header = f"""# VP-Terrain 数据集分析报告

> **数据文件**: `{os.path.basename(self.hdf5_path)}`

---

## 目录

1. [数据集概述](#1-数据集概述)
2. [数据结构分析](#2-数据结构分析)
3. [类别分布统计](#3-类别分布统计)
4. [视觉数据分析](#4-视觉数据分析)
5. [机体感知数据分析](#5-机体感知数据分析)
6. [时序信号分析](#6-时序信号分析)
7. [频域分析](#7-频域分析)
8. [特征相关性分析](#8-特征相关性分析)
9. [数据采集时间分析](#9-数据采集时间分析)
10. [总结](#10-总结)

---

"""
        self.add_to_report(header)
        
    def generate_overview(self):
        content = f"""## 1. 数据集概述

VP-Terrain 是一个用于**地表属性分类**的多模态数据集，包含视觉图像和机体感知信号两种模态的数据。

### 1.1 数据集背景

根据相关研究文献，该数据集：
- **采集平台**: 基于 Autolabor Pro 1 移动机器人
- **传感器配置**: 
  - Point Grey Flea 2 单目摄像头（视觉感知）
  - XW-GI5651 6DOF IMU（惯性测量单元）
  - 轮速编码器
- **采集地点**: 北京大学校园内不同地表环境区域
- **采集条件**: 包括中午阳光充足时段和傍晚光线不足时段

### 1.2 数据集规模

| 指标 | 数值 |
|------|------|
| 总样本数 | **{len(self.labels)}** |
| 地形类别数 | **7** |
| 图像尺寸 | **224 × 224 × 3** (RGB) |
| 信号长度 | **100 时间步 × 8 通道** (1秒@100Hz) |

### 1.3 地形类别

| 类别ID | 中文名称 | 英文名称 |
|--------|----------|----------|
| 0 | 沥青 | Asphalt |
| 1 | 草地 | Grass |
| 2 | 水泥 | Concrete |
| 3 | 砖路 | Brick |
| 4 | 地板 | Board |
| 5 | 塑胶 | Synthetic |
| 6 | 沙地 | Sand |

---

"""
        self.add_to_report(content)
        
    def generate_structure_analysis(self):
        
        # 计算基本统计量
        img_min, img_max = np.min(self.images), np.max(self.images)
        img_mean, img_std = np.mean(self.images), np.std(self.images)
        
        sig_min, sig_max = np.min(self.signals), np.max(self.signals)
        sig_mean, sig_std = np.mean(self.signals), np.std(self.signals)
        
        ts_min = datetime.fromtimestamp(np.min(self.timestamps))
        ts_max = datetime.fromtimestamp(np.max(self.timestamps))
        
        content = f"""## 2. 数据结构分析

### 2.1 HDF5 文件结构

```
{os.path.basename(self.hdf5_path)}
├── images/
│   └── images      # 视觉图像数据
├── labels/
│   └── labels      # 地形类别标签
├── signals/
│   └── signals     # 机体感知信号
└── timeStamps/
    └── timeStamps  # 数据采集时间戳
```

### 2.2 各数据集详细信息

#### 视觉数据 (images/images)

| 属性 | 值 |
|------|-----|
| 原始形状 | `({self.images.shape[0]}, {self.images.shape[1]})` |
| 还原形状 | `({self.images_reshaped.shape[0]}, {self.images_reshaped.shape[1]}, {self.images_reshaped.shape[2]}, {self.images_reshaped.shape[3]})` |
| 数据类型 | `float32` |
| 像素值范围 | [{img_min:.1f}, {img_max:.1f}] |
| 像素均值 | {img_mean:.2f} |
| 像素标准差 | {img_std:.2f} |

**说明**: 原始数据为展平的一维向量 (150528 = 224×224×3)，需要 reshape 为 (N, 224, 224, 3) 的 RGB 图像格式。

#### 标签数据 (labels/labels)

| 属性 | 值 |
|------|-----|
| 形状 | `({self.labels.shape[0]},)` |
| 数据类型 | `int64` |
| 类别范围 | [0, 6] |
| 类别数量 | 7 |

#### 信号数据 (signals/signals)

| 属性 | 值 |
|------|-----|
| 原始形状 | `({self.signals.shape[0]}, {self.signals.shape[1]})` |
| 还原形状 | `({self.signals_reshaped.shape[0]}, {self.signals_reshaped.shape[1]}, {self.signals_reshaped.shape[2]})` |
| 数据类型 | `float32` |
| 数值范围 | [{sig_min:.2f}, {sig_max:.2f}] |
| 均值 | {sig_mean:.2f} |
| 标准差 | {sig_std:.2f} |

**说明**: 原始数据为 800 维向量 (800 = 100时间步 × 8通道)，需要 reshape 为 (N, 100, 8)。

**8个信号通道**:
1. `Acc_X` - X轴加速度
2. `Acc_Y` - Y轴加速度  
3. `Acc_Z` - Z轴加速度
4. `Gyro_X` - X轴角速度
5. `Gyro_Y` - Y轴角速度
6. `Gyro_Z` - Z轴角速度
7. `Wheel_L` - 左轮角速度
8. `Wheel_R` - 右轮角速度

#### 时间戳数据 (timeStamps/timeStamps)

| 属性 | 值 |
|------|-----|
| 形状 | `({self.timestamps.shape[0]},)` |
| 数据类型 | `int64` |
| 起始时间 | {ts_min.strftime('%Y-%m-%d %H:%M:%S')} |
| 结束时间 | {ts_max.strftime('%Y-%m-%d %H:%M:%S')} |
| 采集跨度 | {(ts_max - ts_min).days} 天 |

---

"""
        self.add_to_report(content)
        
    def generate_class_distribution(self):
        
        # 统计
        unique, counts = np.unique(self.labels, return_counts=True)
        total = len(self.labels)
        
        # 计算不平衡比率
        imbalance_ratio = max(counts) / min(counts)
        
        # 生成图表
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 条形图
        bars = axes[0].bar(TERRAIN_NAMES_EN, counts, color=COLORS, edgecolor='black', linewidth=1.2)
        axes[0].set_xlabel('Terrain Class', fontsize=12)
        axes[0].set_ylabel('Sample Count', fontsize=12)
        axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        for bar, count in zip(bars, counts):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                        f'{count}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 饼图
        axes[1].pie(counts, labels=TERRAIN_NAMES_EN, colors=COLORS,
                   autopct='%1.1f%%', startangle=90, explode=[0.02]*7)
        axes[1].set_title('Class Proportion', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        fig_path = os.path.join(self.figures_dir, '1_class_distribution.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 生成表格
        table_rows = ""
        for i, (label_id, count) in enumerate(zip(unique, counts)):
            percentage = count / total * 100
            class_name = TERRAIN_CLASSES[label_id]
            table_rows += f"| {label_id} | {class_name} | {count} | {percentage:.2f}% |\n"
        
        content = f"""## 3. 类别分布统计

### 3.1 类别分布图

![类别分布](_figures/1_class_distribution.png)

### 3.2 详细统计表

| 类别ID | 类别名称 | 样本数量 | 占比 |
|--------|----------|----------|------|
{table_rows}
| **合计** | - | **{total}** | **100%** |

### 3.3 类别平衡性分析

| 指标 | 值 | 说明 |
|------|-----|------|
| 最多样本类别 | {TERRAIN_NAMES_EN[np.argmax(counts)]} ({max(counts)}) | 样本最充足的类别 |
| 最少样本类别 | {TERRAIN_NAMES_EN[np.argmin(counts)]} ({min(counts)}) | 样本最稀缺的类别 |
| 不平衡比率 | **{imbalance_ratio:.2f}** | 最大/最小样本数比值 |

**解读**:
- 不平衡比率 {imbalance_ratio:.2f} 表示样本最多的类别是样本最少类别的 {imbalance_ratio:.2f} 倍
- 一般认为比率 > 3 为中度不平衡，> 10 为严重不平衡
- 当前数据集属于**{'轻度' if imbalance_ratio < 3 else '中度' if imbalance_ratio < 10 else '严重'}不平衡**

---

"""
        self.add_to_report(content)
        
    def generate_visual_analysis(self):
        
        # 采样
        n_samples = min(500, len(self.images_reshaped))
        indices = np.random.choice(len(self.images_reshaped), n_samples, replace=False)
        sample_images = self.images_reshaped[indices] / 255.0
        sample_labels = self.labels[indices]
        
        # 计算RGB统计
        r_mean_all = np.mean(sample_images[..., 0])
        g_mean_all = np.mean(sample_images[..., 1])
        b_mean_all = np.mean(sample_images[..., 2])
        r_std_all = np.std(sample_images[..., 0])
        g_std_all = np.std(sample_images[..., 1])
        b_std_all = np.std(sample_images[..., 2])
        
        # 生成样本图像
        fig, axes = plt.subplots(2, 7, figsize=(18, 6))
        for class_id in range(7):
            mask = self.labels == class_id
            class_indices = np.where(mask)[0]
            if len(class_indices) >= 2:
                selected = np.random.choice(class_indices, 2, replace=False)
                for row, idx in enumerate(selected):
                    img = self.images_reshaped[idx] / 255.0
                    img = np.clip(img, 0, 1)
                    axes[row, class_id].imshow(img)
                    axes[row, class_id].axis('off')
                    if row == 0:
                        axes[row, class_id].set_title(TERRAIN_NAMES_EN[class_id], fontsize=11, fontweight='bold')
        plt.suptitle('Sample Images for Each Terrain Class', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '2_sample_images.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 生成图像统计图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # RGB分布
        r_mean = np.mean(sample_images[..., 0], axis=(1, 2))
        g_mean = np.mean(sample_images[..., 1], axis=(1, 2))
        b_mean = np.mean(sample_images[..., 2], axis=(1, 2))
        axes[0, 0].hist(r_mean, bins=50, alpha=0.6, color='red', label='R', density=True)
        axes[0, 0].hist(g_mean, bins=50, alpha=0.6, color='green', label='G', density=True)
        axes[0, 0].hist(b_mean, bins=50, alpha=0.6, color='blue', label='B', density=True)
        axes[0, 0].set_xlabel('Mean Pixel Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('RGB Channel Distribution')
        axes[0, 0].legend()
        
        # 亮度分布
        brightness = 0.299 * sample_images[..., 0] + 0.587 * sample_images[..., 1] + 0.114 * sample_images[..., 2]
        brightness_mean = np.mean(brightness, axis=(1, 2))
        for class_id in range(7):
            mask = sample_labels == class_id
            if np.sum(mask) > 0:
                axes[0, 1].hist(brightness_mean[mask], bins=25, alpha=0.5, color=COLORS[class_id], 
                               label=TERRAIN_NAMES_EN[class_id], density=True)
        axes[0, 1].set_xlabel('Mean Brightness')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Brightness Distribution by Class')
        axes[0, 1].legend(fontsize=8)
        
        # 亮度箱线图
        brightness_by_class = [brightness_mean[sample_labels == i] for i in range(7)]
        bp = axes[1, 0].boxplot(brightness_by_class, labels=TERRAIN_NAMES_EN, patch_artist=True)
        for patch, color in zip(bp['boxes'], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[1, 0].set_title('Brightness by Class')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 纹理复杂度
        img_std = np.std(sample_images, axis=(1, 2, 3))
        for class_id in range(7):
            mask = sample_labels == class_id
            if np.sum(mask) > 0:
                axes[1, 1].hist(img_std[mask], bins=25, alpha=0.5, color=COLORS[class_id], 
                               label=TERRAIN_NAMES_EN[class_id], density=True)
        axes[1, 1].set_xlabel('Standard Deviation')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Texture Complexity by Class')
        axes[1, 1].legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '3_image_statistics.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 各类别亮度统计
        brightness_stats = []
        for class_id in range(7):
            mask = sample_labels == class_id
            if np.sum(mask) > 0:
                b = brightness_mean[mask]
                brightness_stats.append({
                    'class': TERRAIN_NAMES_EN[class_id],
                    'mean': np.mean(b),
                    'std': np.std(b)
                })
        
        brightness_table = ""
        for s in brightness_stats:
            brightness_table += f"| {s['class']} | {s['mean']:.4f} | {s['std']:.4f} |\n"
        
        content = f"""## 4. 视觉数据分析

### 4.1 样本图像展示

![样本图像](_figures/2_sample_images.png)

**说明**: 展示了每个地形类别的随机样本图像（每类2张），可以直观感受不同地形的视觉特征差异。

### 4.2 图像统计特征

![图像统计](_figures/3_image_statistics.png)

### 4.3 RGB 通道统计

| 通道 | 均值 | 标准差 |
|------|------|--------|
| R (红) | {r_mean_all:.4f} | {r_std_all:.4f} |
| G (绿) | {g_mean_all:.4f} | {g_std_all:.4f} |
| B (蓝) | {b_mean_all:.4f} | {b_std_all:.4f} |

**图表解读**:

1. **RGB通道分布图** (左上)
   - 显示三个颜色通道的像素值分布
   - 可用于判断是否存在颜色偏移或需要颜色标准化

2. **亮度分布图** (右上)
   - 亮度计算公式: \\(L = 0.299R + 0.587G + 0.114B\\)
   - 不同类别的亮度分布有一定差异，可作为分类特征

3. **亮度箱线图** (左下)
   - 直观展示各类别亮度的中位数、四分位数和异常值
   - 便于比较类间差异

4. **纹理复杂度** (右下)
   - 使用图像标准差衡量纹理复杂程度
   - 标准差越大，表示图像细节越丰富

### 4.4 各类别亮度统计

| 类别 | 亮度均值 | 亮度标准差 |
|------|----------|------------|
{brightness_table}

---

"""
        self.add_to_report(content)
        
    def generate_signal_analysis(self):
        
        # 计算各通道统计量
        channel_stats = []
        for ch_idx in range(8):
            ch_data = self.signals_reshaped[:, :, ch_idx]
            channel_stats.append({
                'name': SIGNAL_CHANNELS[ch_idx],
                'mean': np.mean(ch_data),
                'std': np.std(ch_data),
                'min': np.min(ch_data),
                'max': np.max(ch_data)
            })
        
        # 生成信号箱线图
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        axes = axes.flatten()
        
        for ch_idx in range(8):
            ax = axes[ch_idx]
            data_by_class = []
            for class_id in range(7):
                mask = self.labels == class_id
                ch_mean = np.mean(self.signals_reshaped[mask, :, ch_idx], axis=1)
                data_by_class.append(ch_mean)
            bp = ax.boxplot(data_by_class, labels=TERRAIN_NAMES_EN, patch_artist=True)
            for patch, color in zip(bp['boxes'], COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_title(f'{SIGNAL_CHANNELS[ch_idx]}', fontsize=12, fontweight='bold')
            ax.tick_params(axis='x', rotation=45, labelsize=8)
        
        plt.suptitle('Signal Channels by Class', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '4_signal_boxplot.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 生成方差对比图
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        acc_var_by_class = []
        gyro_var_by_class = []
        for class_id in range(7):
            mask = self.labels == class_id
            class_signals = self.signals_reshaped[mask]
            acc_var = np.var(class_signals[:, :, :3], axis=1).mean(axis=1)
            gyro_var = np.var(class_signals[:, :, 3:6], axis=1).mean(axis=1)
            acc_var_by_class.append(acc_var)
            gyro_var_by_class.append(gyro_var)
        
        bp = axes[0].boxplot(acc_var_by_class, labels=TERRAIN_NAMES_EN, patch_artist=True)
        for patch, color in zip(bp['boxes'], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[0].set_title('Acceleration Variance (Bumpiness)', fontsize=12, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        
        bp = axes[1].boxplot(gyro_var_by_class, labels=TERRAIN_NAMES_EN, patch_artist=True)
        for patch, color in zip(bp['boxes'], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[1].set_title('Gyroscope Variance', fontsize=12, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '5_signal_variance.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 统计表格
        stats_table = ""
        for s in channel_stats:
            stats_table += f"| {s['name']} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} |\n"
        
        content = f"""## 5. 机体感知数据分析

### 5.1 信号通道说明

机体感知数据包含 8 个通道，采样率为 100Hz，每个样本记录 1 秒数据（100个时间步）：

| 通道序号 | 通道名称 | 物理含义 | 单位 |
|----------|----------|----------|------|
| 0 | Acc_X | X轴线性加速度 | m/s² |
| 1 | Acc_Y | Y轴线性加速度 | m/s² |
| 2 | Acc_Z | Z轴线性加速度 | m/s² |
| 3 | Gyro_X | X轴角速度 | rad/s |
| 4 | Gyro_Y | Y轴角速度 | rad/s |
| 5 | Gyro_Z | Z轴角速度 | rad/s |
| 6 | Wheel_L | 左轮角速度 | rad/s |
| 7 | Wheel_R | 右轮角速度 | rad/s |

### 5.2 各通道统计量

| 通道 | 均值 | 标准差 | 最小值 | 最大值 |
|------|------|--------|--------|--------|
{stats_table}

### 5.3 信号分布箱线图（按类别）

![信号箱线图](_figures/4_signal_boxplot.png)

**图表解读**:
- 每个子图展示一个信号通道在不同地形类别下的分布
- 箱线图显示中位数（箱内横线）、四分位距（箱体）和异常值（圆点）
- **类间差异越大**，该通道对分类的贡献越大

### 5.4 信号方差对比（颠簸程度指标）

![信号方差](_figures/5_signal_variance.png)

**图表解读**:

1. **加速度方差** (左图)
   - 反映地形的**颠簸程度**
   - 方差越大，地形越不平整
   - 预期：草地、沙地方差较大；塑胶、沥青方差较小

2. **角速度方差** (右图)
   - 反映机器人的**旋转稳定性**
   - 方差越大，行驶越不稳定


---

"""
        self.add_to_report(content)
        
    def generate_time_series_analysis(self):
        
        # 时序信号示例
        fig, axes = plt.subplots(7, 1, figsize=(14, 18))
        t = np.arange(100) / 100
        
        for class_id in range(7):
            ax = axes[class_id]
            mask = self.labels == class_id
            indices = np.where(mask)[0]
            if len(indices) > 0:
                sample_idx = indices[np.random.randint(len(indices))]
                sample = self.signals_reshaped[sample_idx]
                ax.plot(t, sample[:, 0], label='Acc_X', color='red', alpha=0.8, linewidth=1.5)
                ax.plot(t, sample[:, 1], label='Acc_Y', color='green', alpha=0.8, linewidth=1.5)
                ax.plot(t, sample[:, 2], label='Acc_Z', color='blue', alpha=0.8, linewidth=1.5)
            ax.set_title(f'{TERRAIN_CLASSES[class_id]}', fontsize=11, fontweight='bold', color=COLORS[class_id])
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Acceleration')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_xlim([0, 1])
        
        plt.suptitle('Acceleration Time Series Examples', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '6_time_series.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        content = f"""## 6. 时序信号分析

### 6.1 各类别加速度时序示例

![时序信号](_figures/6_time_series.png)

**图表解读**:
- 每个子图展示一个地形类别的加速度信号随时间变化
- 红色: X轴加速度，绿色: Y轴加速度，蓝色: Z轴加速度
- 信号的**振幅**和**波动频率**反映地形特征

**观察要点**:
1. **沥青/塑胶**: 信号相对平稳，振幅小
2. **草地/沙地**: 信号波动大，高频成分多
3. **砖路**: 可能存在周期性振动（砖块间隙）
4. **Z轴加速度**: 通常最能反映地形颠簸程度

---

"""
        self.add_to_report(content)
        
    def generate_frequency_analysis(self):
        
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        sampling_rate = 100
        
        for class_id in range(7):
            row, col = class_id // 4, class_id % 4
            ax = axes[row, col]
            
            mask = self.labels == class_id
            class_signals = self.signals_reshaped[mask]
            z_acc = class_signals[:, :, 2]
            
            n_samples = z_acc.shape[1]
            freq = fftfreq(n_samples, 1/sampling_rate)[:n_samples//2]
            fft_vals = np.abs(fft(z_acc, axis=1))[:, :n_samples//2]
            mean_spectrum = np.mean(fft_vals, axis=0)
            std_spectrum = np.std(fft_vals, axis=0)
            
            ax.plot(freq, mean_spectrum, color=COLORS[class_id], linewidth=1.5)
            ax.fill_between(freq, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum,
                           alpha=0.3, color=COLORS[class_id])
            ax.set_title(f'{TERRAIN_NAMES_EN[class_id]}', fontsize=11, fontweight='bold')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude')
            ax.set_xlim([0, 50])
        
        axes[1, 3].axis('off')
        plt.suptitle('Frequency Spectrum of Z-Axis Acceleration', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '7_frequency_analysis.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        content = f"""## 7. 频域分析

### 7.1 Z轴加速度频谱

![频域分析](_figures/7_frequency_analysis.png)

**图表解读**:
- 使用**快速傅里叶变换 (FFT)** 将时域信号转换为频域
- 横轴为频率 (Hz)，纵轴为幅度
- 阴影区域表示 ±1 标准差范围

**频谱特征含义**:

| 频率范围 | 物理含义 |
|----------|----------|
| 0-5 Hz | 低频成分，与整体运动趋势相关 |
| 5-20 Hz | 中频成分，与地形主要振动相关 |
| 20-50 Hz | 高频成分，与细微颠簸和噪声相关 |


---

"""
        self.add_to_report(content)
        
    def generate_correlation_analysis(self):
        
        # 计算特征
        features = []
        feature_names = []
        for ch_idx in range(8):
            ch_data = self.signals_reshaped[:, :, ch_idx]
            features.append(np.mean(ch_data, axis=1))
            features.append(np.std(ch_data, axis=1))
            feature_names.extend([f'{SIGNAL_CHANNELS[ch_idx]}_mean', f'{SIGNAL_CHANNELS[ch_idx]}_std'])
        features = np.array(features)
        
        # 相关性矩阵
        corr_matrix = np.corrcoef(features)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label='Correlation')
        ax.set_xticks(range(len(feature_names)))
        ax.set_yticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=90, fontsize=8)
        ax.set_yticklabels(feature_names, fontsize=8)
        ax.set_title('Signal Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '8_correlation.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        content = f"""## 8. 特征相关性分析

### 8.1 信号特征相关性矩阵

![相关性矩阵](_figures/8_correlation.png)

**图表解读**:
- 矩阵展示了 16 个信号特征（8通道 × 2统计量）之间的相关系数
- **红色**: 正相关 (接近 +1)
- **蓝色**: 负相关 (接近 -1)
- **白色**: 无相关 (接近 0)

---

"""
        self.add_to_report(content)
        
    def generate_timestamp_analysis(self):
        
        dates = [datetime.fromtimestamp(ts) for ts in self.timestamps]
        hours = [d.hour for d in dates]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        axes[0].hist(hours, bins=24, range=(0, 24), color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Hour of Day')
        axes[0].set_ylabel('Sample Count')
        axes[0].set_title('Data Collection Time Distribution')
        
        for class_id in range(7):
            mask = self.labels == class_id
            class_hours = [hours[i] for i in range(len(hours)) if mask[i]]
            axes[1].hist(class_hours, bins=24, range=(0, 24), alpha=0.5,
                        color=COLORS[class_id], label=TERRAIN_NAMES_EN[class_id])
        axes[1].set_xlabel('Hour of Day')
        axes[1].set_ylabel('Sample Count')
        axes[1].set_title('Collection Time by Class')
        axes[1].legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '9_timestamp.png'), dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # 统计采集时段
        daytime = sum(1 for h in hours if 6 <= h < 18)
        nighttime = len(hours) - daytime
        
        content = f"""## 9. 数据采集时间分析

### 9.1 采集时间分布

![时间戳分析](_figures/9_timestamp.png)

### 9.2 采集时段统计

| 时段 | 样本数 | 占比 |
|------|--------|------|
| 白天 (6:00-18:00) | {daytime} | {daytime/len(hours)*100:.1f}% |
| 夜晚 (18:00-6:00) | {nighttime} | {nighttime/len(hours)*100:.1f}% |


---

"""
        self.add_to_report(content)
        
    def generate_summary(self):
        
        content = f"""## 10. 总结


| 方面 | 特点 |
|------|------|
| **数据规模** | {len(self.labels)} 个样本，7 类地形 |
| **多模态** | 视觉 (224×224 RGB) + 机体感知 (8通道@100Hz) |
| **时间跨度** | 数据采集跨越多个日期 |


---

## 附录

### A. 文件说明

| 文件 | 说明 |
|------|------|
| `_figures/1_class_distribution.png` | 类别分布图 |
| `_figures/2_sample_images.png` | 样本图像展示 |
| `_figures/3_image_statistics.png` | 图像统计特征 |
| `_figures/4_signal_boxplot.png` | 信号箱线图 |
| `_figures/5_signal_variance.png` | 信号方差对比 |
| `_figures/6_time_series.png` | 时序信号示例 |
| `_figures/7_frequency_analysis.png` | 频域分析 |
| `_figures/8_correlation.png` | 特征相关性 |
| `_figures/9_timestamp.png` | 时间戳分析 |

### B. 参考文献

1. TerraX: Visual Terrain Classification Enhanced by Vision-Language Models (IROS 2025)
2. VINet: Visual and Inertial-based Terrain Classification (ICRA 2023)
3. Proprioception Is All You Need: Terrain Classification for Boreal Forests (IROS 2024)

---

"""
        self.add_to_report(content)
        
    def save_report(self):
        with open(self.report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report_content))
        print(f"\n📄 报告已保存: {self.report_file}")
        
    def generate_full_report(self):
        print("=" * 60)
        print("VP-Terrain 数据集分析报告生成器")
        print("=" * 60)
        
        self.load_data()
        
        print("\n正在生成报告...")
        self.generate_header()
        self.generate_overview()
        self.generate_structure_analysis()
        self.generate_class_distribution()
        self.generate_visual_analysis()
        self.generate_signal_analysis()
        self.generate_time_series_analysis()
        self.generate_frequency_analysis()
        self.generate_correlation_analysis()
        self.generate_timestamp_analysis()
        self.generate_summary()
        self.save_report()
        print("\n" + "=" * 60)
        print("报告生成完成")
        print(f"输出目录: {self.output_dir}")
        print(f"报告文件: {self.report_file}")
        print(f"图表目录: {self.figures_dir}")
        print("=" * 60)


# ========== 主程序 ==========
if __name__ == "__main__":
    if os.path.exists(DATA_PATH):
        generator = VPTerrainReportGenerator(
            hdf5_path=DATA_PATH,
            output_dir=OUTPUT_DIR,
            figures_dir=FIGURES_DIR,
            report_file=REPORT_FILE
        )
        generator.generate_full_report()
    else:
        print(f"数据集文件不存在: {DATA_PATH}")
        print("请修改 DATA_PATH 变量为正确的文件路径")
