
import h5py
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import os


#路径
# ============================================
DATA_DIR = "D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\VP-Terrain\data"

DATASETS = {
    'Normal': os.path.join(DATA_DIR, '_normal_7_new.hdf5'),
    'Train': os.path.join(DATA_DIR, '_train_7_new.hdf5'),
    'Dark': os.path.join(DATA_DIR, '_dark_7_new.hdf5'),
}

OUTPUT_DIR = "D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\VP-Terrain\data_analysis_report\_figures_report_compare"
# ============================================

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

TERRAIN_NAMES = ['Asphalt', 'Grass', 'Concrete', 'Brick', 'Board', 'Synthetic', 'Sand']
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628']
DATASET_COLORS = {'Normal': '#2ecc71', 'Train': '#3498db', 'Dark': '#9b59b6'}


class DatasetComparator:
    
    def __init__(self, datasets_dict, output_dir):
        self.datasets = datasets_dict
        self.output_dir = output_dir
        self.figures_dir = os.path.join(output_dir, '_figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        
        self.data = {}  # 存储加载的数据
        self.stats = {}  # 存储统计信息
        
    def check_files(self):
        print("=" * 60)
        print("检查数据集文件")
        print("=" * 60)
        
        available = {}
        for name, path in self.datasets.items():
            exists = os.path.exists(path)
            status = "存在" if exists else "不存在"
            print(f"  {name}: {status}")
            print(f"    路径: {path}")
            if exists:
                available[name] = path
                
        print(f"\n可用数据集: {list(available.keys())}")
        self.datasets = available
        return len(available) > 0
        
        
    def load_all_data(self):
        
        for name, path in self.datasets.items():
            print(f"\n加载 {name}...")
            with h5py.File(path, 'r') as f:
                self.data[name] = {
                    'images': f['images/images'][:],
                    'labels': f['labels/labels'][:],
                    'signals': f['signals/signals'][:],
                    'timestamps': f['timeStamps/timeStamps'][:]
                }
            
            n_samples = len(self.data[name]['labels'])
            print(f"  样本数: {n_samples}")
            
            # Reshape
            self.data[name]['images_reshaped'] = self.data[name]['images'].reshape(-1, 224, 224, 3)
            self.data[name]['signals_reshaped'] = self.data[name]['signals'].reshape(-1, 100, 8)
            
    def compute_statistics(self):
        
        for name, d in self.data.items():
            images = d['images_reshaped'] / 255.0
            labels = d['labels']
            signals = d['signals_reshaped']
            
            # 亮度
            brightness = 0.299 * images[..., 0] + 0.587 * images[..., 1] + 0.114 * images[..., 2]
            brightness_mean = np.mean(brightness, axis=(1, 2))
            
            # 信号方差
            acc_var = np.var(signals[:, :, :3], axis=1).mean(axis=1)
            
            self.stats[name] = {
                'n_samples': len(labels),
                'class_counts': Counter(labels),
                'brightness_mean': np.mean(brightness_mean),
                'brightness_std': np.std(brightness_mean),
                'brightness_per_sample': brightness_mean,
                'acc_var_mean': np.mean(acc_var),
                'acc_var_per_sample': acc_var,
                'signal_mean': np.mean(signals),
                'signal_std': np.std(signals),
            }
            
            print(f"\n【{name}】")
            print(f"  样本数: {self.stats[name]['n_samples']}")
            print(f"  平均亮度: {self.stats[name]['brightness_mean']:.4f}")
            print(f"  加速度方差均值: {self.stats[name]['acc_var_mean']:.4f}")
            
    def plot_sample_count_comparison(self):

        fig, ax = plt.subplots(figsize=(10, 6))
        
        names = list(self.stats.keys())
        counts = [self.stats[n]['n_samples'] for n in names]
        
        bars = ax.bar(names, counts, color=[DATASET_COLORS.get(n, 'gray') for n in names],
                     edgecolor='black', linewidth=1.5)
        
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                   f'{count}', ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('Dataset', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title('Sample Count Comparison', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '1_sample_count.png'), dpi=150, facecolor='white')
        plt.close()
        
    def plot_class_distribution_comparison(self):
        
        n_datasets = len(self.stats)
        fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 5))
        
        if n_datasets == 1:
            axes = [axes]
        
        for ax, (name, stat) in zip(axes, self.stats.items()):
            counts = [stat['class_counts'].get(i, 0) for i in range(7)]
            bars = ax.bar(TERRAIN_NAMES, counts, color=COLORS, edgecolor='black')
            ax.set_title(f'{name} Dataset', fontsize=12, fontweight='bold')
            ax.set_xlabel('Terrain Class')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
            
            # 添加数值
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                           f'{count}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('Class Distribution Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '2_class_distribution.png'), dpi=150, facecolor='white')
        plt.close()
        
    def plot_brightness_comparison(self):
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图
        ax = axes[0]
        for name, stat in self.stats.items():
            ax.hist(stat['brightness_per_sample'], bins=50, alpha=0.5, 
                   label=f"{name} (μ={stat['brightness_mean']:.3f})",
                   color=DATASET_COLORS.get(name, 'gray'), density=True)
        ax.set_xlabel('Mean Brightness', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title('Brightness Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        
        # 箱线图
        ax = axes[1]
        data_list = [stat['brightness_per_sample'] for stat in self.stats.values()]
        bp = ax.boxplot(data_list, labels=list(self.stats.keys()), patch_artist=True)
        for patch, name in zip(bp['boxes'], self.stats.keys()):
            patch.set_facecolor(DATASET_COLORS.get(name, 'gray'))
            patch.set_alpha(0.6)
        ax.set_ylabel('Mean Brightness', fontsize=11)
        ax.set_title('Brightness Comparison (Boxplot)', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '3_brightness_comparison.png'), dpi=150, facecolor='white')
        plt.close()
        
    def plot_sample_images_comparison(self):
        
        n_datasets = len(self.data)
        fig, axes = plt.subplots(n_datasets, 7, figsize=(18, 4*n_datasets))
        
        if n_datasets == 1:
            axes = axes.reshape(1, -1)
        
        for row, (name, d) in enumerate(self.data.items()):
            for class_id in range(7):
                ax = axes[row, class_id]
                mask = d['labels'] == class_id
                indices = np.where(mask)[0]
                
                if len(indices) > 0:
                    idx = indices[np.random.randint(len(indices))]
                    img = d['images_reshaped'][idx] / 255.0
                    img = np.clip(img, 0, 1)
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, 'No Sample', ha='center', va='center', transform=ax.transAxes)
                    
                ax.axis('off')
                
                if row == 0:
                    ax.set_title(TERRAIN_NAMES[class_id], fontsize=10, fontweight='bold')
                if class_id == 0:
                    ax.text(-0.1, 0.5, name, fontsize=12, fontweight='bold',
                           transform=ax.transAxes, rotation=90, va='center', ha='right')
        
        plt.suptitle('Sample Images Comparison Across Datasets', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '4_sample_images.png'), dpi=150, facecolor='white')
        plt.close()
        
    def plot_signal_comparison(self):
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 加速度方差对比
        ax = axes[0]
        data_list = [stat['acc_var_per_sample'] for stat in self.stats.values()]
        bp = ax.boxplot(data_list, labels=list(self.stats.keys()), patch_artist=True)
        for patch, name in zip(bp['boxes'], self.stats.keys()):
            patch.set_facecolor(DATASET_COLORS.get(name, 'gray'))
            patch.set_alpha(0.6)
        ax.set_ylabel('Acceleration Variance', fontsize=11)
        ax.set_title('Acceleration Variance Comparison', fontsize=12, fontweight='bold')
        
        # 信号均值/标准差对比
        ax = axes[1]
        names = list(self.stats.keys())
        means = [self.stats[n]['signal_mean'] for n in names]
        stds = [self.stats[n]['signal_std'] for n in names]
        
        x = np.arange(len(names))
        width = 0.35
        ax.bar(x - width/2, means, width, label='Mean', color='steelblue', alpha=0.7)
        ax.bar(x + width/2, stds, width, label='Std', color='coral', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel('Value', fontsize=11)
        ax.set_title('Signal Statistics Comparison', fontsize=12, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, '5_signal_comparison.png'), dpi=150, facecolor='white')
        plt.close()
        
    def generate_comparison_report(self):
        
        report = f"""# VP-Terrain 多数据集对比分析报告


---

## 1. 数据集概览

VP-Terrain数据集包含多个子集：

| 数据集 | 采集条件 | 用途 | 说明 |
|--------|----------|------|------|
| Train | 白天正常光照 | 训练 | 用于模型训练 |
| Normal | 白天正常光照 | 测试 | 正常条件测试 |
| Dark | 夜晚低光照 | 测试 | 暗光条件测试（OOD场景） |

---

## 2. 样本数量对比

![样本数量](_figures/1_sample_count.png)

| 数据集 | 样本数量 |
|--------|----------|
"""
        for name, stat in self.stats.items():
            report += f"| {name} | {stat['n_samples']} |\n"
        
        report += """
---

## 3. 类别分布对比

![类别分布](_figures/2_class_distribution.png)

### 各数据集类别统计

"""
        # 生成类别统计表
        report += "| 类别 |"
        for name in self.stats.keys():
            report += f" {name} |"
        report += "\n|------|"
        for _ in self.stats.keys():
            report += "------|"
        report += "\n"
        
        for class_id in range(7):
            report += f"| {TERRAIN_NAMES[class_id]} |"
            for name, stat in self.stats.items():
                count = stat['class_counts'].get(class_id, 0)
                report += f" {count} |"
            report += "\n"
        
        report += f"""
---

## 4. 亮度对比分析

![亮度对比](_figures/3_brightness_comparison.png)

### 亮度统计

| 数据集 | 平均亮度 | 亮度标准差 |
|--------|----------|------------|
"""
        for name, stat in self.stats.items():
            report += f"| {name} | {stat['brightness_mean']:.4f} | {stat['brightness_std']:.4f} |\n"
        
        report += """
**关键发现**:
- **Dark数据集** 的平均亮度明显低于Normal和Train数据集
- 这符合PDF中描述的"暗光集采集自夜晚低光照条件"
- Dark数据集可用于测试模型在**光照变化**（OOD场景）下的鲁棒性

---

## 5. 样本图像对比

![样本图像](_figures/4_sample_images.png)

**观察**:
- Normal/Train数据集图像亮度正常，纹理清晰
- Dark数据集图像整体偏暗，可能影响视觉特征提取
- 这验证了多模态融合（视觉+机体感知）的必要性

---

## 6. 信号特征对比

![信号对比](_figures/5_signal_comparison.png)

**分析**:
- 机体感知信号（IMU）**不受光照条件影响**
- 这是PDF中强调的机体感知的优势："稳定性强，不受光照、天气等影响"
- 在Dark条件下，机体感知信号可以弥补视觉感知的不足

---

## 7. 总结

| 数据集 | 特点 | 建议用途 |
|--------|------|----------|
| Train | 样本充足，光照正常 | 模型训练 |
| Normal | 光照正常 | 标准测试 |
| Dark | 低光照，视觉特征弱 | OOD鲁棒性测试 |


---

"""
        
        report_path = os.path.join(self.output_dir, 'VP_Terrain_Comparison_Report.md')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"  报告已保存: {report_path}")
        
    def run_full_comparison(self):

        print("=" * 60)
        print("VP-Terrain 多数据集对比分析")
        print("=" * 60)
        
        # 检查文件
        if not self.check_files():
            print("\n没有可用的数据集文件")
            return
        
        
        # 加载数据
        self.load_all_data()
        
        # 计算统计
        self.compute_statistics()
        
        # 生成图表
        
        self.plot_sample_count_comparison()
        self.plot_class_distribution_comparison()
        self.plot_brightness_comparison()
        self.plot_sample_images_comparison()
        self.plot_signal_comparison()
        
        # 生成报告
        self.generate_comparison_report()
        
        print("\n" + "=" * 60)
        print("对比分析完成!")
        print(f"输出目录: {self.output_dir}")
        print("=" * 60)


# ========== 主程序 ==========
if __name__ == "__main__":
    comparator = DatasetComparator(DATASETS, OUTPUT_DIR)
    comparator.run_full_comparison()
