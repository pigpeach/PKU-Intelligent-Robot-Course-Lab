"""
TMC 证据融合完整测试脚本
对比原始贝叶斯融合与 TMC 证据融合
输出结果同时保存到 txt 文件
"""

import numpy as np
import h5py
import sys
from datetime import datetime

# 自定义输出类，同时输出到控制台和文件
class TeeOutput:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()

# ============================================================
# 配置路径（请根据实际情况修改）
# ============================================================
DATA_PATH = r"D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\VP-Terrain\data\_dark_7_new.hdf5"
RESULT_FOLDER = r"D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\SSL_poss\mobilenet_0.05"

PRO_CONF_PATH = f"{RESULT_FOLDER}\\pro_dark_conf"
VIS_CONF_PATH = f"{RESULT_FOLDER}\\vision_dark_conf"

# 生成日志文件名（包含时间戳）
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = f"{RESULT_FOLDER}\\tmc_fusion_results_{timestamp}.txt"

# 启动输出重定向
tee = TeeOutput(LOG_FILE)
sys.stdout = tee

print("="*70)
print(f"TMC 证据融合测试")
print(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"日志文件: {LOG_FILE}")
print("="*70)

try:
    # 导入证据融合模块
    sys.path.append('.')
    from evidence_fusion import TMCFusion, analyze_uncertainty, print_analysis_report

    # ============================================================
    # 读取数据
    # ============================================================
    print("\n" + "="*70)
    print("加载数据...")
    print("="*70)

    # 读取真实标签
    f = h5py.File(DATA_PATH, 'r')
    true_labels = np.array(f['labels']['labels'][:]).flatten().astype(int)
    f.close()

    # 读取置信度
    pro_conf = np.loadtxt(PRO_CONF_PATH, delimiter=',')
    vis_conf = np.loadtxt(VIS_CONF_PATH, delimiter=',')

    # 确保长度一致
    min_len = min(len(true_labels), len(pro_conf), len(vis_conf))
    true_labels = true_labels[:min_len]
    pro_conf = pro_conf[:min_len]
    vis_conf = vis_conf[:min_len]

    print(f"样本数: {len(true_labels)}")
    print(f"类别数: 7")
    print(f"机体置信度形状: {pro_conf.shape}")
    print(f"视觉置信度形状: {vis_conf.shape}")


    # ============================================================
    # 方法1: 原始贝叶斯融合
    # ============================================================
    print("\n" + "="*70)
    print("方法1: 原始贝叶斯融合")
    print("="*70)

    p_a = np.ones(7) / 7
    pro_exp = np.exp(pro_conf / 0.3)
    min_val, max_val = np.min(pro_exp), np.max(pro_exp)
    normalized_data = (pro_exp - min_val) / (max_val - min_val)
    result_bayes = normalized_data * vis_conf * p_a
    pred_bayes = np.argmax(result_bayes, axis=1)
    acc_bayes = np.mean(pred_bayes == true_labels) * 100

    print(f"准确率: {acc_bayes:.2f}%")


    # ============================================================
    # 方法2: TMC 证据融合
    # ============================================================
    print("\n" + "="*70)
    print("方法2: TMC 证据融合 (Dempster-Shafer)")
    print("="*70)

    tmc = TMCFusion(num_classes=7)

    # 转换为证据
    print("\n[Step 1] 将置信度转换为证据...")
    pro_evidence = tmc.confidence_to_evidence(pro_conf, conf_type='cosine', temperature=0.3, scale=10.0)
    vis_evidence = tmc.confidence_to_evidence(vis_conf, conf_type='softmax', temperature=1.0, scale=10.0)

    print(f"  机体证据范围: [{pro_evidence.min():.2f}, {pro_evidence.max():.2f}]")
    print(f"  视觉证据范围: [{vis_evidence.min():.2f}, {vis_evidence.max():.2f}]")

    # 转换为主观意见
    print("\n[Step 2] 转换为主观意见 (信念 + 不确定性)...")
    pro_belief, pro_uncertainty = tmc.evidence_to_opinion(pro_evidence)
    vis_belief, vis_uncertainty = tmc.evidence_to_opinion(vis_evidence)

    print(f"  机体网络 - 平均不确定性: {pro_uncertainty.mean():.4f}, 标准差: {pro_uncertainty.std():.4f}")
    print(f"  视觉网络 - 平均不确定性: {vis_uncertainty.mean():.4f}, 标准差: {vis_uncertainty.std():.4f}")

    # DS 融合
    print("\n[Step 3] Dempster-Shafer 组合融合...")
    fused_belief, fused_uncertainty = tmc.ds_combine(
        pro_belief, pro_uncertainty,
        vis_belief, vis_uncertainty
    )

    print(f"  融合后平均不确定性: {fused_uncertainty.mean():.4f}")

    # 预测
    print("\n[Step 4] 预测...")
    pred_tmc, conf_tmc = tmc.predict(fused_belief, fused_uncertainty, method='belief')
    acc_tmc = np.mean(pred_tmc == true_labels) * 100

    print(f"  TMC 准确率: {acc_tmc:.2f}%")


    # ============================================================
    # 结果对比
    # ============================================================
    print("\n" + "="*70)
    print("融合方法对比汇总")
    print("="*70)
    print(f"{'方法':<30} {'准确率':<15} {'说明'}")
    print("-"*70)
    
    pro_only_acc = np.mean(np.argmax(pro_conf, axis=1) == true_labels) * 100
    vis_only_acc = np.mean(np.argmax(vis_conf, axis=1) == true_labels) * 100
    
    print(f"{'ProNet (仅机体)':<30} {pro_only_acc:.2f}%{'':<10} 单模态")
    print(f"{'VisionNet (仅视觉)':<30} {vis_only_acc:.2f}%{'':<10} 单模态")
    print(f"{'原始贝叶斯融合':<30} {acc_bayes:.2f}%{'':<10} 简单相乘")
    print(f"{'TMC 证据融合':<30} {acc_tmc:.2f}%{'':<10} DS 组合规则")
    print("-"*70)
    print(f"{'提升幅度 (TMC vs Bayes)':<30} {acc_tmc - acc_bayes:+.2f}%")
    print(f"{'提升幅度 (TMC vs ProNet)':<30} {acc_tmc - pro_only_acc:+.2f}%")
    print("-"*70)


    # ============================================================
    # 不确定性分析
    # ============================================================
    results = analyze_uncertainty(pred_tmc, true_labels, fused_uncertainty)
    print_analysis_report(results)


    # ============================================================
    # 各类别分析
    # ============================================================
    print("\n" + "="*70)
    print("各类别准确率与不确定性")
    print("="*70)
    print(f"{'类别':<8} {'样本数':<10} {'Bayes':<12} {'TMC':<12} {'平均u'}")
    print("-"*70)

    for i in range(7):
        mask = (true_labels == i)
        if mask.sum() > 0:
            bayes_acc = np.mean(pred_bayes[mask] == true_labels[mask]) * 100
            tmc_acc = np.mean(pred_tmc[mask] == true_labels[mask]) * 100
            cls_u = fused_uncertainty[mask].mean()
            print(f"类别 {i:<4} {mask.sum():<10} {bayes_acc:<12.2f} {tmc_acc:<12.2f} {cls_u:.4f}")

    print("-"*70)


    # ============================================================
    # 模态不确定性对比
    # ============================================================
    print("\n" + "="*70)
    print("模态不确定性对比（验证 TMC 自适应性）")
    print("="*70)

    print("\n视觉网络不确定性分布:")
    vis_u_bins = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    for low, high in vis_u_bins:
        mask = (vis_uncertainty >= low) & (vis_uncertainty < high)
        print(f"  u ∈ [{low}, {high}): {mask.sum()} 样本 ({mask.sum()/len(vis_uncertainty)*100:.1f}%)")

    print("\n机体网络不确定性分布:")
    for low, high in vis_u_bins:
        mask = (pro_uncertainty >= low) & (pro_uncertainty < high)
        print(f"  u ∈ [{low}, {high}): {mask.sum()} 样本 ({mask.sum()/len(pro_uncertainty)*100:.1f}%)")


    # ============================================================
    # 完成
    # ============================================================
    print("\n" + "="*70)
    print("程序运行完成")
    print(f"结果已保存到: {LOG_FILE}")
    print("="*70)

except Exception as e:
    print("\n" + "="*70)
    print("错误信息")
    print("="*70)
    print(f"发生错误: {e}")
    import traceback
    traceback.print_exc()
    print("="*70)

finally:
    # 恢复标准输出并关闭日志文件
    sys.stdout = tee.terminal
    tee.close()
    print(f"\n日志已保存到: {LOG_FILE}")

