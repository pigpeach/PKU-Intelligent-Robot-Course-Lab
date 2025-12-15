import numpy as np
import h5py


# ========== 路径配置（直接替换这些路径） ==========
# 数据文件路径
DATA_PATH = "D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\VP-Terrain\data\_dark_7_new.hdf5"

# 机体网络结果文件路径
PRO_PREDICTION_PATH = "D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\SSL_poss\mobilenet_1.0\pro_dark_prediction"
PRO_CONF_PATH = "D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\SSL_poss\mobilenet_1.0\pro_dark_conf"

# 视觉网络结果文件路径
VISION_PREDICTION_PATH = "D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\SSL_poss\mobilenet_1.0\_vision_dark_prediction"
VISION_CONF_PATH = "D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\SSL_poss\mobilenet_1.0\_vision_dark_conf"

# ========== 读取数据 ==========
# 读取真实标签
f = h5py.File(DATA_PATH, 'r')
true_labels = np.array(f['labels']['labels'][:]).flatten().astype(int)
f.close()

# 读取预测结果
pro_pred = np.loadtxt(PRO_PREDICTION_PATH, delimiter=',').astype(int).flatten()
vision_pred = np.loadtxt(VISION_PREDICTION_PATH, delimiter=',').astype(int).flatten()

# 读取置信度
pro_conf = np.loadtxt(PRO_CONF_PATH, delimiter=',')
vision_conf = np.loadtxt(VISION_CONF_PATH, delimiter=',')

# 对齐长度
min_len = min(len(true_labels), len(pro_pred), len(vision_pred))
true_labels = true_labels[:min_len]
pro_pred = pro_pred[:min_len]
vision_pred = vision_pred[:min_len]
pro_conf = pro_conf[:min_len]
vision_conf = vision_conf[:min_len]

# ========== 计算准确率 ==========
# 单模态准确率
pro_acc = np.mean(pro_pred == true_labels) * 100
vision_acc = np.mean(vision_pred == true_labels) * 100

# 贝叶斯融合
prior = np.ones(7) / 7

# 将余弦相似度转换为概率
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

pro_conf_prob = softmax(pro_conf)
fusion_scores = prior * pro_conf_prob * vision_conf
fusion_pred = np.argmax(fusion_scores, axis=1)
fusion_acc = np.mean(fusion_pred == true_labels) * 100

# ========== 输出结果 ==========
print(f"\n测试集样本数: {min_len}")
print("\n" + "="*60)
print("模型准确率:")
print("="*60)
print(f"机体感知网络 (ProNet):     {pro_acc:.2f}%")
print(f"视觉网络 (VisionNet):       {vision_acc:.2f}%")
print(f"融合网络 (TerraFusion):     {fusion_acc:.2f}%")
print("="*60)

# ========== 各类别准确率 ==========
print(f"\n各类别准确率:")
print("-"*60)
print(f"{'类别':<8} {'样本数':<10} {'ProNet':<12} {'VisionNet':<12} {'Fusion'}")
print("-"*60)

for i in range(7):
    mask = (true_labels == i)
    if mask.sum() > 0:
        pro_cls_acc = np.mean(pro_pred[mask] == true_labels[mask]) * 100
        vis_cls_acc = np.mean(vision_pred[mask] == true_labels[mask]) * 100
        fus_cls_acc = np.mean(fusion_pred[mask] == true_labels[mask]) * 100
        print(f"类别 {i:<4} {mask.sum():<10} {pro_cls_acc:<12.2f} {vis_cls_acc:<12.2f} {fus_cls_acc:.2f}")

print("-"*60)
print("\n程序运行完成")
