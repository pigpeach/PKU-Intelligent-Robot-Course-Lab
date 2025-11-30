# 用于测试，只使用视觉数据
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from model import VisionNet, VisionNet_vit, VisionNet_resnet   #从自定义的model.py文件中导入CNN模型
from torch.utils.data import TensorDataset
from torchvision import transforms
from Config import config #从自定义的Config.py参数文件中插入
from tqdm import tqdm 
import torch.nn.functional as F
import time

#加载参数
config = config()

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.vision_cuda_device) # 设置使用的GPU
print(torch.cuda.get_device_name(0)) # 查看使用的设备名称
print(torch.cuda.is_available())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_data(filename):
    f = h5py.File(filename, 'r')
    print(list(f.keys()))

    timeStamps = f['timeStamps']['timeStamps'][:]
    signals = f['signals']['signals'][:]
    images = f['images']['images'][:]
    labels = f['labels']['labels'][:]

    f.close()
    images = images.reshape(images.shape[0], 224, 224, 3)/255.0 # normalize to 0-1
    return [timeStamps, signals, images, labels]


# 1.加载数据集
train_timeStamps, train_signals, train_images, train_labels = read_data(config.train_filename)
test_timeStamps, test_signals, test_images, test_labels = read_data(config.valid_filename)

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]) # normilize helps a little
    ])

class CustomDataset(TensorDataset):  
    def __init__(self, seq_data, img_data, label_data, transform=None):  
        self.seq_data = seq_data  
        self.img_data = img_data  
        self.label_data = label_data  
        self.transform = transform
  
    def __len__(self):  
        # 返回数据集中的样本数量  
        return len(self.seq_data)  
  
    def __getitem__(self, idx):  
        # 返回索引为 idx 的样本  
        sig = self.seq_data[idx]  # 集体数据
        img = self.img_data[idx]  # 图片数据  
        label = self.label_data[idx]  # 标签数据  
        img = self.transform(img)
        return sig, img, label

print(train_timeStamps.shape,"# of train_timeStamps")
print(train_signals.shape,"# of train_signals")
print(train_labels.shape,"# of train_labels")
print(test_timeStamps.shape,"# of test_timeStamps")
print(test_signals.shape,"# of test_signals")
print(test_labels.shape,"# of test_labels")

# 3.将数据转为tensor
x_train_tensor = torch.from_numpy(train_signals).to(torch.float)
# i_train_tensor = torch.from_numpy(train_images).to(torch.float)
y_train_tensor = torch.from_numpy(train_labels).to(torch.long)  # 伪标签
x_test_tensor = torch.from_numpy(test_signals).to(torch.float)
# i_test_tensor = torch.from_numpy(test_images).to(torch.float)
y_test_tensor = torch.from_numpy(test_labels).to(torch.long)

# 5.形成测试数据集
test_data = CustomDataset(x_test_tensor, test_images, y_test_tensor, transform=transform)

# 6.将数据加载成迭代器
test_loader = torch.utils.data.DataLoader(test_data, config.vision_batch_size, False)

# 从这里开始使用
if config.vision_backbone == 'mobilenet':
    model = VisionNet()  # 定义MLP网络
elif config.vision_backbone == 'resnet':
    model = VisionNet_resnet()
elif config.vision_backbone == 'vit':
    model = VisionNet_vit()
model.load_state_dict(torch.load(config.vision_save_path))
model.to(DEVICE)

# 用于测试，只使用视觉数据
model.eval()
with torch.no_grad():

    total_time = 0
    test_correct = 0
    test_total = 0
    conf = np.array([])

    for i, (_, imgs, labels) in enumerate(tqdm(test_loader)):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        start_time = time.time()

        outputs, vision_features = model(imgs)
        _, predicted = outputs.max(1)
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()

        end_time = time.time()
        total_time += (end_time - start_time) * 1000

        softmax_scores = F.softmax(outputs/400.0, dim=1)  # Apply softmax to get the probability distribution
        conf = np.concatenate((conf, softmax_scores.cpu().detach().numpy())) if conf.size else softmax_scores.cpu().detach().numpy()

    test_acc = 100. * test_correct / test_total
    print('Vision Acc: %.2f' % (test_acc))

    # total_time = (end_time - start_time) * 1000
    print('Vision Time: %.2f ms' % (total_time))
    average_time = total_time / len(test_labels)
    print('Vision Average Time: %.2f ms' % (average_time))

    np.savetxt(config.save_folder + 'vision_dark_conf', conf, fmt='%.4f',delimiter=",")
    max_conf = np.max(conf, axis=1)
    np.savetxt(config.save_folder + 'vision_dark_conf_max',max_conf, fmt='%.4f',delimiter=",")
    prediction = np.argmax(conf, axis=1)
    np.savetxt(config.save_folder + 'vision_dark_prediction',prediction, fmt='%d',delimiter=",") 


# 用于计算最终基于贝叶斯决策融合的分类结果
pro_conf = np.loadtxt(config.save_folder + 'pro_dark_conf',delimiter=",")
vis_conf = np.loadtxt(config.save_folder + 'vision_dark_conf',delimiter=",")

start_time = time.time()

p_a = np.array([[1, 1, 1, 1, 1, 1, 1]])
p_a = p_a / 7

pro_exp_cos_dis = np.exp(pro_conf/0.3)

# 计算最小值和最大值
min_value = np.min(pro_exp_cos_dis)
max_value = np.max(pro_exp_cos_dis)

# 归一化到 (0, 1) 之间
normalized_data = (pro_exp_cos_dis - min_value) / (max_value - min_value)
max = np.max(normalized_data, axis=1)
pre_ = np.argmax(normalized_data, axis=1)

# np.savetxt('./cpv/test_pro_exp_cos_dis_0.3_norm',normalized_data,fmt='%.4f',delimiter=",")
# np.savetxt('./cpv/test_pro_exp_cos_dis_0.3_norm_max',max,fmt='%.4f',delimiter=",")
# np.savetxt('./cpv/test_pro_exp_cos_dis_0.3_norm_pre',pre_,fmt='%d',delimiter=",")

result = normalized_data * vis_conf * p_a

a = np.argmax(result, axis=1)

fusion_acc = np.sum(a == test_labels) / len(test_labels) * 100.

print('Fusion Acc: %.3f' % (fusion_acc))

end_time = time.time()
total_time = (end_time - start_time) * 1000
print('Fusion Time: %.2f ms' % (total_time))
average_time = total_time / len(test_labels)
print('Fusion Average Time: %.2f ms' % (average_time))


# np.savetxt('./cpv/fusion_test_conf_exp_0.3_norm',result,fmt='%.4f',delimiter=",")
# np.savetxt('./cpv/fusion_test_pre_exp_0.3_norm',a,fmt='%d',delimiter=",")

# print("sss")