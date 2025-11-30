# 用于训练，只使用视觉数据
import os
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import VisionNet, VisionNet_resnet, VisionNet_vit   #从自定义的model.py文件中导入CNN模型
from torch.utils.data import TensorDataset
from torchvision import transforms
from Config import config #从自定义的Config.py参数文件中插入
from tqdm import tqdm 
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#加载参数
config = config()

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.vision_cuda_device) # 设置使用的GPU
print(torch.cuda.get_device_name(0)) # 查看使用的设备名称
print(torch.cuda.is_available())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_portion_data(filename, alpha):
    f = h5py.File(filename, 'r')
    print(list(f.keys()))

    timeStamps = f['timeStamps']['timeStamps'][:]
    signals = f['signals']['signals'][:]
    images = f['images']['images'][:]
    labels = f['labels']['labels'][:]

    # 随机选取alpha比例的数据
    random_indices = random.sample(range(len(timeStamps)), int(len(timeStamps) * alpha))
    timeStamps = np.array([timeStamps[i] for i in random_indices])
    signals = np.array([signals[i] for i in random_indices])
    images = np.array([images[i] for i in random_indices])
    labels = np.array([labels[i] for i in random_indices])

    print(f"Take {len(labels)} data from {filename} with alpha={alpha}")

    f.close()
    images = images.reshape(images.shape[0], 224, 224, 3)/255.0 # normalize to 0-1
    return [timeStamps, signals, images, labels]


# 1.加载数据集
train_timeStamps, train_signals, train_images, train_labels = read_portion_data(config.train_filename, config.alpha)
valid_timeStamps, valid_signals, valid_images, valid_labels = read_portion_data(config.valid_filename, 1.0)

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
print(valid_timeStamps.shape,"# of valid_timeStamps")
print(valid_signals.shape,"# of valid_signals")
print(valid_labels.shape,"# of valid_labels")

# 3.将数据转为tensor
x_train_tensor = torch.from_numpy(train_signals).to(torch.float)
# i_train_tensor = torch.from_numpy(train_images).to(torch.float)
y_train_tensor = torch.from_numpy(train_labels).to(torch.long)  # 伪标签
x_valid_tensor = torch.from_numpy(valid_signals).to(torch.float)
# i_valid_tensor = torch.from_numpy(valid_images).to(torch.float)
y_valid_tensor = torch.from_numpy(valid_labels).to(torch.long)

# 5.形成训练数据集
train_data = CustomDataset(x_train_tensor, train_images, y_train_tensor, transform=transform)
valid_data = CustomDataset(x_valid_tensor, valid_images, y_valid_tensor, transform=transform)

# 6.将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data, config.vision_batch_size, False)
valid_loader = torch.utils.data.DataLoader(valid_data, config.vision_batch_size, False)

# 从这里开始使用
# model = CNN(config.feature_size, config.out_channels, config.output_size)  # 定义MLP网络
# model.load_state_dict(torch.load('./pth_256_sup_cl/best_train_300.pth')) # 导入网络的参数，训练时注释掉
if config.vision_backbone == 'mobilenet':
    model = VisionNet()  # 定义MLP网络
elif config.vision_backbone == 'resnet':
    model = VisionNet_resnet()
elif config.vision_backbone == 'vit':
    model = VisionNet_vit()
model.to(DEVICE)

loss_s = nn.CrossEntropyLoss()  # 定义损失函数

model_parameters = list(model.parameters()) 
optimizer = torch.optim.AdamW(model_parameters, lr=config.learning_rate)  # 定义优化器

# 8.开始训练
for epoch in range(config.vision_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    train_feature = np.array([])
    labelss = np.array([])
    test_feature = np.array([])
    test_labels = np.array([])

    for i, (_, imgs, labels) in enumerate(tqdm(train_loader)):
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad()
        outputs, vision_features = model(imgs)

        train_feature = np.concatenate((train_feature, vision_features.cpu().detach().numpy())) if train_feature.size else vision_features.cpu().detach().numpy()
        labelss = np.concatenate((labelss, labels.cpu().detach().numpy())) if labelss.size else labels.cpu().detach().numpy()

        loss = loss_s(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    train_acc = 100. * train_correct / train_total
    train_loss /= len(train_loader)
    print('Epoch %d, Train Loss: %.3f, Train Acc: %.3f' % (epoch, train_loss, train_acc))

    model.eval()
    valid_correct = 0
    valid_total = 0
    with torch.no_grad():
        for i, (_, imgs, labels) in enumerate(valid_loader):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs, vision_features = model(imgs)

            test_feature = np.concatenate((test_feature, vision_features.cpu().detach().numpy())) if test_feature.size else vision_features.cpu().detach().numpy()
            test_labels = np.concatenate((test_labels, labels.cpu().detach().numpy())) if test_labels.size else labels.cpu().detach().numpy()

            _, predicted = outputs.max(1)
            valid_total += labels.size(0)
            valid_correct += predicted.eq(labels).sum().item()
        valid_acc = 100. * valid_correct / valid_total
        print('Epoch %d, Valid Acc: %.3f' % (epoch, valid_acc))
    if valid_acc > config.best_all_pre:
        config.best_all_pre = valid_acc
        torch.save(model.state_dict(), config.vision_save_path)
        print('Model Saved in %s' % config.vision_save_path)

    # random_indices = random.sample(range(len(train_feature)), 1000)
    # selected_features = [train_feature[i] for i in random_indices]
    # selected_labels = [labelss[i] for i in random_indices]
    # result = np.array(selected_features)
    # y_res = np.array(selected_labels)
    # print(result.shape, y_res.shape)
    # tsne = TSNE(n_components=2, learning_rate=100).fit_transform(result)
    # plt.figure(figsize=(12, 6))
    # plt.scatter(tsne[:, 0], tsne[:, 1], c=y_res)
    # plt.colorbar()
    # plt.savefig("./ACB/vision_train_feature/{}_epoch.jpg".format(epoch))

    # random_indices = random.sample(range(len(test_feature)), 1000)
    # selected_features = [test_feature[i] for i in random_indices]
    # selected_labels = [test_labels[i] for i in random_indices]
    # result = np.array(selected_features)
    # y_res = np.array(selected_labels)
    # print(result.shape, y_res.shape)
    # tsne = TSNE(n_components=2, learning_rate=100).fit_transform(result)
    # plt.figure(figsize=(12, 6))
    # plt.scatter(tsne[:, 0], tsne[:, 1], c=y_res)
    # plt.colorbar()
    # plt.savefig("./ACB/vision_test_feature/{}_epoch.jpg".format(epoch))

print('Best Valid Acc: %.3f' % config.best_all_pre)