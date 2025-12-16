# 用于训练，只使用机体数据训练，对应CNN based的模型
# 输出loss最小的模型.pth，不同epoch下的tsne降维特征图
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import CNN #从自定义的model.py文件中导入CNN模型
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import TensorDataset
from torchvision.models import mobilenet_v2
from torchvision import transforms
from Config import config #从自定义的Config.py参数文件中插入
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans
from losses import SupConLoss
from tqdm import tqdm 

#加载参数
#config = config()
#if not os.path.exists(config.save_folder):
    #os.mkdir(config.save_folder)
config = config()
if not os.path.exists(config.save_folder):
    os.makedirs(config.save_folder, exist_ok=True)  # 使用 makedirs

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.pro_cuda_device) # 设置使用的GPU
print(torch.cuda.get_device_name(0)) # 查看使用的设备名称KKKKKKKKKK
print(torch.cuda.is_available())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 这里不涉及标签比例的问题，全读进来就行
def read_data(filename):
    f = h5py.File(filename, 'r')
    print(list(f.keys()))

    timeStamps = f['timeStamps']['timeStamps'][:]
    signals = f['signals']['signals'][:]
    images = f['images']['images'][:]
    labels = f['labels']['labels'][:]

    print(timeStamps.shape,"# of timeStamps")
    print(signals.shape,"# of signals")
    print(images.shape,"# of images")
    print(labels.shape,"# of labels")

    f.close()
    images = images.reshape(images.shape[0], 224, 224, 3)/255.0 # normalize to 0-1
    return [timeStamps, signals, images, labels]


# 1.加载数据集
train_timeStamps, train_signals, train_images, train_labels = read_data(config.train_filename)
valid_timeStamps, valid_signals, valid_images, valid_labels = read_data(config.valid_filename)

# new_lab.txt就是用视觉数据跑出来的伪标签
# train_labels = np.loadtxt('./new_lab.txt',delimiter=",")

# 2.数据reshape
train_signals = train_signals.reshape(train_signals.shape[0],100,8)
valid_signals = valid_signals.reshape(valid_signals.shape[0],100,8)

train_signals[:,:,7]-=1
valid_signals[:,:,7]-=1

train_signals[:,:,0]/=2000.0
valid_signals[:,:,0]/=2000.0

train_signals[:,:,1]/=2000.0
valid_signals[:,:,1]/=2000.0

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
y_train_tensor = torch.from_numpy(train_labels).to(torch.long) 
x_valid_tensor = torch.from_numpy(valid_signals).to(torch.float)
y_valid_tensor = torch.from_numpy(valid_labels).to(torch.long)

# 5.形成训练数据集
train_data = CustomDataset(x_train_tensor, train_images, y_train_tensor, transform=transform)
valid_data = CustomDataset(x_valid_tensor, valid_images, y_valid_tensor, transform=transform)

# 6.将数据加载成迭代器
train_loader = torch.utils.data.DataLoader(train_data, config.pro_batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, config.pro_batch_size, False)

model_vision = mobilenet_v2(pretrained=True).to(DEVICE)
# 将训练集的数据通过视觉网络得到特征
train_features = []
# train_labels = []
model_vision.eval()
with torch.no_grad():
    for data in tqdm(train_loader):
        x_train, i_train, y_train = data
        i_train = i_train.to(DEVICE)
        extracted_feature = model_vision(i_train)
        train_features.append(extracted_feature)
        # train_labels.append(y_train)

train_features = torch.cat(train_features, dim=0).cpu().numpy()
# train_labels = torch.cat(train_labels, dim=0)

print(train_features.shape)

# 聚类得到伪标签
kmeans = KMeans(n_clusters=7, random_state=0).fit(train_features)
train_labels = kmeans.labels_

# 保存伪标签
np.savetxt(config.save_folder + 'new_lab.txt', train_labels, delimiter=',', fmt='%d')

# 使用伪标签生成一个新的数据集
# train_data = CustomDataset(x_train_tensor, train_images, torch.from_numpy(train_labels).to(torch.long), transform=transform)
# train_loader = torch.utils.data.DataLoader(train_data, config.batch_size, False)

# 用真实标签，对得到的特征进行降维可视化
# tsne = TSNE(n_components=2, learning_rate=100).fit_transform(train_features)
# plt.figure(figsize=(12, 6))
# plt.scatter(tsne[:, 0], tsne[:, 1], c=train_labels)
# plt.colorbar()
# plt.savefig("./SSL_poss/extracted_feature_from_vision.jpg")  

# 7.训练模型
def fit(epoch, model, loss_s, optimizer, train_loader, test_loader):
    
    model.train()
    running_loss = 0
    train_bar = tqdm(train_loader)  # 形成进度条
    train_feature = [] 
    labelss = []

    for data in train_bar:
        x_train, i_train, y_train = data  # 取出数据中的X和Y
        labelss.append(y_train)
        optimizer.zero_grad() #梯度初始化为零
        y_train_pred, flat = model(x_train) #前向传播求出预测的值
        np_flat = y_train_pred.detach().numpy()
        train_feature.append(np_flat)
        y_train_pred_lo = y_train_pred.unsqueeze(1)
        # print(f"y_train_pred_lo={y_train_pred_lo.shape}, y_train={y_train.shape}")
        loss = loss_s(y_train_pred_lo, y_train)
        # loss = loss_s(y_train_pred_lo)
        loss.backward()# 反向传播求梯度
        optimizer.step()# 更新所有参数
        running_loss += loss.item() #一个epochs里的每次的batchs的loss加起来
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, config.pro_epochs, loss)

# 检查train_feature中是否有nan
    for i in range(len(train_feature)):
        if np.isnan(train_feature[i]).any():
            print("nan in train_feature")
            break
        
    # a = 1/0

    # if epoch % 20 == 0:
    #     result = np.concatenate(train_feature)
    #     print(result.shape)
    #     y_res = np.concatenate(labelss)

    #     tsne = TSNE(n_components=2, learning_rate=100).fit_transform(result)
    #     plt.figure(figsize=(12, 6))
    #     plt.scatter(tsne[:, 0], tsne[:, 1], c=y_res)
    #     plt.colorbar()
    #     plt.savefig("./SSL_poss/pro_con/{}_epoch.jpg".format(epoch))
        # torch.save(model.state_dict(), './SSL_poss/pro_con/best_train_{}.pth'.format(epoch))

    epoch_loss = running_loss / len(train_loader.dataset)
    if epoch_loss < config.best_loss: # 保存loss最小的模型
        config.best_loss = epoch_loss
        torch.save(model.state_dict(), config.pro_save_path)

    # model.eval()
    # test_running_loss = 0
    # with torch.no_grad(): #验证过程不需要计算梯度
    #     test_bar = tqdm(test_loader)# 形成进度条
    #     test_matrix = np.zeros([7,7])
    #     test_feature = [] 
    #     test_labels = []
    #     for data in test_bar:
    #         x_test, i_test, y_test = data #取出数据中的X和Y
    #         test_labels.append(y_test)
    #         y_test_pred,out_flat= model(x_test) #求出预测的值
    #         test_flat = y_test_pred.detach().numpy()
    #         test_feature.append(test_flat)
    #         y_test_pred = y_test_pred.unsqueeze(1)
        
    #     # 对比学习只是为了得到模型，并不得到最终的结果

    # epoch_test_loss = test_running_loss / len(test_loader.dataset) #一个epochs训练完后，把累加的loss除以batch的数量，得到这个epochs的损失
    # if epoch_test_loss < config.best_loss: #保存验证集最优模型
    #     config.best_loss = epoch_test_loss
    #     torch.save(model.state_dict(), config.save_path)
    # if epoch_loss < config.best_loss: #保存验证集最优模型
    #     config.best_loss = epoch_loss
    #     torch.save(model.state_dict(), './best_dark_test_256_sup_cl.pth')

    return 0, 0 #输出每个epoch的loss用来绘图

# 从这里开始使用
model = CNN(config.feature_size, config.out_channels, config.output_size)  # 定义MLP网络
# model.load_state_dict(torch.load('./pth_256_sup_cl/best_train_300.pth')) # 导入网络的参数，训练时注释掉

loss_s = SupConLoss()

model_parameters = list(model.parameters()) 
optimizer = torch.optim.AdamW(model_parameters, lr=config.learning_rate)  # 定义优化器

# 8.开始训练
train_loss = []
valid_loss = []
for epoch in range(config.pro_epochs):
    epoch_loss, epoch_valid_loss = fit(epoch, model, loss_s, optimizer, train_loader, valid_loader)
print('Finished Training')


