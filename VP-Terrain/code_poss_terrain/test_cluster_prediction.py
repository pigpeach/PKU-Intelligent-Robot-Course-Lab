# 用于训练，只使用机体数据训练，对应CNN based的模型
# 输出loss最小的模型.pth，不同epoch下的tsne降维特征图
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import h5py
import numpy as np
import torch
from model import CNN #从自定义的model.py文件中导入CNN模型
from torch.utils.data import TensorDataset
from torchvision.models import mobilenet_v2
from torchvision import transforms
from Config import config #从自定义的Config.py参数文件中插入
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from losses import SupConLoss
from tqdm import tqdm 
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import random
import time

#加载参数
config = config()

os.environ["CUDA_VISIBLE_DEVICES"] = str(config.pro_cuda_device) # 设置使用的GPU
print(torch.cuda.get_device_name(0)) # 查看使用的设备名称
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
train_loader = torch.utils.data.DataLoader(train_data, config.pro_batch_size, False)
valid_loader = torch.utils.data.DataLoader(valid_data, config.pro_batch_size, False)

# 7. 对于训练好的模型和数据集，进行半监督聚类和预测
def fit(model, train_loader, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ##### 先用训练好的模型在训练集上跑一遍，得到特征向量
    model.eval()
    with torch.no_grad():
        train_feature = [] 
        train_labels = []
        for data in tqdm(train_loader):
            x_train, i_train, y_train = data 
            train_labels.append(y_train)
            y_train_pred, out_flat= model(x_train) 
            train_flat = y_train_pred.detach().numpy()
            train_feature.append(train_flat)

        train_features = np.concatenate(train_feature)
        train_labels = np.concatenate(train_labels)

        # 读取和计算小部分的真实标签数据的类别中心
        # 在train_features和train_labels上随机选择alpha个样本，计算其类别中心

        random_indices = random.sample(range(len(train_features)), int(len(train_features)*config.alpha))
        selected_features = train_features[random_indices]
        selected_labels = train_labels[random_indices]
        # 将 feature 向量根据 label 分组
        groups = {}
        for i in set(selected_labels):
            groups[i] = []
        for i in range(len(selected_features)):
            groups[selected_labels[i]].append(selected_features[i])

        # 计算每个组向量的中心
        centers_set = {}
        for y_res, group in groups.items():
            centers_set[y_res] = np.mean(group, axis=0)
        # 按标签的顺序，将中心点存入一个list
        init_centers = []
        for y_res in range(7):
            init_centers.append(centers_set[y_res])
        # print(init_centers)

        # 聚类得到最终预测结果

        # 使用kmeans聚类
        kmeans = KMeans(n_clusters=7, init=init_centers)  
        labels = kmeans.fit_predict(train_features)
        print("Using KMeans")
        # 使用层次聚类
        # clustering = AgglomerativeClustering(n_clusters=7, linkage='ward')
        # labels = clustering.fit_predict(train_features)
        # print("Using AgglomerativeClustering")
        # 使用高斯混合模型
        # gmm = GaussianMixture(n_components=7, covariance_type='full')
        # labels = gmm.fit_predict(train_features)
        # print("Using GaussianMixture")

        new_lab = np.zeros([len(train_features),])
        # 把标签对齐
        for i in range(7): 
            new_lab[labels == i] = np.argmax(np.bincount(train_labels[labels == i]))
            print(i, np.argmax(np.bincount(train_labels[labels == i])))
            print(np.bincount(train_labels[labels == i]))
        # 最终的聚类中心
        centers = [] # 用于存放每个类别的中心点
        for label in set(new_lab):  
            cluster_samples = train_features[new_lab == label]  # 提取属于该类别的所有样本  
            print(len(cluster_samples))
            if len(cluster_samples) > 0:  # 确保类别簇不为空  
                centers.append(np.mean(cluster_samples, axis=0))  # 计算中心点  
        

    ##### 对于测试集数据，计算每个样本到每个类别中心的距离

    model.eval()
    with torch.no_grad():

        # 测量推断时间
        start_time = time.time()

        test_matrix = np.zeros([7,7])
        test_feature = [] 
        test_labels = []
        for data in tqdm(test_loader):
            x_test, i_test, y_test = data 
            test_labels.append(y_test)
            y_test_pred, out_flat= model(x_test) 
            test_flat = y_test_pred.detach().numpy()
            test_feature.append(test_flat)
            y_test_pred = y_test_pred.unsqueeze(1)

        result = np.concatenate(test_feature)
        y_res = np.concatenate(test_labels)
        # print(result.shape)

        sim2center = np.zeros([len(result),7])

        for i, center in enumerate(centers):  
            # print(f"类别簇{i}的中心：{center}")
            sim2center[:,i]= np.squeeze(cosine_similarity(result,[center]), axis=1) 
        
        # 基于置信度的预测结果
        max_sim2center = np.argmax(sim2center,axis=1)   # 每个样本的预测类别
        conf = np.max(sim2center, axis=1)               # 每个样本的最大置信度

        end_time = time.time()

        np.savetxt(config.save_folder + 'pro_dark_conf', sim2center, fmt='%.4f', delimiter=",")
        # # 基于距离的置信度，余弦相似度
        # sim2center = sim2center / 1.0
        # sim2center_tensor = torch.from_numpy(sim2center)
        # sim2center_tensor = F.normalize(sim2center_tensor, p=2, dim=1)
        # np.savetxt('./SSL_poss/dark_test_conf_norm',sim2center_tensor.detach().numpy(),fmt='%.4f',delimiter=",")

        # soft = torch.softmax(sim2center_tensor.to(device), dim=1)
        # soft_all = soft.cpu().detach().numpy()
        # pre = np.max(soft_all, axis=1)
        # np.savetxt('./SSL_poss/dark_test_conf_soft',pre,fmt='%.4f',delimiter=",")


        np.savetxt(config.save_folder + 'pro_dark_conf_max',conf,fmt='%.4f',delimiter=",")
        np.savetxt(config.save_folder + 'pro_dark_prediction',max_sim2center,fmt='%d',delimiter=",")
        
        # 算混淆矩阵
        test_matrix = confusion_matrix(y_res, max_sim2center ,labels=[0,1,2,3,4,5,6])
        print(test_matrix)
        test_matrix = torch.from_numpy(test_matrix).to(torch.float)
        test_pres = test_matrix.diag()/test_matrix.sum(dim=0)
        test_mpre = test_pres.mean()
        test_recalls = test_matrix.diag()/test_matrix.sum(dim=1)
        test_mrecall = test_recalls.mean()
        # print("valid_pres: ",test_pres)
        # print("valid_recalls: ",test_recalls)
        test_all_pre = test_matrix.diag().sum()/test_matrix.sum()
        print("valid_all_pre: ",test_all_pre.item())
        test_all_recall = test_matrix.diag().sum()/test_matrix.sum()
        print("valid_all_recall: ",test_all_recall.item())

        total_time = (end_time - start_time) * 1000
        print(f"Total time: {total_time:.2f} ms")
        average_time = total_time / len(result)
        print(f"Average time: {average_time:.2f} ms")

    return 0, 0 #输出每个epoch的loss用来绘图

# 主函数
model = CNN(config.feature_size, config.out_channels, config.output_size) 
model.load_state_dict(torch.load(config.pro_save_path)) 
epoch_loss, epoch_valid_loss = fit(model, train_loader, valid_loader)
print('Finished Training')

