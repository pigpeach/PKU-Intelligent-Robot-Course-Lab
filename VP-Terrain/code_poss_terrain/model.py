# 定义网络结构
import timm
import torch.nn as nn
from Config import config #从自定义的Config.py参数文件中插入
from torchsummary import summary
import torch.nn.functional as F
import torchvision.models as models
#加载参数
config = config()

# Proprioception Net的网络结构
class CNN(nn.Module):
    def __init__(self, feature_size, out_channels, output_size):
        super(CNN, self).__init__()

        self.model1 = nn.Sequential(
            nn.Conv1d(8, 64, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.MaxPool1d(2,stride=2),
            nn.Dropout(0.2),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(3,stride=3),
            nn.Dropout(0.4),
            nn.Flatten(),
        ) # 输出1024维向量

        self.model2 = nn.Sequential(
            nn.Linear(1024,256),
        ) # 输出256维向量

    def forward(self, x):
        x = x.transpose(1, 2) #转置，将第三维变成第二维度进行卷积操作
        x = self.model1(x)
        x = self.model2(x)
        x = F.normalize(x, p=2, dim=1)
        out_flat = x
        return x, out_flat
        # x是归一化后的输出，out_flat是未归一化的输出
        # x维度是[batch_size, 256]，out_flat维度是[batch_size, 256]
        # x是用于计算损失的，out_flat是用于特征可视化的

# Vision Net的网络结构
class VisionNet(nn.Module):
    def __init__(self):
        super(VisionNet, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)

        self.feature = nn.Sequential(
            nn.Linear(1000, 1280),
        )

        self.inv_head = nn.Sequential(
                            nn.Dropout(0.25),
                            nn.Linear(1280,32),
                            nn.Linear(32, 7)
                            )
        
    def forward(self, x):
        x = self.model(x)
        x = self.feature(x)
        out_flat = x
        x = self.inv_head(x)
        return x, out_flat
        # x是7分类输出，out_flat是1280维特征向量

# 使用resnet50作为backbone的Vision Net的网络结构
class VisionNet_resnet(nn.Module):
    def __init__(self):
        super(VisionNet_resnet, self).__init__()
        self.model = models.resnet50(pretrained=True)

        self.feature = nn.Sequential(
            nn.Linear(1000, 1280),
        )

        self.inv_head = nn.Sequential(
                            nn.Dropout(0.25),
                            nn.Linear(1280,32),
                            nn.Linear(32, 7)
                            )
        
    def forward(self, x):
        x = self.model(x)
        x = self.feature(x)
        out_flat = x
        x = self.inv_head(x)
        return x, out_flat
        # x是7分类输出，out_flat是1280维特征向量

# 使用vit_base_patch16_224作为backbone的Vision Net的网络结构
class VisionNet_vit(nn.Module):
    def __init__(self):
        super(VisionNet_vit, self).__init__()
        self.model = timm.create_model('vit_base_patch16_224', pretrained=True)

        self.feature = nn.Sequential(
            nn.Linear(1000, 1280),
        )

        self.inv_head = nn.Sequential(
                            nn.Dropout(0.25),
                            nn.Linear(1280,32),
                            nn.Linear(32, 7)
                            )
        
    def forward(self, x):
        x = self.model(x)
        x = self.feature(x)
        out_flat = x
        x = self.inv_head(x)
        return x, out_flat
        # x是7分类输出，out_flat是1280维特征向量

# 观察模型输入输出和参数量
# proprioception_net = CNN(config.feature_size, config.out_channels, config.output_size)
# proprioception_net = proprioception_net.cuda()
# vision_net = VisionNet()
# vision_net = vision_net.cuda()

# summary(model=proprioception_net, input_size=(100, 9), batch_size=1)
# summary(model=vision_net, input_size=(3, 224, 224), batch_size=1)

