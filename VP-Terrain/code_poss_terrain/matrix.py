# 用于基于视觉和机体得到的融合结果生成融合模型的混淆矩阵
import torch
import numpy as np
from sklearn.metrics import confusion_matrix

y_pre=np.loadtxt('./cpv/fusion_dark_pre_exp_0.3_norm',delimiter=",")
y_res=np.loadtxt('./cpv/dark_y_res_8857.txt',delimiter=",")

test_matrix = np.zeros([7,7])

test_matrix = confusion_matrix(y_res, y_pre ,labels=[0,1,2,3,4,5,6])

test_matrix = torch.from_numpy(test_matrix).to(torch.float)
print(test_matrix)
test_pres = test_matrix.diag()/test_matrix.sum(dim=0)
test_mpre = test_pres.mean()
test_recalls = test_matrix.diag()/test_matrix.sum(dim=1)
test_mrecall = test_recalls.mean()
print("valid_pres: ",test_pres)
print("valid_recalls: ",test_recalls)
test_all_pre = test_matrix.diag().sum()/test_matrix.sum()
print("valid_all_pre: ",test_all_pre)