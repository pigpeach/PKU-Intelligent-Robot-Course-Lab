# 原版loss，在test_wind_CNN.py中使用
"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.3, contrast_mode='all',
                 base_temperature=0.3):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        
        # print(f"0 mask={mask.sum(1)}")
        
        # 将mask写到文本文件中
        # mask = mask.cpu().detach().numpy()
        # mask = mask.astype(int)
        # mask = mask.tolist()
        # with open('./SSL/mask.txt', 'w') as f:
        #     for i in range(len(mask)):
        #         f.write(str(mask[i]) + '\n')
        #     f.close()

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # print(f"1 mask={mask.sum(1)}")

        # mask-out self-contrast cases
        # 把对角线上的值设置为0
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        # print(f"logits_mask={logits_mask}")

        mask = mask * logits_mask

        # print(f"anchor_dot_contrast={anchor_dot_contrast}, logits_max={logits_max}, logits={logits}, mask={mask}")

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # print(f"exp_logits={exp_logits}, log_prob={log_prob}")

        # print(f"mask={mask.sum(1)}")
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # print(f"mean_log_prob_pos={mean_log_prob_pos}")

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos

        # print(f"loss={loss}")

        loss = loss.view(anchor_count, batch_size).mean()

        # a = 1/0

        return loss


class ContrastiveLoss(nn.Module):
    """
    计算有监督对比学习损失

    Args:
        temperature: 温度参数，用于控制对比损失的平滑程度。

    Returns:
        损失值。
    """

    def __init__(self, temperature=1.0):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        计算损失

        Args:
            features: 模型输入的特征，形状应该是 [batch_size, num_views, ...]。
            labels: 类别标签，形状应该是 [batch_size, 1]。

        Returns:
            损失值。
        """

        batch_size = features.shape[0]
        anchor_count = features.shape[1]
        labels = labels.unsqueeze(1)
        # print(labels.shape)
        # 计算锚点特征和对比特征之间的相似度矩阵
        # print(features.shape)
        # print(features.T.shape)
        features = torch.cat(torch.unbind(features, dim=1), dim=0)
        # print(features.shape)
        # io=1/0
        anchor_dot_contrast = torch.matmul(features, features.T) / self.temperature

        # 为数值稳定性，减去最大值
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 计算log_prob
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.logsumexp(exp_logits, dim=1, keepdim=True)

        # 计算平均log-likelihood
        mean_log_prob_pos = (labels.float() * log_prob).sum(1) / labels.sum(1)

        # 计算损失
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss

    Args:
        temperature (float, optional): Temperature parameter, default: 0.07

    Example:
        >>> loss_fn = NTXentLoss()
        >>> z1, z2 = torch.randn(10, 128), torch.randn(10, 128)
        >>> loss = loss_fn(z1, z2)
    """

    def __init__(self, temperature=0.07):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Args:
            z1 (torch.Tensor): Embeddings of positive pairs, shape (batch_size, embedding_dim)
            z2 (torch.Tensor): Embeddings of negative pairs, shape (batch_size, embedding_dim)

        Returns:
            torch.Tensor: Loss, shape (batch_size,)
        """
        batch_size = z1.size(0)

        # Normalize embeddings
        z1 = z1 / torch.norm(z1, p=2, dim=1, keepdim=True)
        z2 = z2 / torch.norm(z2, p=2, dim=1, keepdim=True)

        # Compute logits
        logits = torch.mm(z1, z2.t())

        # Apply temperature scaling
        logits = logits / self.temperature

        # Compute cross entropy loss
        loss = -torch.mean(torch.nn.functional.log_softmax(logits, dim=1)[:, 0])

        return loss