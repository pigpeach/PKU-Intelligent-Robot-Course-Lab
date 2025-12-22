# TerraFusion 证据融合分析与 TMC 改进方案

## 目录

1. [当前代码的证据融合分析](#一当前代码的证据融合分析)
2. [TMC 论文的证据融合方法](#二tmc-论文的证据融合方法)
3. [代码修改方案](#三代码修改方案)
4. [完整实现代码](#四完整实现代码)

---

## 一、当前代码的证据融合分析

### 1.1 负责证据融合的代码位置

证据融合代码位于 `test_only_vision.py` 的 **第 127-152 行**：

```python
# 用于计算最终基于贝叶斯决策融合的分类结果
pro_conf = np.loadtxt(config.save_folder + 'pro_dark_conf', delimiter=",")
vis_conf = np.loadtxt(config.save_folder + 'vision_dark_conf', delimiter=",")

# 先验概率（均匀分布）
p_a = np.array([[1, 1, 1, 1, 1, 1, 1]])
p_a = p_a / 7

# 温度缩放
pro_exp_cos_dis = np.exp(pro_conf / 0.3)

# 全局 Min-Max 归一化
min_value = np.min(pro_exp_cos_dis)
max_value = np.max(pro_exp_cos_dis)
normalized_data = (pro_exp_cos_dis - min_value) / (max_value - min_value)

# 贝叶斯融合（简单相乘）
result = normalized_data * vis_conf * p_a

# 预测
a = np.argmax(result, axis=1)
fusion_acc = np.sum(a == test_labels) / len(test_labels) * 100.
```

### 1.2 当前融合方法的数学表达

当前方法的融合公式：

$$
P(c|I,S) = P(c) \times \text{conf}^{vis}(c) \times \text{norm}(\exp(\text{conf}^{pro}(c) / T))
$$

其中：
- $P(c) = 1/7$（均匀先验）
- $\text{conf}^{vis}(c)$ 是视觉网络的 softmax 输出
- $\text{conf}^{pro}(c)$ 是机体网络的余弦相似度
- $T = 0.3$ 是温度参数
- $\text{norm}(\cdot)$ 是全局 Min-Max 归一化

### 1.3 当前方法的问题分析

| 问题 | 描述 | 影响 |
|------|------|------|
| **过于简单** | 仅使用概率相乘，没有不确定性建模 | 无法区分"确定的错误"和"不确定的正确" |
| **缺乏理论支撑** | 不是严格的贝叶斯推断 | 融合结果可能不合理 |
| **无冲突处理** | 两模态预测不一致时无处理机制 | 可能产生错误融合 |
| **不能感知模态质量** | 无法动态评估模态可靠性 | 失效模态仍参与决策 |
| **归一化不合理** | 全局 Min-Max 破坏概率分布意义 | 不同样本间不可比 |
| **无不确定性输出** | 只输出预测类别 | 无法进行可信决策 |

### 1.4 问题示例

**场景：Dark 条件下视觉网络失效**

```
视觉置信度: [0.143, 0.142, 0.143, 0.143, 0.142, 0.143, 0.144]  ← 几乎均匀
机体置信度: [0.98, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00]        ← 明确预测类别0

当前融合: result = [0.14, 0.003, 0, 0, 0, 0, 0]  ← 机体主导（偶然正确）
```

**问题**：当前方法**无法显式感知**视觉网络已失效，只是因为数值上视觉置信度接近均匀才"碰巧"让机体网络主导。

---

## 二、TMC 论文的证据融合方法

### 2.1 核心思想

TMC (Trusted Multi-view Classification) 提出基于 **Dempster-Shafer 证据理论** 的融合方法：

```
┌─────────────────────────────────────────────────────────────────┐
│                        TMC 融合流程                              │
└─────────────────────────────────────────────────────────────────┘

     网络输出 o                网络输出 o
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│  View 1 (视觉)   │       │  View 2 (机体)   │
│  α = σ(o) + 1   │       │  α = σ(o) + 1   │
└────────┬────────┘       └────────┬────────┘
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│ Dirichlet 分布   │       │ Dirichlet 分布   │
│ Dir(π|α)        │       │ Dir(π|α)        │
└────────┬────────┘       └────────┬────────┘
         │                         │
         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐
│ 主观意见         │       │ 主观意见         │
│ M₁ = {b₁, u₁}   │       │ M₂ = {b₂, u₂}   │
└────────┬────────┘       └────────┬────────┘
         │                         │
         └──────────┬──────────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │   DS 组合规则        │
         │   M = M₁ ⊕ M₂       │
         └──────────┬──────────┘
                    │
                    ▼
         ┌─────────────────────┐
         │  融合意见 M = {b, u} │
         │  预测 + 不确定性     │
         └─────────────────────┘
```

### 2.2 关键数学公式

#### 2.2.1 从网络输出到 Dirichlet 分布参数

$$
\alpha_k = \sigma(o_k) + 1
$$

其中：
- $o_k$ 是网络对类别 $k$ 的原始输出
- $\sigma(\cdot)$ 是非负激活函数（ReLU 或 Softplus）
- $\alpha_k \geq 1$ 是 Dirichlet 分布的集中参数

#### 2.2.2 从 Dirichlet 参数到证据和主观意见

**证据 (Evidence)**:
$$
e_k = \alpha_k - 1
$$

**Dirichlet 强度 (Strength)**:
$$
S = \sum_{k=1}^{K} \alpha_k = \sum_{k=1}^{K} (e_k + 1)
$$

**信念质量 (Belief Mass)**:
$$
b_k = \frac{e_k}{S} = \frac{\alpha_k - 1}{S}
$$

**不确定性 (Uncertainty)**:
$$
u = \frac{K}{S}
$$

**约束条件**:
$$
u + \sum_{k=1}^{K} b_k = 1
$$

#### 2.2.3 Dempster-Shafer 组合规则

给定两个视角的主观意见：
- $M_1 = \{b_k^1\}_{k=1}^K, u_1$
- $M_2 = \{b_k^2\}_{k=1}^K, u_2$

**冲突因子**:
$$
C = \sum_{i \neq j} b_i^1 \cdot b_j^2
$$

**融合信念**:
$$
b_k = \frac{1}{1-C} (b_k^1 b_k^2 + b_k^1 u_2 + b_k^2 u_1)
$$

**融合不确定性**:
$$
u = \frac{1}{1-C} u_1 u_2
$$

### 2.3 TMC 方法的优势

| 特性 | 当前 TerraFusion | TMC 方法 |
|------|------------------|----------|
| **不确定性建模** | ❌ 无 | ✅ 显式建模 $u$ |
| **冲突处理** | ❌ 无 | ✅ 冲突因子 $C$ |
| **自适应融合** | ⚠️ 隐式（数值偶然） | ✅ 显式（不确定视角权重降低） |
| **可信决策** | ❌ 不支持 | ✅ 可按 $u$ 阈值过滤 |
| **理论保证** | ⚠️ 简单贝叶斯 | ✅ DS 证据理论 + 4个命题 |
| **可解释性** | ❌ 仅输出预测 | ✅ 输出信念 + 不确定性 |

### 2.4 TMC 的四个理论保证（论文 Proposition 3.1-3.4）

| 命题 | 内容 | 意义 |
|------|------|------|
| **Prop 3.1** | 若新视角对正确类别的信念更高，融合后正确类别信念增加 | 融合可能提升准确率 |
| **Prop 3.2** | 当新视角不确定性大时，性能下降有限；若 $u_a=1$，则 $b_t = b_t^o$ | 不确定视角不会破坏原有决策 |
| **Prop 3.3** | 融合后不确定性降低：$u \leq u_o$ | 多视角融合增加确定性 |
| **Prop 3.4** | $u$ 与 $u_1, u_2$ 正相关 | 两个都不确定时，融合仍不确定 |

---

## 三、代码修改方案

### 3.1 修改位置总览

```
SSL_poss/
├── evidence_fusion.py          ← 【新增】TMC 融合模块
├── test_only_vision.py         ← 【修改】添加 TMC 融合调用
├── test_tmc_fusion.py          ← 【新增】独立的 TMC 测试脚本
└── Config.py                   ← 【可选修改】添加 TMC 相关参数
```

### 3.2 核心修改：`test_only_vision.py`

#### 原代码（第 127-152 行）

```python
# 用于计算最终基于贝叶斯决策融合的分类结果
pro_conf = np.loadtxt(config.save_folder + 'pro_dark_conf', delimiter=",")
vis_conf = np.loadtxt(config.save_folder + 'vision_dark_conf', delimiter=",")

p_a = np.array([[1, 1, 1, 1, 1, 1, 1]])
p_a = p_a / 7

pro_exp_cos_dis = np.exp(pro_conf / 0.3)
min_value = np.min(pro_exp_cos_dis)
max_value = np.max(pro_exp_cos_dis)
normalized_data = (pro_exp_cos_dis - min_value) / (max_value - min_value)

result = normalized_data * vis_conf * p_a
a = np.argmax(result, axis=1)
fusion_acc = np.sum(a == test_labels) / len(test_labels) * 100.
print('Fusion Acc: %.5f' % (fusion_acc))
```

#### 修改后代码

```python
# ============================================================
# 方法1: 原始贝叶斯融合（保留）
# ============================================================
pro_conf = np.loadtxt(config.save_folder + 'pro_dark_conf', delimiter=",")
vis_conf = np.loadtxt(config.save_folder + 'vision_dark_conf', delimiter=",")

p_a = np.array([[1, 1, 1, 1, 1, 1, 1]])
p_a = p_a / 7

pro_exp_cos_dis = np.exp(pro_conf / 0.3)
min_value = np.min(pro_exp_cos_dis)
max_value = np.max(pro_exp_cos_dis)
normalized_data = (pro_exp_cos_dis - min_value) / (max_value - min_value)

result = normalized_data * vis_conf * p_a
a = np.argmax(result, axis=1)
fusion_acc = np.sum(a == test_labels) / len(test_labels) * 100.
print('Bayes Fusion Acc: %.2f%%' % (fusion_acc))

# ============================================================
# 方法2: TMC 证据融合（新增）
# ============================================================
from evidence_fusion import TMCFusion

tmc = TMCFusion(num_classes=7)

# 将置信度转换为证据
pro_evidence = tmc.confidence_to_evidence(pro_conf, conf_type='cosine', temperature=0.3)
vis_evidence = tmc.confidence_to_evidence(vis_conf, conf_type='softmax', temperature=1.0)

# 转换为主观意见
pro_belief, pro_uncertainty = tmc.evidence_to_opinion(pro_evidence)
vis_belief, vis_uncertainty = tmc.evidence_to_opinion(vis_evidence)

# DS 组合融合
fused_belief, fused_uncertainty = tmc.ds_combine(
    pro_belief, pro_uncertainty,
    vis_belief, vis_uncertainty
)

# 预测
tmc_pred = np.argmax(fused_belief, axis=1)
tmc_acc = np.sum(tmc_pred == test_labels) / len(test_labels) * 100.

print('TMC Fusion Acc: %.2f%%' % (tmc_acc))
print('平均不确定性: %.4f' % (fused_uncertainty.mean()))

# 可信决策分析
confident_mask = (fused_uncertainty < 0.3)
if confident_mask.sum() > 0:
    confident_acc = np.sum(tmc_pred[confident_mask] == test_labels[confident_mask]) / confident_mask.sum() * 100.
    print('高置信样本准确率 (u<0.3): %.2f%% (%d/%d)' % (confident_acc, confident_mask.sum(), len(test_labels)))
```

---

## 四、完整实现代码

### 4.1 `evidence_fusion.py` - TMC 融合模块

```python
"""
TMC (Trusted Multi-view Classification) 证据融合模块
基于 Dempster-Shafer 证据理论

参考论文: Han et al., "Trusted Multi-View Classification", TPAMI 2022
论文链接: https://arxiv.org/abs/2204.11423
官方代码: https://github.com/hanmenghan/TMC
"""

import numpy as np
from typing import Tuple, List, Optional


class TMCFusion:
    """
    TMC 可信多视角分类融合器
    
    核心思想:
    1. 用 Dirichlet 分布建模类别概率的不确定性
    2. 将 Dirichlet 参数转换为信念(belief)和不确定性(uncertainty)
    3. 使用 Dempster-Shafer 组合规则融合多个视角
    
    关键公式:
    - 证据: e_k = α_k - 1
    - Dirichlet强度: S = Σα_k
    - 信念: b_k = e_k / S
    - 不确定性: u = K / S
    - 约束: u + Σb_k = 1
    """
    
    def __init__(self, num_classes: int = 7):
        """
        初始化融合器
        
        Args:
            num_classes: 类别数量 K
        """
        self.K = num_classes
    
    def confidence_to_evidence(
        self, 
        conf: np.ndarray, 
        conf_type: str = 'softmax',
        temperature: float = 1.0,
        scale: float = 10.0
    ) -> np.ndarray:
        """
        将现有置信度转换为证据
        
        这是一个适配函数，用于将 TerraFusion 的置信度输出
        转换为 TMC 所需的证据格式
        
        Args:
            conf: 置信度矩阵 (N, K)
            conf_type: 置信度类型
                - 'softmax': 视觉网络的 softmax 输出 [0, 1]
                - 'cosine': 机体网络的余弦相似度 [-1, 1]
            temperature: 温度参数
            scale: 缩放因子，控制证据强度
        
        Returns:
            evidence: 证据矩阵 (N, K)，非负值
        """
        if conf_type == 'cosine':
            # 余弦相似度 → 证据
            # 1. 温度缩放
            scaled = conf / temperature
            # 2. 指数变换（类似 softmax 但不归一化）
            exp_scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
            # 3. 缩放到合适范围
            evidence = exp_scaled * scale
            
        elif conf_type == 'softmax':
            # Softmax 概率 → 证据
            # 使用逆 softmax 思想：p_k ∝ exp(e_k)
            # 所以 e_k ∝ log(p_k)
            log_conf = np.log(np.clip(conf, 1e-10, 1.0))
            # 平移使最小值为0
            log_conf = log_conf - np.min(log_conf, axis=1, keepdims=True)
            # 缩放
            evidence = log_conf * scale
            
        else:
            raise ValueError(f"Unknown conf_type: {conf_type}")
        
        # 确保证据非负
        evidence = np.maximum(evidence, 0)
        
        return evidence
    
    def evidence_to_opinion(
        self, 
        evidence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        将证据转换为主观意见（信念 + 不确定性）
        
        对应论文 Eq. (11):
        - b_k = e_k / S
        - u = K / S
        其中 S = Σ(e_k + 1) = Σα_k
        
        Args:
            evidence: 证据矩阵 (N, K)
        
        Returns:
            belief: 信念质量 (N, K)
            uncertainty: 不确定性 (N,)
        """
        # Dirichlet 参数: α_k = e_k + 1
        alpha = evidence + 1
        
        # Dirichlet 强度: S = Σα_k
        S = np.sum(alpha, axis=1, keepdims=True)  # (N, 1)
        
        # 信念质量: b_k = e_k / S = (α_k - 1) / S
        belief = evidence / S  # (N, K)
        
        # 不确定性: u = K / S
        uncertainty = (self.K / S).squeeze()  # (N,)
        
        return belief, uncertainty
    
    def opinion_to_evidence(
        self, 
        belief: np.ndarray, 
        uncertainty: np.ndarray
    ) -> np.ndarray:
        """
        将主观意见转换回证据（逆变换）
        
        对应论文 Eq. (15):
        - S = K / u
        - e_k = b_k × S
        
        Args:
            belief: 信念质量 (N, K)
            uncertainty: 不确定性 (N,)
        
        Returns:
            evidence: 证据矩阵 (N, K)
        """
        # 防止除零
        u = np.clip(uncertainty, 1e-10, 1.0)
        
        # Dirichlet 强度: S = K / u
        S = self.K / u  # (N,)
        S = S.reshape(-1, 1)  # (N, 1)
        
        # 证据: e_k = b_k × S
        evidence = belief * S  # (N, K)
        
        return evidence
    
    def ds_combine(
        self,
        belief1: np.ndarray,
        uncertainty1: np.ndarray,
        belief2: np.ndarray,
        uncertainty2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Dempster-Shafer 组合规则（两个视角）
        
        对应论文 Definition 3.2, Eq. (13):
        - b_k = (b1_k × b2_k + b1_k × u2 + b2_k × u1) / (1 - C)
        - u = (u1 × u2) / (1 - C)
        其中 C = Σ_{i≠j} b1_i × b2_j 是冲突因子
        
        Args:
            belief1, uncertainty1: 视角1的信念和不确定性
            belief2, uncertainty2: 视角2的信念和不确定性
        
        Returns:
            belief: 融合后的信念 (N, K)
            uncertainty: 融合后的不确定性 (N,)
        """
        N = belief1.shape[0]
        K = belief1.shape[1]
        
        # 扩展维度
        u1 = uncertainty1.reshape(-1, 1)  # (N, 1)
        u2 = uncertainty2.reshape(-1, 1)  # (N, 1)
        
        # 计算冲突因子 C = Σ_{i≠j} b1_i × b2_j
        # = (Σ_i b1_i)(Σ_j b2_j) - Σ_k b1_k × b2_k
        # = (1 - u1)(1 - u2) - Σ_k b1_k × b2_k
        
        # 一致部分（对角项）
        agreement = np.sum(belief1 * belief2, axis=1, keepdims=True)  # (N, 1)
        
        # 冲突因子
        # C = Σ_i Σ_j b1_i × b2_j (i≠j)
        # = (Σ_i b1_i)(Σ_j b2_j) - Σ_k b1_k × b2_k
        sum_b1 = np.sum(belief1, axis=1, keepdims=True)  # = 1 - u1
        sum_b2 = np.sum(belief2, axis=1, keepdims=True)  # = 1 - u2
        C = sum_b1 * sum_b2 - agreement  # (N, 1)
        
        # 归一化因子
        norm = 1 - C  # (N, 1)
        norm = np.clip(norm, 1e-10, None)  # 防止除零
        
        # 融合信念: b_k = (b1_k × b2_k + b1_k × u2 + b2_k × u1) / (1 - C)
        belief = (belief1 * belief2 + belief1 * u2 + belief2 * u1) / norm  # (N, K)
        
        # 融合不确定性: u = u1 × u2 / (1 - C)
        uncertainty = (u1 * u2 / norm).squeeze()  # (N,)
        
        return belief, uncertainty
    
    def ds_combine_multiple(
        self,
        beliefs: List[np.ndarray],
        uncertainties: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        多视角 DS 组合（顺序融合）
        
        对应论文 Eq. (14):
        M = M_1 ⊕ M_2 ⊕ ... ⊕ M_V
        
        Args:
            beliefs: 信念列表，每个元素 (N, K)
            uncertainties: 不确定性列表，每个元素 (N,)
        
        Returns:
            belief: 融合后的信念 (N, K)
            uncertainty: 融合后的不确定性 (N,)
        """
        assert len(beliefs) == len(uncertainties) >= 2
        
        belief = beliefs[0]
        uncertainty = uncertainties[0]
        
        for i in range(1, len(beliefs)):
            belief, uncertainty = self.ds_combine(
                belief, uncertainty,
                beliefs[i], uncertainties[i]
            )
        
        return belief, uncertainty
    
    def predict(
        self,
        belief: np.ndarray,
        uncertainty: np.ndarray,
        method: str = 'belief'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        基于融合结果进行预测
        
        Args:
            belief: 融合后的信念 (N, K)
            uncertainty: 融合后的不确定性 (N,)
            method: 预测方法
                - 'belief': 使用最大信念
                - 'probability': 使用期望概率 p_k = α_k / S = b_k + u/K
        
        Returns:
            predictions: 预测类别 (N,)
            confidence: 预测置信度 (N,)
        """
        if method == 'belief':
            predictions = np.argmax(belief, axis=1)
            confidence = np.max(belief, axis=1)
            
        elif method == 'probability':
            # 期望概率: p_k = α_k / S = (e_k + 1) / S
            # 由于 b_k = e_k / S, u = K / S
            # 所以 p_k = b_k + 1/S = b_k + u/K
            prob = belief + uncertainty.reshape(-1, 1) / self.K
            predictions = np.argmax(prob, axis=1)
            confidence = np.max(prob, axis=1)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return predictions, confidence
    
    def get_dirichlet_params(
        self,
        belief: np.ndarray,
        uncertainty: np.ndarray
    ) -> np.ndarray:
        """
        从主观意见恢复 Dirichlet 分布参数
        
        Args:
            belief: 信念 (N, K)
            uncertainty: 不确定性 (N,)
        
        Returns:
            alpha: Dirichlet 参数 (N, K)
        """
        evidence = self.opinion_to_evidence(belief, uncertainty)
        alpha = evidence + 1
        return alpha


def analyze_uncertainty(
    predictions: np.ndarray,
    true_labels: np.ndarray,
    uncertainty: np.ndarray,
    thresholds: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5]
) -> dict:
    """
    分析不确定性与预测正确性的关系
    
    Args:
        predictions: 预测类别
        true_labels: 真实标签
        uncertainty: 不确定性
        thresholds: 不确定性阈值列表
    
    Returns:
        分析结果字典
    """
    correct_mask = (predictions == true_labels)
    
    results = {
        'overall_accuracy': correct_mask.mean() * 100,
        'mean_uncertainty': uncertainty.mean(),
        'correct_mean_uncertainty': uncertainty[correct_mask].mean() if correct_mask.sum() > 0 else 0,
        'wrong_mean_uncertainty': uncertainty[~correct_mask].mean() if (~correct_mask).sum() > 0 else 0,
        'threshold_analysis': []
    }
    
    for thresh in thresholds:
        confident_mask = (uncertainty < thresh)
        if confident_mask.sum() > 0:
            acc = (predictions[confident_mask] == true_labels[confident_mask]).mean() * 100
            retain_ratio = confident_mask.sum() / len(predictions) * 100
            results['threshold_analysis'].append({
                'threshold': thresh,
                'num_samples': confident_mask.sum(),
                'retain_ratio': retain_ratio,
                'accuracy': acc
            })
    
    return results


def print_analysis_report(results: dict):
    """打印分析报告"""
    print("\n" + "="*60)
    print("不确定性分析报告")
    print("="*60)
    
    print(f"\n整体准确率: {results['overall_accuracy']:.2f}%")
    print(f"平均不确定性: {results['mean_uncertainty']:.4f}")
    print(f"正确样本平均不确定性: {results['correct_mean_uncertainty']:.4f}")
    print(f"错误样本平均不确定性: {results['wrong_mean_uncertainty']:.4f}")
    
    print("\n" + "-"*60)
    print("可信决策分析（按不确定性阈值过滤）")
    print("-"*60)
    print(f"{'阈值':<10} {'保留样本':<12} {'保留比例':<12} {'准确率'}")
    print("-"*60)
    
    for item in results['threshold_analysis']:
        print(f"u < {item['threshold']:<6} {item['num_samples']:<12} {item['retain_ratio']:.1f}%{'':<6} {item['accuracy']:.2f}%")
    
    print("-"*60)
```

### 4.2 `test_tmc_fusion.py` - 独立测试脚本

```python
"""
TMC 证据融合完整测试脚本
对比原始贝叶斯融合与 TMC 证据融合
"""

import numpy as np
import h5py
import sys
sys.path.append('.')

from evidence_fusion import TMCFusion, analyze_uncertainty, print_analysis_report


# ============================================================
# 配置路径（请根据实际情况修改）
# ============================================================
DATA_PATH = r"D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\VP-Terrain\data\_dark_7_new.hdf5"
RESULT_FOLDER = r"D:\桌面\智能机器人概论期末大作业\PKU-Intelligent-Robot-Course-Lab\SSL_pos\mobilenet_0.05"

PRO_CONF_PATH = f"{RESULT_FOLDER}\\pro_dark_conf"
VIS_CONF_PATH = f"{RESULT_FOLDER}\\vision_dark_conf"


# ============================================================
# 读取数据
# ============================================================
print("="*70)
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
print(f"{'ProNet (仅机体)':<30} {np.mean(np.argmax(pro_conf, axis=1) == true_labels)*100:.2f}%{'':<10} 单模态")
print(f"{'VisionNet (仅视觉)':<30} {np.mean(np.argmax(vis_conf, axis=1) == true_labels)*100:.2f}%{'':<10} 单模态")
print(f"{'原始贝叶斯融合':<30} {acc_bayes:.2f}%{'':<10} 简单相乘")
print(f"{'TMC 证据融合':<30} {acc_tmc:.2f}%{'':<10} DS 组合规则")
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

print("\n" + "="*70)
print("程序运行完成")
print("="*70)
```

---

## 五、方法对比总结

### 5.1 公式对比

| 项目 | 原始 TerraFusion | TMC 证据融合 |
|------|------------------|--------------|
| **输入** | $\text{conf}^{vis}, \text{conf}^{pro}$ | $e^{vis}, e^{pro}$ (证据) |
| **中间表示** | 归一化概率 | 信念 $b$ + 不确定性 $u$ |
| **融合公式** | $P \cdot P_{vis} \cdot P_{pro}$ | $b_k = \frac{b_1 b_2 + b_1 u_2 + b_2 u_1}{1-C}$ |
| **输出** | 仅预测类别 | 预测 + 不确定性 |
| **冲突处理** | 无 | $C = \sum_{i \neq j} b_i^1 b_j^2$ |

### 5.2 特性对比

| 特性 | 原始方法 | TMC |
|------|----------|-----|
| 不确定性建模 | ❌ | ✅ |
| 冲突处理 | ❌ | ✅ |
| 自适应融合 | ⚠️ 隐式 | ✅ 显式 |
| 可信决策 | ❌ | ✅ |
| 理论保证 | ⚠️ | ✅ |
| 计算开销 | 低 | 低 |

### 5.3 预期效果

1. **准确率**：TMC 与原始方法相近或略高
2. **可信决策**：可按不确定性阈值过滤，高置信样本准确率显著提升
3. **可解释性**：可输出每个样本的不确定性，解释预测可靠程度
4. **鲁棒性**：视觉失效时自动降低其权重

---

## 附录：论文核心引用

```
@article{han2022trusted,
  title={Trusted Multi-View Classification},
  author={Han, Zongbo and Zhang, Changqing and Fu, Huazhu and Zhou, Joey Tianyi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2022},
  publisher={IEEE}
}
```

**官方代码**: https://github.com/hanmenghan/TMC
