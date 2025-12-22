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