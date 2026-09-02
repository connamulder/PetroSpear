"""
    @Project:
    @File   : celoss_cus.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2026-07-09
    @Info   : 自形实现celoss，支持one_hot, smoothing, sim_target
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self, reduction='mean', label_smoothing=0.0, sim_matrix=None):
        """
        参数:
            reduction (str): 'none' | 'mean' | 'sum'
            label_smoothing (float): 标签平滑系数，值范围 [0, 1)
            sim_matrix (Tensor, optional): 类别相似度矩阵，形状为 (C, C)
        """
        super(CustomCrossEntropyLoss, self).__init__()

        # 验证参数
        assert reduction in ['none', 'mean', 'sum'], "reduction must be 'none', 'mean', or 'sum'"
        assert 0 <= label_smoothing < 1, "label_smoothing must be in [0, 1)"

        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.sim_matrix = None
        self.t = 0.1

        if sim_matrix is not None:
            if isinstance(sim_matrix, np.ndarray):
                self.sim_matrix = torch.from_numpy(sim_matrix).float()
            elif isinstance(sim_matrix, torch.Tensor):
                self.sim_matrix = sim_matrix

            temperature = 1.0
            # target_probs = F.softmax(self.sim_matrix / temperature, dim=1)
            target_probs = self.sim_matrix
            # 使用 register_buffer 确保矩阵在 model.to(device) 时自动转移到 GPU/CPU
            self.register_buffer('target_probs', target_probs)

    def forward(self, input, target):
        """
        参数:
            input (Tensor): 未归一化的 logits，形状为 (N, C)
            target (Tensor): 目标类别索引，形状为 (N,)，值为 [0, C-1]

        返回:
            Tensor: 计算得到的损失
        """

        # 如果输入是多维的（如图像），展平空间维度
        if input.dim() > 2:
            # (N, C, H, W) -> (N*H*W, C)
            N, C = input.shape[0], input.shape[1]
            input = input.view(N, C, -1)  # (N, C, H*W)
            input = input.permute(0, 2, 1).contiguous()  # (N, H*W, C)
            input = input.view(-1, C)  # (N*H*W, C)

            target = target.view(-1)  # (N*H*W,)

        if self.sim_matrix is not None:
            # 应用类别相似度知识约束
            target_prob = self.target_probs[target]
            target_prob = F.softmax(target_prob, dim=1)

            # loss = self._cross_entropy_with_embedding(input, target_prob)
            loss = self._cross_entropy(input, target)
            loss_rank = self._cross_entropy_with_rank(input, target_prob)

            return loss.mean(), loss_rank.mean()
        elif self.label_smoothing > 0:
            # 应用标签平滑, 可以改为考虑类别相似性的非均匀平滑（类别感知平滑）!!!
            num_classes = input.size(1)
            target_smooth = self._smooth_labels(target, num_classes)
            loss = self._cross_entropy_with_embedding(input, target_smooth)

            return loss.mean()
        else:
            # 标准交叉熵
            # loss = self._cross_entropy(input, target)

            # target转变为one-hot编码后再计算
            num_classes = input.size(1)
            # One-hot
            target_one_hot = F.one_hot(target, num_classes=num_classes)
            target_one_hot = target_one_hot.float()
            loss = self._cross_entropy_with_embedding(input, target_one_hot)

            return loss.mean()

    def _smooth_labels(self, target, num_classes):
        """
        应用标签平滑
            smooth_label[i] = (1 - smoothing) if i == target else smoothing / (num_classes - 1)

        参数:
            target (Tensor): 目标类别索引，形状为 (N, )，值为 [0, C-1]
            num_classes (int): 类别数目
        返回:
            Tensor: 平滑处理后的 target, 形状为 (N, C)
        """
        smooth_target = torch.full(
            (target.size(0), num_classes),
            self.label_smoothing / (num_classes - 1),
            device=target.device,
            dtype=torch.float32
        )
        smooth_target.scatter_(1, target.unsqueeze(1), 1 - self.label_smoothing)

        return smooth_target

    def _soft_rank(self, x):
        """
        计算可导的软排名 (Soft Rank)
        x: (N, C)
        return: (N, C) 的软排名，值域约为 [0, C-1]
        """
        # 计算两两差值: x_j - x_i
        # 形状变化: (N, 1, C) - (N, C, 1) -> (N, C, C)
        # 注意：这里计算的是 x_j - x_i，如果 x_j > x_i，sigmoid 趋近于 1
        diff = x.unsqueeze(1) - x.unsqueeze(2)

        # 使用 sigmoid 近似阶跃函数，并沿 j 维度求和
        # 结果即为：比 x_i 大的元素的“软数量”，也就是 x_i 的软排名 (0-indexed)
        ranks = torch.sigmoid(diff / self.t).sum(dim=2)

        return ranks

    def _cross_entropy(self, input, target):
        """
        标准交叉熵损失（无标签平滑）
            使用数值稳定的 log-sum-exp 技巧
            log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
        返回:
            Tensor: 计算得到的标准交叉熵损失
        """
        max_val = input.max(dim=1, keepdim=True)[0]
        log_sum_exp = max_val.squeeze(1) + torch.log(
            torch.sum(torch.exp(input - max_val), dim=1)
        )

        target_logits = input.gather(1, target.unsqueeze(1)).squeeze(1)

        # loss = -target_logits + log_sum_exp
        loss = -target_logits + log_sum_exp

        return loss

    def _cross_entropy_with_embedding(self, input, target_embedding):
        """
        带标签平滑的交叉熵损失
            loss = -sum(target_embedding * log_softmax(input))
        参数:
            input (Tensor): 未归一化的 logits，形状为 (N, C)
            target_embedding (Tensor): 向量化的 target, 形状为 (N, C)
        返回:
            Tensor: 计算得到的带标签平滑的交叉熵损失
        """
        log_probs = F.log_softmax(input, dim=1)

        # 计算交叉熵
        # 等价于最小化 KL 散度，促使 pred的log_probs 的排序分布逼近 target_embedding 的排序分布
        loss = -(target_embedding * log_probs).sum(dim=1)
        # print("函数 CustomCrossEntropyLoss 调用")

        return loss

    def _cross_entropy_with_rank(self, input, target_embedding):
        """
        input: 模型预测的 logits (N, C)
        target_embedding: 真实标签的得分/Teacher logits (N, C)
        """
        # 1. 计算软排名
        pred_ranks = self._soft_rank(input)
        target_ranks = self._soft_rank(target_embedding)

        # 2. 计算秩差的平方和 (Spearman 的核心部分)
        # 最小化秩差平方和 等价于 最大化 Spearman 相关系数
        d_squared = (pred_ranks - target_ranks) ** 2

        # 3. 返回 Loss
        return d_squared


def test_custom_celoss():
    torch.manual_seed(42)

    # batch_size=8, Classes=13
    inputs = torch.randn(8, 13, requires_grad=True)
    targets = torch.tensor([1, 0, 4, 2, 3, 5, 9, 12])

    def compare(name, official_fn, custom_fn, inps, tgts):
        loss_official = official_fn(inps, tgts)
        loss_custom = custom_fn(inps, tgts)
        match = torch.allclose(loss_official, loss_custom, atol=1e-5)
        print(f"[{name}] Official: {loss_official.item():.6f} | Custom: {loss_custom.item():.6f} | Match: {match}")

    # 测试 1: 基础 Label Smoothing
    print("--- 测试 1: 基础 Label Smoothing (α=0.0) ---")
    compare("Smoothing 0.0",
            nn.CrossEntropyLoss(),
            CustomCrossEntropyLoss(label_smoothing=0.0),
            inputs, targets)

    # 测试 2: 基础 Label Smoothing
    print("--- 测试 2: 基础 Label Smoothing (α=0.1) ---")
    compare("Smoothing 0.1",
            nn.CrossEntropyLoss(label_smoothing=0.1),
            CustomCrossEntropyLoss(label_smoothing=0.1),
            inputs, targets)

    """
    # 测试 3: 多维输入 (N, C, H, W)
    print("--- 测试 3: 多维输入 (N, C, H, W) = (2, 5, 4, 4) ---")
    batch_size, channels, height, width = 2, 5, 4, 4
    logits_2d = torch.randn(batch_size, channels, height, width)
    targets_2d = torch.randint(0, channels, (batch_size, height, width))
    compare("Smoothing 0.0",
            nn.CrossEntropyLoss(label_smoothing=0.0),
            CustomCrossEntropyLoss(label_smoothing=0.0),
            logits_2d, targets_2d)
    """

    # 测试 4: sim_matrix输入
    print("--- 测试 4: sim_matrix输入 Label Smoothing (α=0.0) ---")
    from similarity_matrix_neo4j import build_similarity_matrix_by_neo4j

    dataset_name = "plutonicrocks13_v3"
    sim_matrix = build_similarity_matrix_by_neo4j(dataset_name)
    criteron = CustomCrossEntropyLoss(label_smoothing=0.0)
    criteron_cus = CustomCrossEntropyLoss(sim_matrix=sim_matrix)
    loss = criteron(inputs, targets)
    loss_cus, loss_cus_rank = criteron_cus(inputs, targets)
    print(f"Official: {loss.item():.6f} | Custom: {loss_cus.item():.6f} | Custom_rank: {loss_cus_rank.item():.6f}")


if __name__ == "__main__":
    test_custom_celoss()
