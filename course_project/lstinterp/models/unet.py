"""概率U-Net模型用于图像插值"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple


@dataclass
class UNetConfig:
    """U-Net配置"""
    in_channels: int = 2  # temp + mask
    base_channels: int = 32
    lr: float = 1e-3
    num_epochs: int = 50
    batch_size: int = 8
    dropout: float = 0.2  # dropout率
    init_log_var: float = 0.0  # log_var的初始值（0对应标准差=1）


class ConvBlock(nn.Module):
    """卷积块（带dropout）"""
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend([
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ProbUNet(nn.Module):
    """
    概率U-Net：输出 mean & log_var
    适用于像素级高斯似然
    """
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.config = config
        ch = config.base_channels

        # Encoder
        self.down1 = ConvBlock(config.in_channels, ch, dropout=config.dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(ch, ch * 2, dropout=config.dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ConvBlock(ch * 2, ch * 4, dropout=config.dropout)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.mid = ConvBlock(ch * 4, ch * 8, dropout=config.dropout)

        # Decoder
        self.up3 = nn.ConvTranspose2d(ch * 8, ch * 4, 2, stride=2)
        self.conv3 = ConvBlock(ch * 8, ch * 4, dropout=config.dropout)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.conv2 = ConvBlock(ch * 4, ch * 2, dropout=config.dropout)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.conv1 = ConvBlock(ch * 2, ch, dropout=config.dropout)

        # Output head: mean and log_var
        self.mean_head = nn.Conv2d(ch, 1, 1)
        self.logvar_head = nn.Conv2d(ch, 1, 1)
        
        # 初始化log_var输出为指定值
        nn.init.constant_(self.logvar_head.weight, 0.0)
        nn.init.constant_(self.logvar_head.bias, config.init_log_var)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        参数:
        x: (B, C, H, W) - 输入图像（温度+mask）
        
        返回:
        mean: (B, 1, H, W) - 预测均值
        log_var: (B, 1, H, W) - 预测对数方差
        """
        # Encoder
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)

        # Bottleneck
        mid = self.mid(p3)

        # Decoder - 注意处理尺寸不匹配的问题
        u3 = self.up3(mid)
        # 裁剪或填充以匹配d3的尺寸 (pad格式: [left, right, top, bottom])
        diff_h3 = d3.size()[2] - u3.size()[2]
        diff_w3 = d3.size()[3] - u3.size()[3]
        if diff_h3 > 0 or diff_w3 > 0:
            u3 = torch.nn.functional.pad(u3, [
                diff_w3 // 2, diff_w3 - diff_w3 // 2,
                diff_h3 // 2, diff_h3 - diff_h3 // 2
            ])
        elif diff_h3 < 0 or diff_w3 < 0:
            # 如果上采样后尺寸更大，需要裁剪
            u3 = u3[:, :, :d3.size()[2], :d3.size()[3]]
        c3 = self.conv3(torch.cat([u3, d3], dim=1))
        
        u2 = self.up2(c3)
        # 裁剪或填充以匹配d2的尺寸
        diff_h2 = d2.size()[2] - u2.size()[2]
        diff_w2 = d2.size()[3] - u2.size()[3]
        if diff_h2 > 0 or diff_w2 > 0:
            u2 = torch.nn.functional.pad(u2, [
                diff_w2 // 2, diff_w2 - diff_w2 // 2,
                diff_h2 // 2, diff_h2 - diff_h2 // 2
            ])
        elif diff_h2 < 0 or diff_w2 < 0:
            u2 = u2[:, :, :d2.size()[2], :d2.size()[3]]
        c2 = self.conv2(torch.cat([u2, d2], dim=1))
        
        u1 = self.up1(c2)
        # 裁剪或填充以匹配d1的尺寸
        diff_h1 = d1.size()[2] - u1.size()[2]
        diff_w1 = d1.size()[3] - u1.size()[3]
        if diff_h1 > 0 or diff_w1 > 0:
            u1 = torch.nn.functional.pad(u1, [
                diff_w1 // 2, diff_w1 - diff_w1 // 2,
                diff_h1 // 2, diff_h1 - diff_h1 // 2
            ])
        elif diff_h1 < 0 or diff_w1 < 0:
            u1 = u1[:, :, :d1.size()[2], :d1.size()[3]]
        c1 = self.conv1(torch.cat([u1, d1], dim=1))

        # Output - 分离的head
        mean = self.mean_head(c1)  # (B, 1, H, W)
        log_var = self.logvar_head(c1)  # (B, 1, H, W)
        
        # 限制log_var的范围以避免数值不稳定
        # log_var在-5到5之间（对应标准差约0.006到2.7，归一化后合理）
        log_var = torch.clamp(log_var, min=-5.0, max=5.0)
        
        return mean, log_var


def gaussian_nll_loss(
    mean: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    高斯负对数似然损失（只在观测点上计算）
    改进的数值稳定版本
    
    参数:
    mean: (B, 1, H, W) - 预测均值
    log_var: (B, 1, H, W) - 预测对数方差（应该已经被clamp过）
    target: (B, 1, H, W) - 真实值
    mask: (B, 1, H, W) - 观测mask（1=观测，0=缺失）
    eps: 数值稳定性参数
    
    返回:
    loss: 标量
    """
    # 确保log_var在合理范围内（已在forward中clamp，这里再次确保）
    log_var = torch.clamp(log_var, min=-5.0, max=5.0)
    
    # 使用数值稳定的公式
    diff = target - mean
    diff_sq = diff ** 2
    
    # 计算方差，确保数值稳定
    var = torch.exp(log_var) + eps
    var = torch.clamp(var, min=eps, max=100.0)  # 限制方差范围
    
    # 计算NLL
    nll = 0.5 * (log_var + diff_sq / var)
    
    # 添加正则化项：惩罚过大的不确定性（鼓励模型学习合理的uncertainty）
    # 如果不确定性太大，说明模型不确定，应该被惩罚
    # 添加一个小的正则化项：lambda * exp(log_var)，鼓励较小的不确定性
    uncertainty_reg = 0.01 * var.mean()  # 小的正则化系数
    
    # 检查是否有NaN或Inf
    nll = torch.clamp(nll, min=-100.0, max=100.0)
    
    # 只在观测点上累加
    valid_mask = (mask > 0.5) & torch.isfinite(nll)
    if valid_mask.sum() > 0:
        loss = (nll * mask).sum() / valid_mask.sum() + uncertainty_reg
        # 最终检查
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(1000.0, device=mean.device)
    else:
        loss = torch.tensor(0.0, device=mean.device)
    
    return loss

