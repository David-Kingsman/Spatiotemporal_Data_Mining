"""Probabilistic U-Net Model for Image Interpolation"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple


@dataclass
class UNetConfig:
    """U-Net Configuration"""
    in_channels: int = 2  # temp + mask
    base_channels: int = 32
    lr: float = 1e-3
    num_epochs: int = 50
    batch_size: int = 8
    dropout: float = 0.2  # Dropout rate
    init_log_var: float = 0.0  # Initial value for log_var (0 corresponds to std=1)


class ConvBlock(nn.Module):
    """Convolution Block (with Dropout)"""
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
    Probabilistic U-Net: Outputs mean & log_var
    Suitable for pixel-wise Gaussian likelihood
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
        
        # Initialize log_var output to specified value
        nn.init.constant_(self.logvar_head.weight, 0.0)
        nn.init.constant_(self.logvar_head.bias, config.init_log_var)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
        x: (B, C, H, W) - Input image (temperature + mask)
        
        Returns:
        mean: (B, 1, H, W) - Predicted mean
        log_var: (B, 1, H, W) - Predicted log variance
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

        # Decoder - Handle size mismatch
        u3 = self.up3(mid)
        # Crop or pad to match d3 size (pad format: [left, right, top, bottom])
        diff_h3 = d3.size()[2] - u3.size()[2]
        diff_w3 = d3.size()[3] - u3.size()[3]
        if diff_h3 > 0 or diff_w3 > 0:
            u3 = torch.nn.functional.pad(u3, [
                diff_w3 // 2, diff_w3 - diff_w3 // 2,
                diff_h3 // 2, diff_h3 - diff_h3 // 2
            ])
        elif diff_h3 < 0 or diff_w3 < 0:
            # If upsampled size is larger, crop
            u3 = u3[:, :, :d3.size()[2], :d3.size()[3]]
        c3 = self.conv3(torch.cat([u3, d3], dim=1))
        
        u2 = self.up2(c3)
        # Crop or pad to match d2 size
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
        # Crop or pad to match d1 size
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

        # Output - Separate heads
        mean = self.mean_head(c1)  # (B, 1, H, W)
        log_var = self.logvar_head(c1)  # (B, 1, H, W)
        
        # Clamp log_var to avoid numerical instability
        # log_var between -5 and 5 (corresponds to std approx 0.006 to 2.7, reasonable after normalization)
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
    Gaussian Negative Log-Likelihood Loss (calculated only on observed points)
    Improved numerically stable version
    
    Args:
    mean: (B, 1, H, W) - Predicted mean
    log_var: (B, 1, H, W) - Predicted log variance (should be clamped)
    target: (B, 1, H, W) - True values
    mask: (B, 1, H, W) - Observation mask (1=observed, 0=missing)
    eps: Numerical stability parameter
    
    Returns:
    loss: Scalar
    """
    # Ensure log_var is within reasonable range (clamped in forward, ensuring here again)
    log_var = torch.clamp(log_var, min=-5.0, max=5.0)
    
    # Use numerically stable formula
    diff = target - mean
    diff_sq = diff ** 2
    
    # Calculate variance, ensure numerical stability
    var = torch.exp(log_var) + eps
    var = torch.clamp(var, min=eps, max=100.0)  # Limit variance range
    
    # Calculate NLL
    nll = 0.5 * (log_var + diff_sq / var)
    
    # Add regularization term: penalize excessive uncertainty
    # Encourages model to learn reasonable uncertainty
    uncertainty_reg = 0.01 * var.mean()  # Small regularization coefficient
    
    # Check for NaN or Inf
    nll = torch.clamp(nll, min=-100.0, max=100.0)
    
    # Sum only on observed points
    valid_mask = (mask > 0.5) & torch.isfinite(nll)
    if valid_mask.sum() > 0:
        loss = (nll * mask).sum() / valid_mask.sum() + uncertainty_reg
        # Final check
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(1000.0, device=mean.device)
    else:
        loss = torch.tensor(0.0, device=mean.device)
    
    return loss
