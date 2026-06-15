"""
Model components: Backbone, DetectionHead, Detector, Loss.
Supports both lightweight CNN (CPU) and YOLOv8-style (GPU) backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
#  Lightweight CNN Backbone (for quick experiments / CPU)
# ═══════════════════════════════════════════════════════════════

class ConvBnAct(nn.Module):
    """Conv2d + BatchNorm2d + SiLU activation."""
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return F.silu(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    """Cross Stage Partial block (simplified C2f from YOLOv8)."""
    def __init__(self, c_in, c_out, n_bottleneck=1):
        super().__init__()
        c_mid = c_out // 2
        self.cv1 = ConvBnAct(c_in, c_mid, 1, 1, 0)
        self.cv2 = ConvBnAct(c_in, c_mid, 1, 1, 0)
        self.cv3 = ConvBnAct(c_mid * (n_bottleneck + 2), c_out, 1, 1, 0)
        self.m = nn.ModuleList([
            nn.Sequential(ConvBnAct(c_mid, c_mid, 3, 1, 1),
                          ConvBnAct(c_mid, c_mid, 3, 1, 1))
            for _ in range(n_bottleneck)
        ])

    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y = [y1, y2]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    def __init__(self, c_in, c_out, k=5):
        super().__init__()
        c_mid = c_in // 2
        self.cv1 = ConvBnAct(c_in, c_mid, 1, 1, 0)
        self.cv2 = ConvBnAct(c_mid * 4, c_out, 1, 1, 0)
        self.k = k

    def forward(self, x):
        x = self.cv1(x)
        y1 = F.max_pool2d(x, self.k, stride=1, padding=self.k // 2)
        y2 = F.max_pool2d(y1, self.k, stride=1, padding=self.k // 2)
        y3 = F.max_pool2d(y2, self.k, stride=1, padding=self.k // 2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class DetectionBackbone(nn.Module):
    """
    YOLOv8-style CSPDarknet backbone.
    
    Architecture (configurable width):
      Stem → Stage1 (P2) → Stage2 (P3) → Stage3 (P4) → Stage4 (P5) → SPPF
    
    Args:
        width: channel width multiplier (16=lightweight, 32=standard, 64=large)
        out_dim: output feature dimension after global pooling + FC
    """
    def __init__(self, width=32, out_dim=256):
        super().__init__()
        w = width
        self.stem = ConvBnAct(3, w, 3, 2, 1)           # /2

        # Stage 1: P2
        self.stage1 = nn.Sequential(
            ConvBnAct(w, w * 2, 3, 2, 1),              # /4
            CSPBlock(w * 2, w * 2, n_bottleneck=1),
        )
        # Stage 2: P3
        self.stage2 = nn.Sequential(
            ConvBnAct(w * 2, w * 4, 3, 2, 1),          # /8
            CSPBlock(w * 4, w * 4, n_bottleneck=2),
        )
        # Stage 3: P4
        self.stage3 = nn.Sequential(
            ConvBnAct(w * 4, w * 8, 3, 2, 1),          # /16
            CSPBlock(w * 8, w * 8, n_bottleneck=2),
        )
        # Stage 4: P5
        self.stage4 = nn.Sequential(
            ConvBnAct(w * 8, w * 8, 3, 2, 1),          # /32
            SPPF(w * 8, w * 8),
            CSPBlock(w * 8, w * 8, n_bottleneck=1),
        )

        # Global pooling + projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(w * 8, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        return self.proj(x.view(x.size(0), -1))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class DetectionHead(nn.Module):
    """
    Decoupled detection head (separate box + classification branches).
    One per client — never shared or aggregated.
    
    Args:
        in_dim: backbone output dimension
        nc: number of object classes
        max_obj: maximum objects per image (anchor slots)
    """
    def __init__(self, in_dim, nc, max_obj=5):
        super().__init__()
        self.max_obj = max_obj
        self.nc = nc
        hidden = max(in_dim, 256)

        # Box regression: [cx, cy, w, h] per anchor
        self.box_head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, max_obj * 4),
        )
        # Classification: nc classes per anchor
        self.cls_head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, max_obj * nc),
        )

    def forward(self, feat):
        """
        Args:
            feat: [B, in_dim] backbone features
        Returns:
            pred_box: [B, max_obj, 4] predicted boxes (cx, cy, w, h)
            pred_cls: [B, max_obj, nc] class logits
        """
        box = self.box_head(feat).view(feat.size(0), self.max_obj, 4)
        cls = self.cls_head(feat).view(feat.size(0), self.max_obj, self.nc)
        return box, cls

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class Detector(nn.Module):
    """Full detector = Backbone + Head."""
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

    def count_params(self, part='all'):
        if part == 'backbone':
            return self.backbone.count_params()
        elif part == 'head':
            return self.head.count_params()
        return self.backbone.count_params() + self.head.count_params()


# ═══════════════════════════════════════════════════════════════
#  Detection Loss
# ═══════════════════════════════════════════════════════════════

class DetectionLoss(nn.Module):
    """
    Combined box regression + classification loss.
    
    L = λ_box * L_box + λ_cls * L_cls
    
    Box loss: Smooth L1 on (cx, cy, w, h) for object slots only.
    Cls loss: Cross-entropy for object slots, ignored for background.
    """
    def __init__(self, nc, max_obj, lambda_box=5.0, lambda_cls=1.0):
        super().__init__()
        self.nc = nc
        self.max_obj = max_obj
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls

    def forward(self, pred_box, pred_cls, target):
        """
        Args:
            pred_box: [B, max_obj, 4]
            pred_cls: [B, max_obj, nc]
            target: [B, max_obj, 5] (cx, cy, w, h, class_id)
        """
        obj_mask = (target[..., 4] >= 0).float()  # [B, max_obj]
        n_obj = max(obj_mask.sum().item(), 1)

        # Box loss: Smooth L1
        box_diff = (pred_box - target[..., :4]) * obj_mask.unsqueeze(-1)
        box_loss = F.smooth_l1_loss(box_diff, torch.zeros_like(box_diff),
                                     reduction='sum') / n_obj

        # Classification loss: CE on object slots
        target_cls = target[..., 4].long().clamp(0, self.nc - 1)
        cls_logits = pred_cls.view(-1, self.nc)
        cls_targets = target_cls.view(-1)
        cls_loss_all = F.cross_entropy(cls_logits, cls_targets, reduction='none')
        cls_loss = (cls_loss_all * obj_mask.view(-1)).sum() / n_obj

        return self.lambda_box * box_loss + self.lambda_cls * cls_loss
