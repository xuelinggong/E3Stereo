"""
Geometric Edge Branch: Reuses IGEV's Feature as backbone, connected to EdgeHead to learn depth-discontinuity geometric edges predicted from RGB.

Used for validation: Whether the model can learn geometric edge features from a single RGB image on SceneFlow synthetic data (labels are GT edges generated from disparity gradients).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.extractor import Feature
from core.submodule import BasicConv_IN


class SpatialAttention(nn.Module):
    """
    Spatial Attention: Generates a spatial weight map from multi-scale edge predictions to enhance edge regions and suppress background.
    Edges are sparse; spatial attention helps the network focus more around boundaries.
    """
    def __init__(self, in_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        x: [B, C, H, W] concatenation of multi-scale edge predictions
        return: [B, 1, H, W] spatial weights, close to 1 in edge regions, close to 0 in background
        """
        return self.conv(x)


class EdgeRefinementModule(nn.Module):
    """
    Fine Edge Refinement Module: Uses RGB guidance at full resolution to refine blurred coarse edges.
    Addresses the issue of "fine edges blurring together": Learns to sharpen via residual, making predictions finer and clearer.
    """
    def __init__(self, in_channels=4):
        super().__init__()
        # in_channels: 1(edge) + 3(rgb)
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 1)
        self.norm = nn.InstanceNorm2d(32)

    def forward(self, edge_logits, rgb):
        """
        edge_logits: [B, 1, H, W] coarse edge prediction (upsampled to full-res)
        rgb: [B, 3, H, W] original image (used for guidance)
        """
        x = torch.cat([edge_logits, rgb], dim=1)
        x = F.leaky_relu(self.norm(self.conv1(x)), 0.1)
        x = F.leaky_relu(self.norm(self.conv2(x)), 0.1)
        delta = self.conv3(x)
        return edge_logits + delta  # residual


class EdgeHead(nn.Module):
    """
    Edge prediction head with multi-scale feature fusion.
    Input: Multi-scale features [x4, x8, x16, x32] output from Feature backbone
    Output: Single-channel edge map (logits) with the same resolution as the input
    use_spatial_attn: Whether to use spatial attention
    """
    def __init__(self, feat_channels=(48, 64, 192, 160), use_spatial_attn=True):
        super().__init__()
        self.use_spatial_attn = use_spatial_attn
        # feat_channels correspond to [x4, x8, x16, x32] channel counts output from Feature
        c4, c8, c16, c32 = feat_channels

        # Predict edges from each scale, then upsample and fuse
        self.edge_4 = nn.Sequential(
            BasicConv_IN(c4, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, 1),
        )
        self.edge_8 = nn.Sequential(
            BasicConv_IN(c8, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, 1),
        )
        self.edge_16 = nn.Sequential(
            BasicConv_IN(c16, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, 1),
        )
        self.edge_32 = nn.Sequential(
            BasicConv_IN(c32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 1, 1),
        )

        # Fuse predictions from different scales; use a learnable scale to make e4 more dominant, favoring fine edges
        self.scale = nn.Parameter(torch.ones(4) * 0.4)  # e4 is slightly higher by default
        self.scale.data[0] = 1.2  # e4 has higher weight for fine scales
        self.spatial_attn = SpatialAttention(in_channels=4) if use_spatial_attn else None
        self.fuse = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, features, target_size=None):
        """
        features: list of [x4, x8, x16, x32]
        target_size: (H, W) final output upsampling size, fusion is always performed at e4 resolution
        """
        e4 = self.edge_4(features[0])
        e8 = self.edge_8(features[1])
        e16 = self.edge_16(features[2])
        e32 = self.edge_32(features[3])

        # Unify to e4 resolution during fusion (x4 is 1/4 of input)
        h, w = e4.shape[2], e4.shape[3]

        e8_up = F.interpolate(e8, size=(h, w), mode='bilinear', align_corners=False)
        e16_up = F.interpolate(e16, size=(h, w), mode='bilinear', align_corners=False)
        e32_up = F.interpolate(e32, size=(h, w), mode='bilinear', align_corners=False)

        # Multi-scale fusion, scale makes e4 more dominant, favoring fine edges
        scale = F.softmax(self.scale, dim=0)
        fused = torch.cat([
            e4 * scale[0], e8_up * scale[1], e16_up * scale[2], e32_up * scale[3]
        ], dim=1)
        # Spatial attention: enhance edge regions, suppress background
        if self.spatial_attn is not None:
            attn = self.spatial_attn(fused)
            fused = fused * (1.0 + attn)
        edge_logits = self.fuse(fused)
        return edge_logits


class GeoEdgeNet(nn.Module):
    """
    Geometric Edge Network: Reuses IGEV Feature backbone + EdgeHead.
    Input: RGB image [B, 3, H, W], normalized to [-1, 1]
    Output: edge logits [B, 1, H, W]
    use_refinement: Whether to use EdgeRefinementModule to sharpen fine edges
    refine_iters: Number of Refine iterations, 1=single pass, 2/3=iterative sharpening (shares the same Refine module)
    use_spatial_attn: Whether to use spatial attention (edge enhancement, background suppression)
    """
    def __init__(self, use_refinement=True, refine_iters=1, use_spatial_attn=True):
        super().__init__()
        self.use_refinement = use_refinement
        self.refine_iters = max(1, int(refine_iters))
        self.backbone = Feature()
        self.edge_head = EdgeHead(
            feat_channels=(48, 64, 192, 160),
            use_spatial_attn=use_spatial_attn,
        )
        if use_refinement:
            self.refine = EdgeRefinementModule(in_channels=4)

    def forward(self, x, target_size=None):
        """
        x: [B, 3, H, W], recommended range [-1, 1] or [0, 1]
        target_size: (H, W) output size, defaults to match input
        """
        features = self.backbone(x)
        if target_size is None:
            target_size = (x.shape[2], x.shape[3])
        edge_logits = self.edge_head(features, target_size)
        # Upsample to input resolution
        if edge_logits.shape[2:] != target_size:
            edge_logits = F.interpolate(
                edge_logits, size=target_size,
                mode='bilinear', align_corners=False
            )
        # Iterative Refine: coarse -> refine -> refine -> ... (shares the same Refine)
        if self.use_refinement:
            for _ in range(self.refine_iters):
                edge_logits = self.refine(edge_logits, x)
        return edge_logits
