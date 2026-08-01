"""
Compact Detection Head (CDH) with Center-Point Localization (CPL) module.
Shares classification convolutions across P3/P4/P5 scales.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

class CDH(nn.Module):
    """Compact Detection Head with shared classification branch.
    
    Args:
        nc (int): Number of classes (default: 2, unripe/ripe).
        ch (list): Input channel list for P3, P4, P5.
    """
    def __init__(self, nc=2, ch=[64, 128, 256]):
        super().__init__()
        self.nc = nc
        self.nl = len(ch)  # number of detection levels
        
        # Shared classification branch (across all scales)
        # DW-Conv 3x3 -> PW-Conv 1x1 -> BN -> SiLU -> Conv 1x1 -> nc
        max_ch = max(ch)
        self.cls_shared = nn.Sequential(
            nn.Conv2d(max_ch, max_ch, 3, 1, 1, groups=max_ch, bias=False),
            nn.Conv2d(max_ch, max_ch, 1, bias=False),
            nn.BatchNorm2d(max_ch),
            nn.SiLU(inplace=True),
        )
        self.cls_pred = nn.Conv2d(max_ch, nc, 1)
        
        # Scale-specific channel adapters (project each scale to max_ch)
        self.cls_adapters = nn.ModuleList([
            nn.Conv2d(c, max_ch, 1) if c != max_ch else nn.Identity()
            for c in ch
        ])
        
        # Scale-specific regression branches
        self.reg_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
                nn.Conv2d(c, 4, 1),  # bbox: x, y, w, h
            ) for c in ch
        ])
    
    def forward(self, features):
        """
        Args:
            features: list of [P3, P4, P5] feature maps
        Returns:
            list of (cls_pred, reg_pred) per scale
        """
        outputs = []
        for i, feat in enumerate(features):
            # Classification (shared weights)
            cls_feat = self.cls_adapters[i](feat)
            cls_out = self.cls_pred(self.cls_shared(cls_feat))
            
            # Regression (scale-specific)
            reg_out = self.reg_branches[i](feat)
            
            outputs.append((cls_out, reg_out))
        return outputs


class CPLModule:
    """Center-Point Localization (CPL) Module.
    
    Post-detection geometry-based extraction with Gaussian refinement.
    NOT a supervised regression head — operates on predicted bounding boxes.
    
    Produces 2D image-plane center-point coordinates (cx, cy) for
    ripe-class detections only.
    """
    
    @staticmethod
    def extract_center_points(detections, conf_maps=None, refine=True):
        """Extract center points from ripe detections.
        
        Args:
            detections: tensor of shape (N, 6) [x1, y1, x2, y2, conf, cls]
            conf_maps: optional confidence heatmaps for Gaussian refinement
            refine: whether to apply Gaussian sub-pixel refinement
            
        Returns:
            centers: tensor of shape (M, 2) [cx, cy] for ripe detections
            ripe_mask: boolean mask of ripe detections
        """
        # Filter ripe detections (class = 1)
        ripe_mask = detections[:, 5] == 1  # class 1 = ripe
        ripe_dets = detections[ripe_mask]
        
        if len(ripe_dets) == 0:
            return torch.empty(0, 2), ripe_mask
        
        # Step 1: Geometric center extraction
        # cx = (x1 + x2) / 2, cy = (y1 + y2) / 2
        cx = (ripe_dets[:, 0] + ripe_dets[:, 2]) / 2
        cy = (ripe_dets[:, 1] + ripe_dets[:, 3]) / 2
        centers = torch.stack([cx, cy], dim=1)
        
        # Step 2: Gaussian sub-pixel refinement (optional)
        if refine and conf_maps is not None:
            centers = CPLModule._gaussian_refine(
                centers, ripe_dets, conf_maps
            )
        
        return centers, ripe_mask
    
    @staticmethod
    def _gaussian_refine(centers, detections, conf_maps, sigma=1.5):
        """Refine center points using confidence heatmap within bbox.
        
        Shifts geometric center toward the sub-pixel peak response,
        correcting for asymmetric occlusion or partial visibility.
        
        Args:
            centers: (M, 2) geometric centers
            detections: (M, 6) ripe detections
            conf_maps: confidence heatmaps from detection head
            sigma: Gaussian smoothing sigma
            
        Returns:
            refined: (M, 2) refined center points
        """
        refined = centers.clone()
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det[:4].int()
            
            # Ensure valid bbox
            x1, y1 = max(0, x1), max(0, y1)
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract confidence patch within bbox
            if conf_maps is not None:
                patch = conf_maps[y1:y2, x1:x2].cpu().numpy()
                
                if patch.size == 0:
                    continue
                
                # Apply Gaussian smoothing
                smoothed = gaussian_filter(patch.astype(float), sigma=sigma)
                
                # Find sub-pixel peak
                peak_idx = smoothed.argmax()
                peak_y, peak_x = divmod(peak_idx, smoothed.shape[1])
                
                # Convert back to image coordinates
                refined[i, 0] = x1 + peak_x + 0.5
                refined[i, 1] = y1 + peak_y + 0.5
        
        return refined
