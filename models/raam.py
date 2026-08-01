"""
Ripeness-Aware Attention Module (RAAM)
Dual-branch channel attention with learnable ripeness bias vector.
Inserted at P3 and P4 scales in the LFPN neck.
"""
import torch
import torch.nn as nn

class RAAM(nn.Module):
    """Ripeness-Aware Attention Module.
    
    Args:
        channels (int): Number of input channels.
        reduction (int): Channel reduction ratio for FC bottleneck. Default: 16.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 8)
        
        # Shared FC bottleneck (used by both GAP and GMP branches)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        
        # Learnable ripeness bias vector (key novelty)
        # Provides persistent inductive preference toward
        # color-discriminative channels
        self.ripeness_bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Global Average Pooling branch (captures uniform color response)
        gap = x.mean(dim=[2, 3])                    # (B, C)
        gap_out = self.fc(gap)                       # (B, C)
        
        # Global Max Pooling branch (captures peak spectral response)
        gmp = x.amax(dim=[2, 3])                    # (B, C)
        gmp_out = self.fc(gmp)                       # (B, C)
        
        # Element-wise add + ripeness bias + sigmoid
        attn = gap_out + gmp_out                     # (B, C)
        attn = attn.view(b, c, 1, 1)                # (B, C, 1, 1)
        attn = self.sigmoid(attn + self.ripeness_bias)
        
        # Channel-wise multiply: F' = F * Attention
        return x * attn
