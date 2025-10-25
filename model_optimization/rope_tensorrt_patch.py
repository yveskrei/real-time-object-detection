
import torch
import torch.nn as nn

class RoPE2DTensorRT(nn.Module):
    """TensorRT-compatible RoPE implementation without conditional logic"""
    
    def __init__(self, freq=10000.0, max_res=224):
        super().__init__()
        self.freq = freq
        self.max_res = max_res
        
        # Pre-compute frequencies
        freqs = 1.0 / (freq ** (torch.arange(0, max_res, 2).float() / max_res))
        self.register_buffer('freqs', freqs)
    
    def forward(self, x, patch_shape):
        """
        x: [B, N, C] where N = h * w
        patch_shape: (h, w)
        """
        h, w = patch_shape
        
        # Always create 2D grid (no conditionals!)
        # We explicitly use torch.float32, and the 'Range' op_block_list
        # will protect this from being converted to HALF.
        pos_h = torch.arange(h, device=x.device, dtype=torch.float32)
        pos_w = torch.arange(w, device=x.device, dtype=torch.float32)
        
        # Create meshgrid
        grid_h, grid_w = torch.meshgrid(pos_h, pos_w, indexing='ij')
        
        # Flatten
        pos_h_flat = grid_h.reshape(-1, 1)
        pos_w_flat = grid_w.reshape(-1, 1)
        
        # Compute frequencies
        freqs_h = pos_h_flat @ self.freqs[:h//2].unsqueeze(0)
        freqs_w = pos_w_flat @ self.freqs[:w//2].unsqueeze(0)
        
        # Concatenate
        freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        
        # Apply rotation
        cos = freqs.cos().to(x.dtype) # Cast to x.dtype (HALF)
        sin = freqs.sin().to(x.dtype) # Cast to x.dtype (HALF)
        
        # Rotate x
        x_reshape = x.reshape(*x.shape[:-1], -1, 2)
        x_rotated = torch.stack([
            x_reshape[..., 0] * cos - x_reshape[..., 1] * sin,
            x_reshape[..., 0] * sin + x_reshape[..., 1] * cos
        ], dim=-1)
        
        return x_rotated.reshape_as(x)

# Monkey patch to replace original
import sys
if 'dinov3.layers' in sys.modules:
    sys.modules['dinov3.layers'].RoPE2D = RoPE2DTensorRT
