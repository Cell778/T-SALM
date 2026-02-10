# Attention Pooling Implementation for sCLAP

import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentivePooling(nn.Module):
    """
    Attentive Pooling module that computes weighted average of temporal features.
    
    Args:
        dim (int): Input feature dimension
    """
    def __init__(self, dim):
        super().__init__()
        # Simple attention mechanism: Linear projection to score
        self.attn = nn.Linear(dim, 1)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (B, T, C)
            mask: Optional mask (B, T) with 1 for valid, 0 for padded/masked
        
        Returns:
            pooled: (B, C)
        """
        # Calculate scores
        scores = self.attn(x).squeeze(-1) # (B, T)
        
        if mask is not None:
             # Apply mask: Set invalid positions to -inf
             scores = scores.masked_fill(mask == 0, -1e9)
             
        # Softmax to get weights
        attn_weights = F.softmax(scores, dim=-1) # (B, T)
        
        # Weighted sum
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1) # (B, 1, C) -> (B, C)
        
        return pooled, attn_weights

class MultiGrainAttentionPooling(nn.Module):
    """
    Apply attentive pooling within the time boundaries defined by the masks.
    """
    def __init__(self, dim):
        super().__init__()
        self.pool = AttentivePooling(dim)
        
    def forward(self, x, mask1, mask2):
        """
        Args:
            x: (B, T, C)
            mask1: (B, T) - 1.0 for event 1
            mask2: (B, T) - 1.0 for event 2
        """
        # Pool Event 1
        pool1, weight1 = self.pool(x, mask1)
        
        # Pool Event 2
        pool2, weight2 = self.pool(x, mask2)
        
        # Stack
        pooled = torch.stack([pool1, pool2], dim=1) # (B, 2, C)
        
        # Combine weights for visualization if needed
        # Note: weights are separate softmax distributions
        return pooled, weight1, weight2
