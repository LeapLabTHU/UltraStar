import einops
import torch
import torch.nn as nn
import torchvision
import pdb
from einops import rearrange
import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def build_proj(proj_type, in_dim, out_dim):
    if proj_type == 'mlp' or proj_type == 'mlp2':
        linear = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.GELU(),
                nn.Linear(in_dim // 2, out_dim)
            )
    elif proj_type == 'mlp3':
        linear = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.GELU(),
                nn.Linear(in_dim // 2, in_dim // 4),
                nn.GELU(),
                nn.Linear(in_dim // 4, out_dim)
            )
    elif proj_type == 'mlp4':
        linear = nn.Sequential(
                nn.Linear(in_dim, in_dim // 2),
                nn.GELU(),
                nn.Linear(in_dim // 2, in_dim // 4),
                nn.GELU(),
                nn.Linear(in_dim // 4, in_dim // 8),
                nn.GELU(),
                nn.Linear(in_dim // 8, out_dim)
            ) 
    elif proj_type == 'fc':
        linear = nn.Linear(in_dim, out_dim)
    else:
        raise NotImplementedError
    return linear


class IJEPA_Ultrastar_Model(nn.Module):
    def __init__(self, feature_model, num_classes=6, proj_type='mlp', refine_depth=2):
        super().__init__()
        self.feature_model = feature_model
        self.out_dim = feature_model.embed_dim
        self.num_classes = num_classes
        
        self.fc_out = nn.ModuleList([build_proj(proj_type, self.out_dim, num_classes) for _ in range(10)])
        
        self.action_encoder = nn.Linear(num_classes, self.out_dim, bias=True)
        
        self.refine_blocks = nn.ModuleList([
            AnchorSelfAttnBlock(dim=self.out_dim * 2, num_heads=4)
            for _ in range(refine_depth)
        ])

        # Cross-Attention: Current Query -> Refined Anchors
        self.locator_attn = GeometricCrossAttn_Concat(
            query_dim=self.out_dim, 
            key_dim=self.out_dim * 2, 
            num_heads=4
        )

    def forward(self, imgs, acts):
        B, N = imgs.shape[0], imgs.shape[1]

        # 1. Image Features
        all_img_feats = self.feature_model(imgs.reshape(-1, *imgs.shape[2:]))
        if all_img_feats.ndim == 3:
            all_img_feats = all_img_feats.mean(1)
        all_img_feats = all_img_feats.view(B, N, -1)

        curr_feat = all_img_feats[:, -1:, :]  # [B, 1, D]
        hist_feats = all_img_feats[:, :-1, :] # [B, N-1, D]

        # 2. Action Features
        act_feats = self.action_encoder(acts) # [B, N-1, D]

        # 3. Construct Anchors 
        anchors = torch.cat([hist_feats, act_feats], dim=-1) # [B, N-1, 2D]

        # 4. Trajectory Refinement 
        for blk in self.refine_blocks:
            anchors = blk(anchors)

        # 5. Localization (Cross Attention)
        context_feat = self.locator_attn(curr_feat, anchors)
        
        final_feat = context_feat.squeeze(1)

        # 6. Prediction
        outputs = [fc(final_feat) for fc in self.fc_out]
        outputs = torch.stack(outputs, dim=1).view(B, -1)

        return outputs



class AnchorSelfAttnBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + mlp_out
        return x

class GeometricCrossAttn_Concat(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads 
        
        self.q_proj = nn.Linear(query_dim, query_dim)
        self.k_proj = nn.Linear(key_dim, query_dim) 
        self.v_proj = nn.Linear(key_dim, query_dim) 
        
        self.out_proj = nn.Linear(query_dim, query_dim)

    def forward(self, curr_feat, anchors):
        B = curr_feat.shape[0]
        N_hist = anchors.shape[1]
        H, D = self.num_heads, self.head_dim
        
        q = self.q_proj(curr_feat).view(B, 1, H, D).transpose(1, 2)
        
        k = self.k_proj(anchors).view(B, N_hist, H, D).permute(0, 2, 1, 3)
        v = self.v_proj(anchors).view(B, N_hist, H, D).permute(0, 2, 1, 3)
        
        scores = (q @ k.transpose(-2, -1)) / (D ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        
        context = (attn_weights @ v).transpose(1, 2).reshape(B, 1, -1)
        
        return self.out_proj(context) + curr_feat
    
    
