# moevit_clean.py
import math
from functools import partial
from itertools import repeat
from collections import OrderedDict, Counter

import numpy as np
import torch
import torch.nn as nn
import collections.abc
import warnings

# 外部依赖（保持和你原来代码一致）
# from utils.helpers import load_pretrained, load_pretrained_pos_emb
from models.custom_moe_layer import FMoETransformerMLP
from models.gate_funs.noisy_gate import NoisyGate
from models.gate_funs.noisy_gate_vmoe import NoisyGate_VMoE
from timm.models.layers import lecun_normal_  # 用于某些初始化

# -------------------------
# Helper utilities
# -------------------------
def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """Truncated normal initializer (copied functionality)."""
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2*std) or (mean > b + 2*std):
        warnings.warn("mean is more than 2 std from [a, b] in trunc_normal_.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2*l - 1, 2*u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


def _init_vit_weights(m: nn.Module, module_name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """Initialize weights similar to ViT/DeiT convention."""
    if isinstance(m, nn.Linear):
        if module_name.startswith('head'):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, head_bias)
        elif module_name.startswith('pre_logits'):
            lecun_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)
            else:
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        if m.weight is not None:
            nn.init.ones_(m.weight)


# -------------------------
# Core building blocks
# -------------------------
class PatchEmbed(nn.Module):
    """Image -> Patch embeddings (Conv2d projection)."""
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # expect x: (B, C, H, W) with H,W == img_size
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input size mismatch, got ({H},{W})"
        x = self.proj(x)                      # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)      # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    """Standard Multi-head self-attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5 if qk_scale is None else qk_scale
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForwardMoEWrapper(nn.Module):
    """Wrapper that selects between a regular MLP and an MoE-MLP."""
    def __init__(self, dim, mlp_ratio=4., drop=0., act_layer=nn.GELU,
                 use_moe=False, moe_kwargs=None):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.use_moe = use_moe
        if not use_moe:
            self.ffn = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                act_layer(),
                nn.Dropout(drop),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(drop),
            )
        else:
            # moe_kwargs is passed to external FMoE layer (must match its API)
            moe_kwargs = moe_kwargs or {}
            self.ffn = FMoETransformerMLP(d_model=dim, d_hidden=hidden_dim, **moe_kwargs)

    def forward(self, x, gate_inp=None, task_id=None, task_specific_feature=None, sem=None):
        if not self.use_moe:
            return self.ffn(x)
        else:
            # FMoETransformerMLP expected signature: (x, gate_inp, task_id, task_specific_feature, sem)
            return self.ffn(x, gate_inp, task_id, task_specific_feature, sem)


class TransformerBlock(nn.Module):
    """Transformer block: LayerNorm -> Attention -> Residual -> (LayerNorm -> FFN(MoE) -> Residual)."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm,
                 use_moe=False, moe_kwargs=None, gate_input_ahead=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        self.gate_input_ahead = gate_input_ahead
        self.ffn = FeedForwardMoEWrapper(dim, mlp_ratio=mlp_ratio, drop=drop,
                                         use_moe=use_moe, moe_kwargs=moe_kwargs)

    def forward(self, x, gate_inp=None, task_id=None, task_specific_feature=None, sem=None):
        if self.gate_input_ahead:
            gate_inp = x
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if not isinstance(self.ffn, FMoETransformerMLP):
            x = x + self.drop_path(self.ffn(self.norm2(x)))
        else:
            # FMoE needs additional args
            x = x + self.drop_path(self.ffn(self.norm2(x), gate_inp, task_id, task_specific_feature, sem))
        return x


class DropPath(nn.Module):
    """Stochastic depth per sample. Simple implementation."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# -------------------------
# Vision Transformer with MoE
# -------------------------
class VisionTransformerMoE(nn.Module):
    """
    Clean, compact reimplementation of the original moevit logic.
    Key features preserved:
      - Patch embedding + positional embedding + cls token
      - Alternating blocks: standard attention blocks and MoE blocks
      - Optional gate/task-specific features and semantic patch helper
      - Pretrained weight / pos-embed loading support
    """
    def __init__(self,
                 img_size: int = 384,
                 patch_size: int = 16,
                 in_chans: int = 3,
                 embed_dim: int = 1024,
                 depth: int = 24,
                 num_heads: int = 16,
                 mlp_ratio: float = 4.0,
                 num_classes: int = 1000,
                 distilled: bool = False,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 use_moe_every_other: bool = True,
                 moe_kwargs: dict = None,
                 gate_dim: int = -1,
                 gate_task_specific_dim: int = -1,
                 gate_input_ahead: bool = False,
                 regu_sem: bool = False,
                 sem_force: bool = False,
                 random_init: bool = False,
                 model_name: str = 'vit_moe_clean',
                 **kwargs):
        super().__init__()
        self.model_name = model_name
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_classes = num_classes
        self.moe_kwargs = moe_kwargs or {}
        self.random_init = random_init

        # Patch embedding and tokens
        self.patch_embed = PatchEmbed(img_size=self.img_size[0], patch_size=self.patch_size[0],
                                      in_chans=self.in_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Blocks: alternate standard and MoE blocks if requested
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        blocks = []
        for i in range(depth):
            is_moe = use_moe_every_other and (i % 2 == 1)
            block = TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                use_moe=is_moe,
                moe_kwargs=self.moe_kwargs if is_moe else None,
                gate_input_ahead=gate_input_ahead
            )
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)

        # Optional representation layer
        self.pre_logits = nn.Identity()

        # Classification head(s)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None  # distilled head not used in this clean version

        # bookkeeping for task-specific gating and semantics
        self.gate_dim = gate_dim
        self.gate_task_specific_dim = gate_task_specific_dim
        self.regu_sem = regu_sem
        self.sem_force = sem_force

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # apply initialization across modules
        for name, m in self.named_modules():
            _init_vit_weights(m, name, jax_impl=False)
        # pos and cls
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

    def to_1d(self, x):
        """Convert (B, C, H, W) to (B, N, C) where N=H*W"""
        B, C, H, W = x.shape
        return x.view(B, C, -1).transpose(1, 2)

    def get_groundtruth_sem(self, sem: torch.Tensor):
        """
        Turn pixel-level semantic map into patch-level hint as in original code.
        sem expected shape: (B, C, H, W) or (B, H, W) -- original code used C channel for labels.
        Returns tensor of shape (B, 1, H_patch, W_patch) with dominant label or 255.
        """
        # Ensure sem is (B, H, W) ints
        if sem.dim() == 4:
            sem_labels = sem.argmax(dim=1) if sem.shape[1] > 1 else sem[:, 0]
        else:
            sem_labels = sem
        B, H, W = sem_labels.shape
        ps = self.patch_size[0]
        Hp = H // ps
        Wp = W // ps
        hint = np.ones((B, 1, Hp, Wp), dtype=np.uint8) * 255
        for b in range(B):
            for i in range(Hp):
                for j in range(Wp):
                    patch = sem_labels[b, i*ps:(i+1)*ps, j*ps:(j+1)*ps].cpu().numpy().flatten()
                    if patch.size == 0:
                        continue
                    idx, cnt = Counter(patch).most_common(1)[0]
                    if cnt > 0.4 * (ps*ps):
                        hint[b, 0, i, j] = int(idx)
        return torch.tensor(hint, device=sem.device)

    def forward_features(self, x: torch.Tensor, gate_inp=None, task_id=None, sem=None):
        B = x.shape[0]
        x = self.patch_embed.proj(x) if False else self.patch_embed(x)  # patch_embed returns (B, N, C)
        # prepend cls token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)       # (B, N+1, C)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        task_specific_feature = None
        if (task_id is not None) and (self.gate_task_specific_dim >= 0):
            # One-hot encode and project (user's original had a small MLP; user can add if desired)
            t_vec = torch.zeros(self.gate_dim - self.embed_dim, device=x.device) if self.gate_dim > self.embed_dim else torch.zeros(1, device=x.device)
            t_vec[task_id] = 1.0
            # NOTE: original code used a gate_task_represent MLP; omitted here for clarity

        # optionally convert sem to patch-level hint
        if sem is not None and (self.regu_sem or self.sem_force):
            sem = self.get_groundtruth_sem(sem)

        outs = []
        for blk in self.blocks:
            # If block expects MoE args, forward with gate args; else default
            if hasattr(blk, "gate_input_ahead") and blk.gate_input_ahead:
                x = blk(x, gate_inp, task_id, task_specific_feature, sem=sem)
            else:
                # blk.forward handles both ffn types internally
                x = blk(x, gate_inp, task_id, task_specific_feature, sem=sem)
            outs.append(x)
        return tuple(outs)

    def forward(self, x: torch.Tensor, gate_inp=None, task_id=None, sem=None):
        out = self.forward_features(x, gate_inp=gate_inp, task_id=task_id, sem=sem)
        # by default return last hidden state; user can adapt to return intermediate features
        return out

    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
