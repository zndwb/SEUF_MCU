import torch
import torch.nn as nn
import torch.nn.functional as F

# ============= 基础模块 =============
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, N, C]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return x

# ============= MLP & MoE-MLP =============
class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoEMLP(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=1, mlp_ratio=4):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        hidden_dim = int(dim * mlp_ratio)

        # 多专家 MLP
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim)
            ) for _ in range(num_experts)
        ])
        # 门控网络
        self.gate = nn.Linear(dim, num_experts)

    def forward(self, x):
        # x: [B, N, D]
        gate_logits = self.gate(x)
        gate_probs = torch.softmax(gate_logits, dim=-1)

        # top-1
        topk_val, topk_idx = torch.topk(gate_probs, self.top_k, dim=-1)
        dispatch_mask = torch.nn.functional.one_hot(topk_idx.squeeze(-1), self.num_experts).float()

        # 输出
        out = torch.zeros_like(x)
        idx = topk_idx.squeeze(-1)
        score = topk_val.squeeze(-1).unsqueeze(-1)
        for i in range(self.num_experts):
            mask = (idx == i).float().unsqueeze(-1)
            if mask.sum() > 0:
                out += mask * score * self.experts[i](x)

        # load balance loss
        meangate = gate_probs.mean(dim=(0, 1))
        expert_usage = dispatch_mask.mean(dim=(0, 1))
        load_balance_loss = (self.num_experts * (meangate * expert_usage).sum())
        return out, load_balance_loss

# ============= Transformer Block =============
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, use_moe=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.use_moe = use_moe
        self.mlp = MoEMLP(dim, mlp_ratio=mlp_ratio) if use_moe else MLP(dim, mlp_ratio)

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = h + x

        h = x
        x = self.norm2(x)
        if self.use_moe:
            out, aux_loss = self.mlp(x)
        else:
            out = self.mlp(x)
            aux_loss = torch.tensor(0., device=x.device)
        x = h + out
        return x, aux_loss

# ============= MoE-ViT =============
class MoEVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, use_moe=(i % 2 == 1))
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        aux_loss_total = 0
        for blk in self.blocks:
            x, aux_loss = blk(x)
            aux_loss_total += aux_loss
        x = self.norm(x)
        cls_token = x[:, 0]
        logits = self.head(cls_token)
        return logits, aux_loss_total
