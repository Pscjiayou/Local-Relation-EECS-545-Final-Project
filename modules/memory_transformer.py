import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryAugmentedTransformer(nn.Module):
    def __init__(self, base_model, memory_size=512):
        super().__init__()
        self.base = base_model
        self.memory_size = memory_size
        self.hidden_dim = base_model.vision_width  # 取决于 BLIP 配置中定义的 hidden size

        # 训练中可学习的记忆向量 [memory_size, hidden_dim]
        self.memory_bank = nn.Parameter(torch.randn(memory_size, self.hidden_dim))

        # 投影层（输入+memory concat → 原始维度）
        self.memory_fusion = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def forward(self, video_embeddings, video_mask, vsum_labels, tsum_labels):
        """
        video_embeddings: [B, T, D]
        memory_bank: [M, D]
        """

        # Normalize for dot-product attention
        queries = F.normalize(video_embeddings, dim=-1)     # [B, T, D]
        keys    = F.normalize(self.memory_bank, dim=-1)      # [M, D]

        # Dot product similarity: [B, T, M]
        attn_scores = torch.matmul(queries, keys.t()) / (self.hidden_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)        # [B, T, M]

        # Attention output: weighted memory lookup → [B, T, D]
        memory_output = torch.matmul(attn_weights, self.memory_bank)  # [B, T, D]

        # Concatenate + fuse
        fused = torch.cat([video_embeddings, memory_output], dim=-1)  # [B, T, 2D]
        video_embeddings_enhanced = self.memory_fusion(fused)         # [B, T, D]

        # Feed into base model
        return self.base(video_embeddings_enhanced, video_mask, vsum_labels, tsum_labels)

    def generate(self, video_embeddings, video_mask=None, **kwargs):
        # 1. 做 memory 增强
        queries = F.normalize(video_embeddings, dim=-1)
        keys = F.normalize(self.memory_bank, dim=-1)
        attn_scores = torch.matmul(queries, keys.t()) / (self.hidden_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        memory_output = torch.matmul(attn_weights, self.memory_bank)
        fused = torch.cat([video_embeddings, memory_output], dim=-1)
        video_embeddings_enhanced = self.memory_fusion(fused)
        # 2. 调用 base.generate
        return self.base.generate(video_embeddings_enhanced, video_mask=video_mask, **kwargs)
