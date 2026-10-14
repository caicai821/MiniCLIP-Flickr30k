"""MiniCLIP：双塔对比学习的图文融合模型（CLIP 的教学版实现）

结构（对应论文 CLIP, Radford et al., 2021）：
    图像 -> ViT-B/16 (ImageNet-21k 预训练) -> [CLS] -> 线性投影 -> L2 归一化
    文本 -> DistilBERT (预训练)            -> 掩码平均池化 -> 线性投影 -> L2 归一化
    融合方式 = 对比融合：两个模态被拉进同一个 512 维语义空间，
    配对的（图，文）余弦相似度拉大，不配对的压小（InfoNCE / 对称交叉熵）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class MiniCLIP(nn.Module):
    def __init__(self,
                 image_encoder_name="google/vit-base-patch16-224-in21k",
                 text_encoder_name="distilbert-base-uncased",
                 embed_dim=512,
                 init_temperature=0.07):
        super().__init__()
        self.image_encoder_name = image_encoder_name
        self.text_encoder_name = text_encoder_name
        self.embed_dim = embed_dim

        self.image_encoder = AutoModel.from_pretrained(image_encoder_name)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name)
        self.image_proj = nn.Linear(self.image_encoder.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, embed_dim)
        # 可学习的温度系数：logit_scale = 1/tau，CLIP 初始化为 1/0.07
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / init_temperature).log())

    def encode_image(self, pixel_values):
        # interpolate_pos_encoding=True：位置编码按新分辨率插值。
        # ViT 权重按 224(14x14 patch) 训练，输入 384 时 patch 数变成 24x24，
        # 不插值会报 "Input image size (384*384) doesn't match model (224*224)"。
        # 224 输入时插值是无操作，所以这个开关对所有分辨率都安全。
        cls_feat = self.image_encoder(
            pixel_values=pixel_values, interpolate_pos_encoding=True
        ).last_hidden_state[:, 0]
        return F.normalize(self.image_proj(cls_feat), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-6)  # 掩码平均池化
        return F.normalize(self.text_proj(pooled), dim=-1)

    def forward(self, pixel_values, input_ids, attention_mask):
        image_feats = self.encode_image(pixel_values)
        text_feats = self.encode_text(input_ids, attention_mask)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        return image_feats, text_feats, logit_scale

    def model_args(self):
        return {
            "image_encoder_name": self.image_encoder_name,
            "text_encoder_name": self.text_encoder_name,
            "embed_dim": self.embed_dim,
        }


def contrastive_loss(image_feats, text_feats, logit_scale):
    """对称 InfoNCE：一个 batch 内有 N 个正对，其余 N-1 个为负对。"""
    logits = logit_scale * image_feats @ text_feats.t()      # (N, N)
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, targets)              # 图 -> 文
    loss_t2i = F.cross_entropy(logits.t(), targets)          # 文 -> 图
    return (loss_i2t + loss_t2i) / 2
