"""Flickr30k 图文对数据管道（MiniCLIP 项目）—— 本地文件版（适配离线服务器）

数据来源：nlphuji/flickr30k 的原始文件（不再依赖 datasets 库在线加载）：
  - flickr_annotations_30k.csv（已随项目提供，放在本文件同级目录）：
      每行一张图，共 31014 行，字段：
        raw      = 5 条英文描述（JSON list 字符串）
        split    = Karpathy 官方划分：train 29000 / val 1014 / test 1000（图片级）
        filename = 图片文件名（如 1000092795.jpg）
        img_id / sentids = 图片与句子编号
  - flickr30k-images/：图片目录（由 flickr30k-images.zip 解压得到）

图片目录解析优先级：
  1. 环境变量 FLICKR30K_IMAGES 指向的目录（推荐在服务器上用这个）
  2. 本文件同级的 flickr30k-images/

设计要点：不预先解码全部图片（省内存），__getitem__ 里按需解码，
由 DataLoader 的 num_workers 并行处理。
"""
import ast
import os
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent
ANNOTATION_CSV = PROJECT_ROOT / "flickr_annotations_30k.csv"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _images_dir():
    env = os.environ.get("FLICKR30K_IMAGES")
    if env:
        return Path(env)
    return PROJECT_ROOT / "flickr30k-images"


class Flickr30kImages:
    """轻量图片数据表：支持 len() 与索引访问，图片在取用时才解码（省内存）。"""

    def __init__(self, df: pd.DataFrame, images_dir: Path):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        from PIL import Image

        row = self.df.iloc[idx]
        image = Image.open(self.images_dir / row["filename"]).convert("RGB")
        return {
            "image": image,
            "caption": ast.literal_eval(row["raw"]),
            "img_id": int(row["img_id"]),
            "filename": row["filename"],
        }


def load_flickr30k(split="train", max_samples=None, seed=42):
    """加载 Flickr30k 指定划分，返回 Flickr30kImages（惰性解码图片）。

    Args:
        split: "train" / "val" / "test"，传 None 则不筛选。
        max_samples: 只取这么多条（冒烟测试 / 小规模实验用）。
    """
    images_dir = _images_dir()
    if not images_dir.exists():
        raise FileNotFoundError(
            f"找不到图片目录 {images_dir}。请先解压 flickr30k-images.zip，"
            f"或用环境变量 FLICKR30K_IMAGES 指向图片目录"
        )
    df = pd.read_csv(ANNOTATION_CSV)
    if split is not None:
        available = sorted(set(df["split"]))
        if split not in available:
            raise ValueError(f"split 取值为 {available}，不包含 '{split}'")
        df = df[df["split"] == split]

    if max_samples is not None and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=seed).sort_index()

    print(f"[data] split={split} 样本数={len(df)} 图片目录={images_dir}")
    return Flickr30kImages(df, images_dir)


def caption_collate(batch):
    """DataLoader 自定义 collate：保持 captions 为 [批内每图的5句列表] 结构。

    必须用它的原因：PyTorch 默认 collate 会把 captions（list of str 的序列）
    按位置转置成 [5个位置][批大小句]，导致 evaluate 里 img2text 检索矩阵
    行数错位（RuntimeError: size of tensor a (N_img) must match (批数x5)）。
    """
    from torch.utils.data._utils.collate import default_collate

    out = {k: default_collate([b[k] for b in batch])
           for k in batch[0] if k != "captions"}
    out["captions"] = [b["captions"] for b in batch]
    return out


class ImageTextDataset(Dataset):
    """每条样本 = 一张图 + 一条文本（训练时每 epoch 随机抽 1 条 caption）。"""

    def __init__(self, flickr_ds, tokenizer_name="distilbert-base-uncased",
                 image_size=224, max_length=32, train=True):
        self.ds = flickr_ds
        self.train = train
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        if train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0), antialias=True),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            resize_to = int(image_size * 256 / 224)
            self.transform = transforms.Compose([
                transforms.Resize(resize_to, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        pixel_values = self.transform(row["image"])
        captions = list(row["caption"])
        text = random.choice(captions) if self.train else captions[0]
        tok = self.tokenizer(
            text, padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
            "captions": captions,
            "index": idx,
        }
