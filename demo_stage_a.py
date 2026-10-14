"""Stage A：不训练，先立即看到多模态效果（建立信心用）

演示两件事，正好对应你的方向「检索 + 生成」：
  1) BLIP 图生文：给一张图，生成自然语言描述（生成）
  2) CLIP zero-shot 文搜图：给一句文本，在图库里检索最相关的图（检索）

用法:
    python demo_stage_a.py --query "a dog running on the beach"
    python demo_stage_a.py --images cat.jpg dog.jpg --query "a cat sitting on the sofa"

不传 --images 时，会自动从 Flickr30k 抽 8 张样例图存到 samples/。
"""
import argparse
import os
import random

import torch
from PIL import Image
from transformers import (BlipForConditionalGeneration, BlipProcessor,
                          CLIPModel, CLIPProcessor)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_sample_images(n=8, out_dir="samples", seed=7):
    from data import load_flickr30k
    os.makedirs(out_dir, exist_ok=True)
    hf_ds = load_flickr30k("test", max_samples=None)
    idxs = random.Random(seed).sample(range(len(hf_ds)), n)
    paths = []
    for i, idx in enumerate(idxs):
        row = hf_ds[idx]
        path = os.path.join(out_dir, f"sample_{i}_{row['filename']}")
        row["image"].convert("RGB").save(path)
        paths.append(path)
        print(f"  保存样例图: {path}")
    return paths


def caption_images(image_paths):
    print("\n===== 1) BLIP 图生文（生成）=====")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base").to(DEVICE)
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        print(f"  [{os.path.basename(path)}] -> {processor.decode(out[0], skip_special_tokens=True)}")
    del model
    torch.cuda.empty_cache()


def zero_shot_retrieve(image_paths, query, topk=3):
    print(f"\n===== 2) CLIP zero-shot 文搜图（检索）：query = “{query}” =====")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    images = [Image.open(p).convert("RGB") for p in image_paths]
    img_in = processor(images=images, return_tensors="pt").to(DEVICE)
    txt_in = processor(text=[query], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        img_feat = model.get_image_features(**img_in)
        txt_feat = model.get_text_features(**txt_in)
    img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

    sims = (img_feat @ txt_feat.t()).squeeze(1).cpu()
    order = sims.argsort(descending=True)[:topk]
    for rank, i in enumerate(order, 1):
        print(f"  top{rank}: {os.path.basename(image_paths[i])}  相似度={sims[i]:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=None, help="自定义图片路径")
    ap.add_argument("--query", default="a group of people standing outdoors")
    ap.add_argument("--n_samples", type=int, default=8)
    args = ap.parse_args()

    image_paths = args.images if args.images else get_sample_images(args.n_samples)
    caption_images(image_paths)
    zero_shot_retrieve(image_paths, args.query)

    print("\n提示：换个 --query 多试几句（英文），观察排序变化 —— 这就是图文对齐的效果。")
    print("下一步：回到 README.md 的 Stage B，自己训练一个这样的对齐模型。")


if __name__ == "__main__":
    main()
