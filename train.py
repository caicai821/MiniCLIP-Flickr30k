"""训练 MiniCLIP：单卡 A100 即可完成的图文对比学习（模态融合）训练。

推荐流程（详见 README.md）：
1. 冒烟测试（几百条样本、1 个 epoch，确认 loss 下降）：
     python train.py --max_train_samples 512 --val_samples 128 --epochs 1 --batch_size 64
2. 全量训练：
     python train.py --epochs 5 --batch_size 128
"""
import argparse
import math
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import ImageTextDataset, load_flickr30k
from evaluate import evaluate_retrieval, format_results
from model import MiniCLIP, contrastive_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def warmup_cosine_lambda(warmup_steps, total_steps):
    def fn(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn


def main():
    ap = argparse.ArgumentParser()
    # --- 数据 ---
    ap.add_argument("--max_train_samples", type=int, default=None, help="冒烟测试用，限制训练样本数")
    ap.add_argument("--val_samples", type=int, default=500, help="每个 epoch 用多少 val 样本算 R@K")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--max_length", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    # --- 模型 ---
    ap.add_argument("--image_encoder", default="google/vit-base-patch16-224-in21k")
    ap.add_argument("--text_encoder", default="distilbert-base-uncased")
    ap.add_argument("--embed_dim", type=int, default=512)
    # --- 优化 ---
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr_encoders", type=float, default=2e-5, help="预训练编码器用小学习率")
    ap.add_argument("--lr_heads", type=float, default=1e-4, help="投影层用较大学习率")
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="checkpoints/miniclip")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device.startswith("cuda")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[train] device={device}  torch={torch.__version__}")

    # ---------- 数据 ----------
    train_ds = ImageTextDataset(
        load_flickr30k("train", max_samples=args.max_train_samples, seed=args.seed),
        image_size=args.image_size, max_length=args.max_length, train=True)
    val_ds = ImageTextDataset(
        load_flickr30k("val", max_samples=args.val_samples, seed=args.seed),
        image_size=args.image_size, max_length=args.max_length, train=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0)

    # ---------- 模型与优化器 ----------
    model = MiniCLIP(args.image_encoder, args.text_encoder, embed_dim=args.embed_dim).to(device)
    optimizer = torch.optim.AdamW([
        {"params": list(model.image_encoder.parameters()) + list(model.text_encoder.parameters()),
         "lr": args.lr_encoders},
        {"params": list(model.image_proj.parameters()) + list(model.text_proj.parameters())
                   + [model.logit_scale], "lr": args.lr_heads},
    ], weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine_lambda(warmup_steps, total_steps))

    # ---------- 训练 ----------
    best_r1 = -1.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                image_feats, text_feats, logit_scale = model(
                    pixel_values, input_ids, attention_mask)
                loss = contrastive_loss(image_feats, text_feats, logit_scale)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            running += loss.item()

            if step % 20 == 0 or step == len(train_loader):
                print(f"[epoch {epoch}] step {step}/{len(train_loader)} "
                      f"loss={running / step:.4f} "
                      f"logit_scale={model.logit_scale.exp().item():.2f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")

        # ---------- 验证 ----------
        results = evaluate_retrieval(model, val_ds, device,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     max_length=args.max_length)
        print(f"\n[epoch {epoch}] 平均训练 loss = {running / len(train_loader):.4f}")
        print(format_results(results))

        ckpt = {"model": model.state_dict(),
                "model_args": model.model_args(),
                "epoch": epoch,
                "val_results": results,
                "train_args": vars(args)}
        torch.save(ckpt, os.path.join(args.output_dir, "last.pt"))
        if results["t2i_R@1"] > best_r1:
            best_r1 = results["t2i_R@1"]
            torch.save(ckpt, os.path.join(args.output_dir, "best.pt"))
            print(f"[checkpoint] t2i_R@1 提升到 {best_r1 * 100:.2f}%，已保存 best.pt\n")
        else:
            print()

    print(f"[done] 最优 t2i_R@1 = {best_r1 * 100:.2f}%，模型保存在 {args.output_dir}/")


if __name__ == "__main__":
    main()
