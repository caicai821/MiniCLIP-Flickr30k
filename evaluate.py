"""图文检索评估：报告 i2t / t2i 的 R@1、R@5、R@10。

用法:
    python evaluate.py --checkpoint checkpoints/miniclip/best.pt --split test

指标含义（图文检索的标准指标 Recall@K）：
  - t2i（文搜图）：给定一句描述，在所有图里检索，正确的图出现在前 K 个的比例
  - i2t（图搜文）：给定一张图，在所有句子里检索，5 条正确描述中有任一条在前 K 的比例
"""
import argparse

import torch
from torch.utils.data import DataLoader

from data import ImageTextDataset, caption_collate, load_flickr30k
from model import MiniCLIP


def _autocast(device):
    if device.startswith("cuda"):
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return torch.amp.autocast("cpu", enabled=False)


@torch.no_grad()
def encode_images(model, dataset, device, batch_size=256, num_workers=8):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        collate_fn=caption_collate)
    feats, cap_lists = [], []
    for batch in loader:
        with _autocast(device):
            f = model.encode_image(batch["pixel_values"].to(device, non_blocking=True))
        feats.append(f.float().cpu())
        cap_lists.extend(batch["captions"])
    return torch.cat(feats, 0), cap_lists


@torch.no_grad()
def encode_texts(model, captions, device, tokenizer, max_length=32, chunk=512):
    feats = []
    for i in range(0, len(captions), chunk):
        tok = tokenizer(captions[i:i + chunk], padding="max_length", truncation=True,
                        max_length=max_length, return_tensors="pt")
        with _autocast(device):
            f = model.encode_text(tok["input_ids"].to(device), tok["attention_mask"].to(device))
        feats.append(f.float().cpu())
    return torch.cat(feats, 0)


def _recall_single_target(sims, targets, ks=(1, 5, 10)):
    """sims: (Q, N)；targets: (Q,) 每个 query 唯一正确答案的列号。"""
    topk = sims.topk(max(ks), dim=1).indices                     # (Q, maxk)
    return {k: (topk[:, :k] == targets.unsqueeze(1)).any(1).float().mean().item() for k in ks}


def _recall_multi_target(sims, targets, ks=(1, 5, 10)):
    """sims: (Q, N)；targets: (Q, T) 每个 query 有 T 个可接受答案（命中任一即算对）。"""
    topk = sims.topk(max(ks), dim=1).indices                     # (Q, maxk)
    return {k: (topk[:, :k].unsqueeze(-1) == targets.unsqueeze(1)).any(dim=(1, 2)).float().mean().item()
            for k in ks}


@torch.no_grad()
def evaluate_retrieval(model, dataset, device, batch_size=256, num_workers=8,
                       max_length=32, ks=(1, 5, 10)):
    model.eval()
    image_feats, cap_lists = encode_images(model, dataset, device, batch_size, num_workers)

    lengths = {len(caps) for caps in cap_lists}
    if len(lengths) != 1:
        raise ValueError(f"每张图的 caption 数量不一致：{sorted(lengths)}，无法构建标准检索评估")
    n_caps = lengths.pop()

    captions = [c for caps in cap_lists for c in caps]
    text2img = torch.tensor([i for i, caps in enumerate(cap_lists) for _ in caps])
    img2text = torch.arange(len(captions)).view(-1, n_caps)

    text_feats = encode_texts(model, captions, device, dataset.tokenizer, max_length)

    sims_t2i = text_feats @ image_feats.t()          # (Q_text, N_img)
    sims_i2t = image_feats @ text_feats.t()          # (N_img, Q_text)

    results = {}
    results.update({f"t2i_R@{k}": v for k, v in _recall_single_target(sims_t2i, text2img, ks).items()})
    results.update({f"i2t_R@{k}": v for k, v in _recall_multi_target(sims_i2t, img2text, ks).items()})
    return results


def format_results(results):
    lines = ["", f"{'指标':<12}{'结果':>10}", "-" * 24]
    for key in ["t2i_R@1", "t2i_R@5", "t2i_R@10", "i2t_R@1", "i2t_R@5", "i2t_R@10"]:
        lines.append(f"{key:<12}{results[key] * 100:>9.2f}%")
    return "\n".join(lines)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location="cpu")
    model = MiniCLIP(**ckpt["model_args"])
    model.load_state_dict(ckpt["model"])
    return model.to(device), ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--max_samples", type=int, default=None, help="只评估前 N 个样本（快速检查）")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--max_length", type=int, default=32)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_checkpoint(args.checkpoint, device)

    hf_ds = load_flickr30k(split=args.split, max_samples=args.max_samples)
    dataset = ImageTextDataset(hf_ds, image_size=args.image_size,
                               max_length=args.max_length, train=False)

    results = evaluate_retrieval(model, dataset, device,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 max_length=args.max_length)
    print(format_results(results))


if __name__ == "__main__":
    main()
