"""Gradio Demo：把你训练好的 MiniCLIP 变成可交互的检索系统

两个页面：
  1) 文搜图：输入英文描述 -> 从测试集图库中检索 top-6 图像
  2) 图生文：上传一张图 -> BLIP 生成描述 + MiniCLIP 检索图库中最相似的图

用法:
    python demo_app.py --checkpoint checkpoints/miniclip/best.pt --port 7860
然后在本地浏览器做端口转发（AI STATION 是远程机器）：
    ssh -L 7860:localhost:7860 <你的用户名>@<AI STATION 地址>
再打开 http://localhost:7860
"""
import argparse

import gradio as gr
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from data import ImageTextDataset, load_flickr30k
from evaluate import load_checkpoint
from model import MiniCLIP

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 32


class SearchEngine:
    def __init__(self, checkpoint, gallery_size=1000, image_size=224):
        self.model, _ = load_checkpoint(checkpoint, DEVICE)
        self.model.eval()
        self.dataset = ImageTextDataset(
            load_flickr30k("test", max_samples=gallery_size),
            image_size=image_size, max_length=MAX_LENGTH, train=False)
        print("[demo] 正在构建图库向量 ...")
        self.gallery_images = []
        self.gallery_captions = []
        feats = []
        with torch.no_grad():
            for i in range(len(self.dataset)):
                row = self.dataset.ds[i]
                self.gallery_images.append(row["image"].convert("RGB"))
                self.gallery_captions.append(row["caption"][0])
                pixel = self.dataset[i]["pixel_values"].unsqueeze(0).to(DEVICE)
                feats.append(self.model.encode_image(pixel).cpu())
        self.image_feats = torch.cat(feats, 0)
        self.tokenizer = self.dataset.tokenizer
        self.blip_processor = None
        self.blip_model = None
        print(f"[demo] 图库就绪：{len(self.gallery_images)} 张图")

    @torch.no_grad()
    def search_by_text(self, query, topk=6):
        if not query or not query.strip():
            return []
        tok = self.tokenizer(query.strip(), return_tensors="pt", truncation=True,
                             max_length=MAX_LENGTH, padding="max_length")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=DEVICE.startswith("cuda")):
            q = self.model.encode_text(tok["input_ids"].to(DEVICE),
                                       tok["attention_mask"].to(DEVICE)).cpu()
        sims = (self.image_feats @ q.t()).squeeze(1)
        top = sims.argsort(descending=True)[:topk]
        return [(self.gallery_images[i], f"{sims[i]:.3f} | {self.gallery_captions[i]}") for i in top]

    @torch.no_grad()
    def caption_and_search(self, image, topk=6):
        if image is None:
            return "请先上传一张图片", []
        if self.blip_model is None:
            self.blip_processor = BlipProcessor.from_pretrained(
                "Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base").to(DEVICE).eval()
        inputs = self.blip_processor(image, return_tensors="pt").to(DEVICE)
        out = self.blip_model.generate(**inputs, max_new_tokens=30)
        caption = self.blip_processor.decode(out[0], skip_special_tokens=True)

        pixel = self.dataset.transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                enabled=DEVICE.startswith("cuda")):
            q = self.model.encode_image(pixel).cpu()
        sims = (self.image_feats @ q.t()).squeeze(1)
        top = sims.argsort(descending=True)[:topk]
        results = [(self.gallery_images[i], f"{sims[i]:.3f} | {self.gallery_captions[i]}") for i in top]
        return f"BLIP 生成的描述：{caption}", results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/miniclip/best.pt")
    ap.add_argument("--gallery_size", type=int, default=1000)
    ap.add_argument("--image_size", type=int, default=224,
                    help="须与训练时一致：384 模型请传 --image_size 384")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    engine = SearchEngine(args.checkpoint, args.gallery_size, args.image_size)

    with gr.Blocks(title="MiniCLIP 图文检索") as demo:
        gr.Markdown("# MiniCLIP 图文检索 Demo\n输入英文描述检索图像，或上传图像看生成描述与相似图。")
        with gr.Tab("文搜图（文本 -> 图像）"):
            text_input = gr.Textbox(label="英文查询，例如：a dog running on the beach")
            text_btn = gr.Button("检索", variant="primary")
            text_gallery = gr.Gallery(label="Top-6 检索结果", columns=3, height=420)
            text_btn.click(engine.search_by_text, inputs=text_input, outputs=text_gallery)
        with gr.Tab("图生文（图像 -> 文本 + 相似图）"):
            image_input = gr.Image(type="pil", label="上传一张图片")
            image_btn = gr.Button("生成描述并检索", variant="primary")
            caption_output = gr.Textbox(label="生成结果")
            image_gallery = gr.Gallery(label="图库中最相似的图", columns=3, height=420)
            image_btn.click(engine.caption_and_search, inputs=image_input,
                            outputs=[caption_output, image_gallery])

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
