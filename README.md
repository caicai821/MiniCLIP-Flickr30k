# MiniCLIP：从零跑通图文检索与生成（模态融合入门必跑项目）

> 目标读者：人工智能研 0 / 刚学完 Transformer、多模态基础薄弱
> 硬件：AI STATION，单卡 A100 40GB（本项目全程单卡即可，无需分布式）
> 方向对齐：多模态**检索**（文搜图 / 图搜文）+ **生成**（图生文）+ **模态融合**（对比学习双塔）

---

## 0. 一页总览：你到底要跑一个什么东西

一句话：**用 Flickr30k 数据集（3.1 万张图，每张 5 条英文描述），训练一个"图和文映射到同一语义空间"的模型 CLIP，然后用它做文搜图 / 图搜文，最后接上 BLIP 做图生文。**

这一个项目同时覆盖你方向的三个关键词：

| 你的方向关键词 | 本项目的对应物 | 代码位置 |
|---|---|---|
| 模态融合 | **对比融合（双塔 + InfoNCE）**：ViT 看图、DistilBERT 读句，把两者压进同一个 512 维空间 | [model.py](model.py) |
| 检索 | 训练完后一句英文描述 → 从 3 万张图里搜出最相关的图，用 R@1/5/10 打分 | [evaluate.py](evaluate.py)、[demo_app.py](demo_app.py) |
| 生成 | BLIP 图生文：给一张图生成自然语言描述（生成方向的最小闭环） | [demo_stage_a.py](demo_stage_a.py) |

**为什么是 CLIP？** 它是 2021 年 OpenAI 的工作，至今仍是多模态检索和所有多模态大模型（BLIP、BLIP-2、LLaVA……）的地基。学到的东西不会过时，而且它足够小、足够经典，单卡 A100 能训练、能评估、能部署 demo，是公认的"多模态第一课"。把 CLIP 吃透，你再看实验室任何多模态论文都会觉得眼熟。

**这个项目你会亲手搞懂：**
1. Transformer 编码器怎么分别处理图像（ViT）和文本（BERT 系）；
2. "模态融合"最基础的范式——对比学习（InfoNCE 损失、温度系数、batch 内负样本）；
3. 图文检索的完整训练/评估闭环（R@K 指标）；
4. 从数据管道 → 训练 → 评估 → Demo 的端到端工程能力。

**预期时间投入**：环境 + Stage A 一个下午内可看到效果；Stage B 训练在 A100 上单卡数轮即可完成（每轮的耗时取决于数据加载，属于"一晚跑完"的量级）。

---

## 1. 项目结构

```
MiniCLIP-Flickr30k/
├── README.md          <- 本指南（端到端说明书）
├── requirements.txt   <- 依赖清单
├── data.py            <- 数据管道：Flickr30k 加载、图文对 Dataset、数据增强
├── model.py           <- 模型：MiniCLIP 双塔 + 对比损失（核心学习对象）
├── train.py           <- 训练脚本：混合精度、学习率调度、逐 epoch 验证、存 checkpoint
├── evaluate.py        <- 评估脚本：i2t / t2i 的 R@1、R@5、R@10
├── demo_stage_a.py    <- Stage A：零训练的即时效果（BLIP 图生文 + CLIP zero-shot 检索）
└── demo_app.py        <- Stage C：Gradio 网页 Demo（文搜图 + 图生文）
```

代码总量约 500 行，每一段都建议亲自读一遍——**读代码本身是这个项目学习价值的一半**。

---

## 2. 环境准备（AI STATION）

以下命令在 AI STATION 的终端（Linux）执行。

**你选的镜像 `pytorch:25.03-py3` 是 NVIDIA NGC 官方 PyTorch 容器（实验室私有仓库转存），里面已经自带 Python 3.x、PyTorch 2.x、CUDA、torchvision、pillow、numpy——不需要建 conda 环境，也不需要再装 PyTorch。** 只补装项目额外用到的几个库即可：

```bash
# 1) 验证镜像自带的 PyTorch 能看到 A100（应该直接成功）
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望输出：2.x True NVIDIA A100-SXM4-40GB（或类似）

# 2) 只补装镜像里没有的库（transformers / datasets / tqdm / gradio）
#    若报 externally-managed-environment（Ubuntu 24.04 的 PEP 668 保护），加 --break-system-packages：
pip install -r requirements.txt --break-system-packages
# 或者一次性免除后续反复加参数：export PIP_BREAK_SYSTEM_PACKAGES=1
# requirements 里已满足的包（torch、torchvision 等）会被 pip 自动跳过，不会覆盖镜像自带版本

# 3) 再验证一次关键依赖
python -c "import transformers, datasets, gradio; print('deps OK')"
```

> **补充说明**
> - 上面 `--break-system-packages` 是安全的：容器镜像用完即弃，不存在"破坏系统 Python"的问题。
>   如果偏好虚拟环境，可用 `python3 -m venv --system-site-packages ~/miniclip-env && source ~/miniclip-env/bin/activate`
>   （注意 `--system-site-packages` 必须加，否则 venv 里看不到镜像自带的 PyTorch）。
> - 容器重建/换镜像后 pip 装的库会丢失，重跑一次 pip install 即可；
>   如果 AI STATION 的家目录/数据盘有持久化挂载，建议把数据集缓存也放过去：
>   `export HF_HOME=/你的持久化目录/hf_cache`（否则容器重建后几 GB 的 Flickr30k 要重新下载）。

**国内网络加速（重要）**：数据和模型权重都从 HuggingFace Hub 下载，建议设置镜像，否则容易卡在下载：

```bash
export HF_ENDPOINT=https://hf-mirror.com
# 想长期生效就写进 ~/.bashrc
```

### 2.5 离线部署（实验室服务器无法联网时的完整方案）

需要上传 **3 个文件**（都已在本地电脑桌面备好）：

| 文件 | 大小 | 作用 |
|---|---|---|
| `miniclip-wheels.zip` | 178 MB | 全部 pip 依赖的离线安装包（已补齐 Linux 专属依赖 hf-xet，并模拟 Linux/Py3.12 验证过解析） |
| `hf_models.tar.gz` | 2.6 GB | 4 个模型权重的 HF 缓存（ViT / DistilBERT / BLIP / CLIP），已离线加载验证 |
| `flickr30k-images.zip` | 4.1 GB | Flickr30k 图片包（标注 csv 已随项目代码提供） |

上传后（JupyterLab 拖拽或 scp），在服务器上按顺序执行（假设都放在 `/cairf`）：

```bash
cd /cairf

# ① 离线安装 pip 依赖（--no-index 表示完全不用联网）
rm -rf miniclip-wheels && mkdir miniclip-wheels
python -m zipfile -e miniclip-wheels.zip /cairf/miniclip-wheels/
ls miniclip-wheels/*.whl | head -3        # 必须能看到一堆 .whl 文件
export PIP_BREAK_SYSTEM_PACKAGES=1
pip install --no-index --find-links /cairf/miniclip-wheels \
    --break-system-packages -r MiniCLIP-Flickr30k/requirements.txt
python -c "import transformers, pandas, gradio; print('deps OK')"

# ② 部署模型缓存 + 解压数据
mkdir -p /cairf/hf_cache
tar -xzf hf_models.tar.gz -C /cairf       # 得到 /cairf/hf_cache/hub/models--*
python -m zipfile -e flickr30k-images.zip /cairf/MiniCLIP-Flickr30k/   # 得到图片目录

# ③ 环境变量（建议追加到 ~/.bashrc）
export HF_HOME=/cairf/hf_cache
export HF_HUB_OFFLINE=1                   # 强制离线：模型只从本地缓存读
export PIP_BREAK_SYSTEM_PACKAGES=1

# ④ 最终验证
cd /cairf/MiniCLIP-Flickr30k
python -c "
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained('distilbert-base-uncased'); print('模型缓存 OK')
from data import load_flickr30k
print(load_flickr30k('val', max_samples=4)[0]['image'].size, '<- 数据 OK')
"
```

> 注意：模型包是 tar.gz（内含符号链接，必须用 `tar` 解压，不能用 python zipfile）；
> 另两个 zip 无符号链接，用 `python -m zipfile -e` 或 `unzip` 都可以。
> 容器重建后：pip 装的库和环境变量会丢，重跑 ①③ 两条命令即可；`/cairf` 若是持久化挂载，
> 模型与数据只需部署一次。

**把代码传上服务器**（三选一）：

```bash
# 方式 A：scp 整个文件夹
scp -r MiniCLIP-Flickr30k 用户名@AI站地址:~/

# 方式 B：先 push 到自己的 GitHub/Gitee，再在服务器 git clone

# 方式 C：JupyterLab 界面直接拖拽上传文件夹
```

---

## 3. 数据准备：Flickr30k

本项目使用 **Flickr30k 原始文件**（来自 HuggingFace 数据集 `nlphuji/flickr30k`）：

- `flickr_annotations_30k.csv`：**已随项目提供**（项目根目录），每行一张图、5 条英文描述，
  `split` 列就是学界通用的 **Karpathy 官方划分**：train 29000 / val 1014 / test 1000（图片级）——
  论文里报 R@K 用的就是这套划分，你测出来的数字可以直接和别人对比；
- `flickr30k-images.zip`（约 4.1 GB）：图片压缩包，解压出 `flickr30k-images/` 目录（31783 张 jpg）。

**部署数据**（内网服务器上，在项目根目录执行）：

```bash
# 假设 flickr30k-images.zip 已上传到项目根目录（或 /cairf，自行调整路径）
cd ~/MiniCLIP-Flickr30k
python -m zipfile -e flickr30k-images.zip .
# 解压后得到 flickr30k-images/ 目录；__MACOSX/ 是 macOS 垃圾文件，可删可留
ls flickr30k-images | head -3        # 应看到若干 .jpg
```

> 如果 zip 放在别的位置，解压后也可以用环境变量指路：`export FLICKR30K_IMAGES=/图片目录`。
> `data.py` 会按优先级找图：环境变量 `FLICKR30K_IMAGES` > 项目根目录下的 `flickr30k-images/`。

**数据"验货"**（确认图片、标注、划分都对上了）：

```bash
python -c "
from data import load_flickr30k
for split in ['train', 'val', 'test']:
    ds = load_flickr30k(split)
    row = ds[0]
    print(split, '| 字段:', list(row.keys()))
    print('  caption 数量:', len(row['caption']), '| 第一句:', row['caption'][0])
"
```

期望输出：train 29000 条、val 1014 条、test 1000 条；每张图 5 条 caption，字段含 image/caption/img_id/filename。

---

## 4. Stage A：先看到效果，再理解原理（强烈建议先跑）

在训练任何东西之前，先用**预训练模型**亲眼看到"文搜图"和"图生文"长什么样——这是建立信心最快的一步。

```bash
python demo_stage_a.py --query "a dog running on the beach"
```

它会：
1. 自动从 Flickr30k 抽 8 张图存到 `samples/`；
2. 用 **BLIP** 给每张图生成英文描述（你会看到生成模型的输出）；
3. 用 **CLIP（zero-shot，完全没在 Flickr30k 上训练过）** 对你的 query 打分排序，打印 top-3 最相似的图。

多换几个 query 观察排序变化，比如 `"two men in green shirts"`、`"a child playing with a ball"`。**感受一下：一个没在该数据集训练过的模型，已经能做语义级检索了——这就是"图文对齐"的威力。**

如果想用自己的图片：`python demo_stage_a.py --images my1.jpg my2.jpg --query "..."`。

---

## 5. Stage B：核心训练（学习价值的主体）

### 5.1 先读懂模型（对照你已学的 Transformer）

打开 [model.py](model.py)（只有 60 行），对照下表理解：

| 组件 | 是什么 | 对应论文 |
|---|---|---|
| `image_encoder` = ViT-B/16 | 图切成 16×16 小块 → 当作 token 序列 → 过 Transformer 编码器 → 取 [CLS] | ViT (Dosovitskiy, 2020) |
| `text_encoder` = DistilBERT | 词 → token 序列 → Transformer 编码器 → 掩码平均池化 | BERT / DistilBERT |
| `image_proj` / `text_proj` | 各一个线性层，把 768 维压到同一 512 维空间 | CLIP |
| `logit_scale` | 可学习温度 τ，控制 softmax 的尖锐程度 | CLIP |
| `contrastive_loss` | 对称 InfoNCE：batch 内 N 个正对，其余 N−1 个当负样本 | CLIP |

关键直觉：**"模态融合"在这里不是把两个模态的特征拼在一起，而是把它们"对齐"到同一个空间**——让匹配的图文余弦相似度接近 1，不匹配的接近 0。这正是"对比融合"（contrastive fusion），是后续 BLIP（cross-attention 融合）、BLIP-2（Q-Former 融合）的基础对照物。

**训练前的数值预期（重要 sanity check）**：随机初始化时，InfoNCE 的 loss ≈ ln(batch_size)。batch=64 时初始 loss ≈ 4.16，batch=128 时 ≈ 4.85。训练后 loss 应持续下降，最终通常降到 1 上下或更低。**loss 不低于 ln(N) 就说明训练没跑通。**

### 5.2 冒烟测试（先用小数据确认一切正常）

```bash
python train.py --max_train_samples 512 --val_samples 128 --epochs 1 --batch_size 64
```

通过标准：
- 每 20 步打印的 loss 从 ~4.2 明显下降（几百条样本就能过拟合到 loss<1，这是正常的、也是目的）；
- epoch 末尾打印出 R@1/5/10 的表格（小数据上 R@1 可能只有百分之几，正常）；
- `checkpoints/miniclip/` 下出现 `last.pt`、`best.pt`。

### 5.3 全量训练

```bash
export HF_ENDPOINT=https://hf-mirror.com     # 如果还没设置
python train.py --epochs 5 --batch_size 128
```

脚本会自动完成：混合精度（bf16，A100 原生支持）→ AdamW 双学习率（编码器 2e-5、投影头 1e-4）→ warmup+cosine 学习率 → 每个 epoch 在 val 集上算 R@K → 保存最优 checkpoint。

**A100 40GB 参数建议：**
- `--batch_size 128` 稳妥；显存有富余可以试 `256`（对比学习中 batch 越大、负样本越多，效果通常越好）；
- 若 OOM：`--batch_size 64`，或 `--image_size 192`；
- 想更充分训练：`--epochs 10`。

### 5.4 独立评估（测试集 R@K）

```bash
python evaluate.py --checkpoint checkpoints/miniclip/best.pt --split test
```

输出形如：

```
指标            结果
------------------------
t2i_R@1       xx.xx%
t2i_R@5       xx.xx%
t2i_R@10      xx.xx%
i2t_R@1       xx.xx%
i2t_R@5       xx.xx%
i2t_R@10      xx.xx%
```

**怎么看这些数字：**
- 随机基线：1000 张图里瞎猜，t2i R@1 ≈ 0.1%；你的模型应远高于此（收敛后 R@1 通常在百分之二三十到五十的量级，取决于训练轮数和是否充分解冻编码器）；
- 规律应满足 R@1 < R@5 < R@10；
- **不要和论文 SOTA 比绝对值**（它们用了更大的模型/更多的数据/更多技巧），看的是"训练是否有效、是否逐 epoch 提升"；
- 建议每跑一次就记录一行（epoch 数、batch、R@1），对比不同超参——这是你研究生阶段做实验记录的第一课。

---

## 6. Stage C：把它变成可用的检索系统（Gradio Demo）

```bash
python demo_app.py --checkpoint checkpoints/miniclip/best.pt --port 7860
```

因为模型在远程 AI STATION 上，本地浏览器需要做**端口转发**：

```bash
# 在你自己电脑的终端执行（不是 AI STATION 上）
ssh -L 7860:localhost:7860 用户名@AI站地址
```

然后本地打开 `http://localhost:7860`，你会看到两个页面：
- **文搜图**：输入英文描述 → 检索 top-6 图像（用的是你训练的模型）；
- **图生文**：上传任意图片 → BLIP 生成描述 + 检索图库中最相似的图。

这一步做完，你就拥有了一个可以演示给导师/同学看的完整多模态检索系统。

---

## 7. Stage D（选修）：走向"生成"——BLIP 图文生成

Stage A 里 BLIP 只是拿来推理。如果你想往"生成"方向深入，下一步是**微调 BLIP 做图像描述生成**，对应"检索+生成"里的生成主线：

1. 数据复用：Flickr30k 的 5 条 caption 就是现成的 (image → text) 生成训练对；
2. 模型：`Salesforce/blip-image-captioning-base`（transformers 库直接加载）；
3. 训练目标：自回归语言建模 loss（给定图像，最大化 caption 的似然）——与 Stage B 的对比损失形成对照：**对比融合 vs 跨注意力融合**；
4. 评估指标：CIDEr / BLEU / METEOR（`pycocoevalcap` 库）。

这一阶段建议在吃透 Stage B 之后再做，做完你就完整走过"检索 + 生成"两条线，正是你课题的缩影。

---

## 8. 排错速查表

| 现象 | 原因与解法 |
|---|---|
| 下载数据/权重极慢或卡住 | `export HF_ENDPOINT=https://hf-mirror.com` 后重试；或提前在有网机器下载后拷贝缓存目录 `~/.cache/huggingface` |
| `CUDA out of memory` | `--batch_size 64`（或 32）；`--image_size 192`；`--num_workers 4` |
| loss 一直不降 / 等于 ln(N) | 检查学习率是否为 0、模型是否 `.to(device)`、print 一下 batch 里图像与文本是否同序（`data.py` 是同一行取的，不会错位） |
| loss 先降后升 | 编码器学习率过大，`--lr_encoders 1e-5` |
| DataLoader 卡住或报 shared memory | `--num_workers 4`；容器环境可加 `export TMPDIR=/tmp` |
| 非 A100 老卡（如 V100）报 bf16 相关错误 | 把 `train.py` / `evaluate.py` 里的 `dtype=torch.bfloat16` 改成 `torch.float16` |
| 显存碎片化报错 | `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Gradio 打不开 | 确认 ssh 端口转发建立；确认 `--port` 与转发端口一致；服务器防火墙无需开放端口 |
| R@1 数字很低 | 先跑 evaluate 的 `--max_samples 200` 快速验证链路，再看是不是训练轮数不够 / batch 太小 / 用了 `best.pt` 而不是 `last.pt` |

---

## 9. 学习路线：把这个项目榨干

**第 1 周 — 跑通 + 复盘**
- 跑完 Stage A/B/C；把 `model.py` 的 `contrastive_loss` 手推一遍（写出 logits 矩阵、正负样本位置）；
- 复盘 Transformer：注意力、位置编码、[CLS]；想清楚 ViT 和 BERT 的编码器结构其实同源。

**第 2 周 — 对照实验（开始有自己的发现）**
- 换图像编码器：把 `--image_encoder` 换成 `google/vit-base-patch32-224` 或 timm 的 ResNet50，对比 R@1；
- 换 batch size（64/128/256），画出"batch size vs R@1"曲线——对比学习的经典结论；
- 阅读：**ViT**、**CLIP 论文**（重点看对比损失与温度、batch 内负样本）。

**第 3 周 — 进入"生成"与融合对比**
- 做 Stage D（BLIP 微调）；理解 BLIP 的三头结构（ITC 对比 + ITM 匹配 + LM 生成）；
- 阅读：**ALBEF**、**BLIP**（重点：对比融合 vs 跨注意力融合的差异）。

**第 4 周 — 站到前沿**
- 阅读：**BLIP-2（Q-Former）**、**CoCa**、**BLIP-3 / 现代统一多模态模型**的综述；
- 写一份 3~5 页的技术报告：对比融合 / 跨注意力融合 / Query 融合三种范式的优缺点 + 你自己的实验数据。这份报告就是你进组汇报的现成材料。

**推荐论文清单（按顺序）**：ViT → CLIP → ALBEF → BLIP → BLIP-2 → CoCa。前两篇是地基，中间两篇对应"检索+生成"，后两篇对应当前时代的融合范式。

---

## 10. 命令速查表

```bash
conda activate miniclip
export HF_ENDPOINT=https://hf-mirror.com

python demo_stage_a.py --query "a dog running on the beach"          # Stage A：即时效果
python train.py --max_train_samples 512 --val_samples 128 --epochs 1 --batch_size 64   # 冒烟测试
python train.py --epochs 5 --batch_size 128                          # 全量训练
python evaluate.py --checkpoint checkpoints/miniclip/best.pt --split test   # 测试集评估
python demo_app.py --checkpoint checkpoints/miniclip/best.pt          # Gradio Demo（本地另开 ssh -L 7860:localhost:7860 ...）
```

---

## 11. 完成标准（Checklist）

- [ ] Stage A 打印出 BLIP 生成的描述 + CLIP 检索排序
- [ ] 冒烟测试 loss 从 ~ln(N) 明显下降，checkpoint 保存成功
- [ ] 全量训练逐 epoch 的 t2i R@1 总体上升
- [ ] `evaluate.py` 在 test 集输出完整 R@1/5/10 表格，远高于随机基线
- [ ] Gradio Demo 两个页面都能正常交互
- [ ] 能不看资料向别人讲清楚：什么是对比学习融合、InfoNCE 里的正负样本是什么、R@K 怎么算

全部打勾后，你已经具备了多模态方向的最小完整能力栈，接下来换数据集、换编码器、加生成头都是在这个骨架上的变体。欢迎进组后拿着你的 R@K 实验记录去和导师讨论下一步。
