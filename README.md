# 鸟类多模态分类实验

配套《实验 2：鸟类多模态分类实验手册》，从零实现「图像 + 文本 → 鸟类类别」的多模态分类，
并逐一实现手册提到的四种融合方式、编码器特征拆解与 CLIP 零样本实验。

```
图片 ─► Image Encoder ─► 图像向量 / 视觉 token ─┐
                                                 ├─► Fusion ─► Classifier ─► 鸟的种类
描述 ─► Text Encoder  ─► 文本向量 / 文本 token ─┘
```

---

## 1. 目录结构

```
bird_multimodal/
├── requirements.txt                 依赖清单
├── config.yaml                      全局配置（换融合方式只改 fusion.name）
│
├── birdmm/                          可复用库
│   ├── utils.py                     配置加载 / 随机种子 / 设备选择 / 参数量统计
│   ├── data.py                      §1.3 数据扫描 + 6:2:2 分层划分 + annotations.json + Dataset
│   ├── metrics.py                   Accuracy / 宏平均 Precision、Recall、F1
│   ├── clip_utils.py                §5 CLIP 共同语义空间：编码、零样本、检索 R@k
│   └── models/
│       ├── image_encoder.py         §3 ResNet → 特征图 [B,512,7,7] → 池化向量 / 视觉 token
│       ├── text_encoder.py          §3 BERT → [B,L,768] → cls/mean 池化（含 mask 讲解）
│       ├── fusion.py                §4 四种融合，统一接口
│       └── multimodal.py            双塔 + 融合头组装
│
├── 01_prepare_data.py               §1.3 生成 annotations.json（6:2:2 冻结划分）
├── 02_singleModal_ImageOnly.py      §2.1 Image-Only 训练 / 测试 / 指标 + 特征拆解
├── 02_singleModal_TextOnly.py       §2.2 Text-Only 训练 / 测试 / 指标 + 分词演示
├── 03_inspect_features.py           §3 模型拆解 + Cross-Attention 注意力可视化
├── 04_train_multimodal.py           §4 多模态训练 + 融合方式消融
├── 05_clip_zero_shot.py             §5 CLIP 零样本分类 + 图文检索
│
├── tools/make_dummy_data.py         合成数据生成器（无真实数据时先跑通流程）
├── datasets/                        数据目录（当前为合成演示数据 + annotations.json）
└── runs/                            各实验的输出：best.pt + metrics.json
```

---

## 2. 环境准备

> ⚠️ **本机注意**：`pip` / `python` 不在 PATH 上（PATH 里那个 `python.exe` 是微软商店的
> 占位别名，不是真解释器）。请统一用 **`py -m pip`** 和 **`py`** 来执行命令。
> 如果报 `无法将"pip"项识别为 cmdlet…`，就是把 `pip` 换成 `py -m pip`。

```powershell
cd bird_multimodal

# 安装依赖
py -m pip install -r requirements.txt

# 国内网络慢，加镜像（推荐）
py -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

> 命令末尾如果出现一段红字 `py : At line:… RemoteException`，那是 pip 往 stderr 输出的
> 升级提示（`[notice] A new release of pip is available`），**不是安装失败**，可以忽略。
>
> `requirements.txt` 内含中文注释，**已保存为「UTF-8 with BOM」**，这样中文 Windows 下
> pip 也不会用 GBK 去解码而报 `UnicodeDecodeError`。若你重新保存过这个文件导致 BOM 丢失，
> 见 §6 FAQ（可临时用 `$env:PYTHONUTF8 = 1` 绕过）。

**国内网络建议**（下载 HuggingFace 模型必需）：

```powershell
# 永久生效
setx HF_ENDPOINT https://hf-mirror.com
# 或仅当前会话
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

**torch 按 CUDA 版本装**：

```powershell
# CPU 版（本机验证环境用的就是这个）
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.1
py -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> 已验证环境：Python 3.11 / torch 2.14.0+cpu / torchvision 0.29.0+cpu / **transformers 5.19.0** /
> matplotlib 3.11.2 / scikit-learn 1.9.1 / numpy 2.4.6。
> 代码对 transformers 4.x 和 5.x 都做了兼容。

---

## 3. 数据准备

目录约定：**类别文件夹 + 同名 jpg/png + 同名 txt**（CUB 风格，每张图 10 条描述）。

```
datasets/
├── 001.Red_bird/
│   ├── Red_bird_001_0000.jpg
│   └── Red_bird_001_0000.txt        # 每行一条英文描述
├── 002.Blue_bird/
└── ...
```

没有真实数据时，先生成一份合成数据把流程跑通：

```bash
py tools/make_dummy_data.py --classes 5 --per-class 16 --size 224
```

生成标注文件（**只需跑一次**，划分会被冻结，三种实验共用同一套划分）：

```bash
py 01_prepare_data.py
```

输出示例：

```
[数据] 类别数：5
[数据] 样本数：{'train': 50, 'val': 15, 'test': 15}
```

换用真实数据：把 `config.yaml` 里的 `data.root` 指到你的数据目录，删掉旧的
`datasets/annotations.json` 后重跑 `01_prepare_data.py` 即可。

---

## 4. 实验流程

### §2.1 Image-Only（单模态图像）

```bash
py 02_singleModal_ImageOnly.py --epochs 10
# CPU 想更快：加 --freeze-backbone 只训练分类头
```

学习目标对应：
1. 训练 / 测试 / 指标验证 → 见脚本 `evaluate()` 与 `classification_metrics()`
2. 模型拆解 → 脚本会打印一次真实前向的张量流动：

```
输入图像        : (2, 3, 224, 224)
ResNet 特征图    : (2, 512, 7, 7)     # 去掉 avgpool/fc 后的 [B, C, h, w]
视觉 token       : (2, 49, 512)       # flatten+transpose：每张图切成 49 个"视觉单词"
全局池化向量     : (2, 512)
```

### §2.2 Text-Only（单模态文本）

```bash
py 02_singleModal_TextOnly.py --epochs 10
```

脚本会先演示分词：`"a small blue bird" → ['[CLS]','a','small','blue','bird','[SEP]']`。
主干的 `pooling` 支持 `--pooling cls|mean`，`mean` 会按 `attention_mask` 过滤 padding。

### §3 特征拆解与注意力可视化

```bash
py 03_inspect_features.py
# 有训练好的 cross_attention 模型时，顺带导出注意力热图
py 03_inspect_features.py --checkpoint runs/multimodal_cross_attention/best.pt
```

会打印图像与文本两侧的完整张量流动，并演示 padding 对句向量的稀释效应。
注意力部分输出「图像最关注的 5 个词」，热图存到 `runs/inspect/cross_attention_weights.png`。

### §4 多模态训练（四种融合）

```bash
py 04_train_multimodal.py --fusion cross_attention    # 推荐
py 04_train_multimodal.py --fusion early
py 04_train_multimodal.py --fusion late
py 04_train_multimodal.py --fusion gated
```

**换融合方式只需要改 `--fusion` 一个参数**（或改 `config.yaml` 的 `fusion.name`）。
四种融合的统一接口：

```python
forward(image_feat, text_feat, attention_mask) -> {
    "logits":       [B, C],
    "image_logits": [B, C],       # 仅 late
    "text_logits":  [B, C],       # 仅 late
    "attn_weights": [B, H, N_img, L],  # 仅 cross_attention
    "gate":         [B, d],       # 仅 gated
}
```

| 融合方式 | 做法 | 特点 |
|---|---|---|
| `early` | `[img; txt]` 拼接 → MLP | 最简单、参数最少；模态间无交互 |
| `late` | 各自分类 → 概率加权（权重 `w` 可学习） | 任一模态缺失都能工作；可解释 |
| `cross_attention` | 图像 token 作 Query，文本 token 作 K/V | 显式建模跨模态对齐；可导出注意力 |
| `gated` | `a = σ(W[img; txt])`，`a·img + (1-a)·txt` | 逐样本决定信谁，抑制不可靠文本 |

### §5 CLIP 思想实验

```bash
py 05_clip_zero_shot.py
```

没有分类器 —— 类别通过「和类别描述的相似度」被比较出来。输出零样本 Accuracy
与图文检索 `R@1 / R@5`。

---

## 5. 已验证结果（合成演示数据，5 类 80 图）

> ⚠️ 下面是**流程验证**用的合成数据结果，不是科学结论。换成真实 CUB 数据后需重跑。

| 实验 | 测试集 Accuracy | 宏平均 P / R / F1 |
|---|---|---|
| Image-Only（3 epoch） | 0.9333 | 0.9500 / 0.9333 / 0.9314 |
| Text-Only（3 epoch） | 0.4000 | 0.2550 / 0.4000 / 0.2891 |
| CLIP 零样本 | 1.0000 | 1.0000 / 1.0000 / 1.0000 |

**融合方式消融**（3 epoch，主干冻结）：

| fusion | 可训练参数 | 测试 Acc | 运行时可解释性输出 |
|---|---|---|---|
| early | 1.51 M | 0.4667 | — |
| late | 0.86 M | 0.4667 | 模态权重 w = 0.500 |
| **cross_attention** | 3.09 M | **0.7333** | attn `(15, 8, 49, 64)` |
| gated | 2.43 M | 0.6000 | 门控 a 均值 0.500 |

Cross-Attention 最优，与手册 §4 的推荐一致。CLIP 检索 `R@1 = 0.3333`、`R@5 = 1.0000`。

---

## 6. 常见问题

**Q: `无法将"pip"项识别为 cmdlet、函数、脚本文件或可运行程序的名称`**
本机 `pip` / `python` 不在 PATH 上，用 `py -m pip` / `py` 代替（见 §2）。

**Q: 安装时报 `UnicodeDecodeError: 'gbk' codec can't decode byte 0xa3 in position 6`**
中文 Windows 下 pip 默认按 GBK 读取 `requirements.txt`，而文件是 UTF-8。
本项目的 `requirements.txt` **已带 UTF-8 BOM**，pip 会优先识别 BOM 并用 UTF-8 解码。
如果这个错误又出现了，说明 BOM 被编辑器/格式化工具去掉了，任选一种修复：

```powershell
# 方式 1：让 Python 强制使用 UTF-8（临时，当前会话有效）
$env:PYTHONUTF8 = 1
py -m pip install -r requirements.txt

# 方式 2：重新补上 BOM
py -c "p='requirements.txt';d=open(p,'rb').read();open(p,'wb').write(b'\xef\xbb\xbf'+d if not d.startswith(b'\xef\xbb\xbf') else d)"
```

**Q: `ImportError: 文本编码器需要 transformers`**
装依赖：`py -m pip install transformers`（用上面的镜像）。

**Q: 下载 `bert-base-uncased` / CLIP 权重卡住或 429**
先设 `HF_ENDPOINT=https://hf-mirror.com`。镜像偶发限流，会自动重试；
必要时多跑一次，权重会进 `~/.cache/huggingface` 缓存。

**Q: `get_text_features` 返回的对象没有 `.norm`**
这是 transformers 5.x 的接口变化 —— 已在本项目中处理（`birdmm/clip_utils.py`
的 `_feature_tensor()` 同时兼容 4.x 返回张量和 5.x 返回 `BaseModelOutputWithPooling`）。

**Q: 没有 GPU，跑不动 BERT 微调**
`config.yaml` 里 `text_encoder.freeze: true`（默认）会冻结 BERT，只训练融合层，
CPU 上一轮几分钟即可跑完。

**Q: `runs/` 占了 2 GB 多**
因为冻结的 BERT（约 110 M 参数）也被完整存进了每个 `best.pt`。若不需要复现推理，
可以直接删掉 `best.pt`，保留 `metrics.json` 即可。

---

## 7. 把项目改到真实数据上

1. 把 `config.yaml` 的 `data.root` 指向真实数据目录（保持「类别文件夹 + 同名 jpg/txt」结构）
2. 删掉 `datasets/annotations.json`，重跑 `py 01_prepare_data.py`
3. 按需要调整 `config.yaml`：
   - `data.split`：划分比例（默认 `[0.6, 0.2, 0.2]`）
   - `train.epochs` / `train.lr` / `train.batch_size`
   - `image_encoder.backbone`：`resnet18` / `resnet34` / `resnet50`
   - `text_encoder.model_name`：`bert-base-uncased` / `distilbert-base-uncased`
   - `fusion.name`：`early` / `late` / `cross_attention` / `gated`
4. 依次跑 §4 的实验流程
