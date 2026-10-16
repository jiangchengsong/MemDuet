# <img src="docs/assets/memduet-icon.svg" width="38" height="38" alt=""> MemDuet

**Learning Dual-Process Memory for Long-Context Reasoning**

用于长上下文推理的双过程记忆学习。

[English](README.md) · [简体中文](README.zh-CN.md)

[项目主页](https://jiangchengsong.github.io/MemDuet/) · [arXiv](https://arxiv.org/abs/2610.19928) · [论文 PDF](https://arxiv.org/pdf/2610.19928) · [BibTeX](docs/assets/memduet.bib)

Jiangcheng Song, Haodong Jing, Yulin Mu, Lesong Tao, Jixin Wang, Jin Cui, Tianjing Ren, Runxuan Xie, Yongqiang Ma, Nanning Zheng

MemDuet 使用小模型并行筛选上下文中的证据，由大模型完成候选验收，并按照文档顺序更新记忆。

## 方法

![MemDuet 推理流程](docs/assets/framework.png)

- **system1：** 并行判断上下文块是否需要触发记忆更新。主要骨干为 Qwen3.5-2B，论文还评估了 Qwen2.5-0.5B、Qwen2.5-1.5B 和 Qwen3.5-0.8B。
- **system2（7B）：** 验收更新候选、按文档顺序改写记忆，并生成最终答案。
- **训练流程：** 监督学习、记忆更新强化学习，以及两个系统的联合优化。

## 实验结果

在 7K 至 1.75M tokens 的九档共享上下文长度上，MemDuet 的平均 RULER-HQA 准确率为 **84.20%**。3.5M tokens 的结果单独报告。完整的分长度结果和论文原图见项目主页。

在 RULER-HQA、LongBench v2 和 2WikiMultiHopQA 的评测设置上，论文报告相较 RL-MemAgent 平均 **9.74 倍推理加速**，并减少 **73.91% 的输出 tokens**。

## 代码资源

推理代码位于 [`src/memduet`](src/memduet)，包含 vLLM 请求接口、候选验收与记忆更新、模型合并导出，以及支持断点续跑的评测流程。模型权重和数据集单独提供。

## 示例配置

[`configs/inference.example.json`](configs/inference.example.json) 使用以下模型组合：

- **system1：** 冻结的 Stage I Qwen3.5-2B 模型。
- **system2：** 完成 MemDuet 三阶段训练流程后得到的最终 7B 模型。

将两个完整推理模型分别放在仓库根目录下的 `system1/` 和 `system2/`，目录内需包含模型权重、配置文件、tokenizer 和对应的聊天模板。

本示例直接加载已合并的完整模型。若 system2 提供的是 Stage III 适配器，需要先与它对应的、经过 Stage II 训练的骨干模型合并，再将完整模型放入 `system2/`。

完成下文的环境安装和模型放置后，检查路径：

```bash
python -m memduet validate --config configs/inference.example.json --check-paths
```

在不同终端启动两个服务，每个服务就绪后再继续下一步：

```bash
# 终端 1：启动最终 MemDuet system2，负责验收和记忆更新。
CUDA_VISIBLE_DEVICES=0 python -m memduet serve --config configs/inference.example.json --role system2

# 终端 2：system2 就绪后，启动 system1。
CUDA_VISIBLE_DEVICES=0 python -m memduet serve --config configs/inference.example.json --role system1

# 终端 3：两个服务都就绪后，评测一个上下文长度的 RULER-HQA 数据。
python -m memduet evaluate --config configs/inference.example.json \
  --input data/ruler_hqa/112k.jsonl --output runs/eval/ruler_hqa_112k.json \
  --metric sub_em

python -m memduet summarize --results runs/eval/ruler_hqa_112k.json \
  --expected-ids configs/splits/paper_ruler_hqa.ids.json --strict
```

RULER-HQA 的各档长度使用同一组 [128 个问题 ID](configs/splits/paper_ruler_hqa.ids.json)。准备输入时，将发布数据中的 `source_id` 保留为输入记录的 `id`，不要用行号或 HotpotQA 原始 ID 代替。数据文件需单独准备。

LongBench v2 使用 `--metric choice`，RULER-HQA 和 2WikiMultihopQA 使用 `--metric sub_em`。

## 安装环境

vLLM 服务使用 Linux 和 NVIDIA GPU。每个 GPU 上放置完整模型对，每次处理一道题；同一道题的窗口内最多同时处理四个候选块。请根据显存容量调整配置。

```bash
git clone https://github.com/jiangchengsong/MemDuet.git
cd MemDuet
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[inference,serve,export]' -c requirements-lock.txt
python -m memduet --help
```

`requirements-lock.txt` 固定 vLLM 推理使用的依赖版本。导入 Python 包本身不需要启动 vLLM，也不会下载模型。

## 使用适配器导出模型

若已持有完整推理模型，直接使用上面的示例配置。若持有的是训练适配器，修改 [`configs/inference.json`](configs/inference.json)，填入对应的模型和适配器路径：

- system1 的适配器路径为 `checkpoints/system1`，默认骨干为 `Qwen/Qwen3.5-2B`。
- system2 的最终适配器路径为 `checkpoints/system2`，其配套的训练后骨干位于 `checkpoints/system2/backbone`。

所有路径均相对于仓库根目录。运行合并导出：

```bash
python -m memduet export-inference --config configs/inference.json
```

导出目录为 `models/memduet-system1` 和 `models/memduet-system2`，包含 tokenizer。导出过程要求目标目录尚不存在。导出后，可将示例配置中的两处 `model_path` 分别改为这两个目录。

## 输入数据与评测输出

输入采用 JSONL，每条记录包含 `id`、`question`、`answers`，以及 `context` 或非空的 `chunks` 列表。`answers` 是评分使用的参考答案。也支持使用 `input`、`context` 和 `answers` 字段的 JSON 数组。

下面是一条演示格式的合成数据：

```json
{"id":"example","question":"Which city hosted the duet?","answers":["Xi'an"],"chunks":["The duet took place in Xi'an.","Both performers returned for an encore."]}
```

将记录保存为 `data/eval.jsonl`，并在两个模型服务就绪后运行：

```bash
python -m memduet evaluate --config configs/inference.example.json \
  --input data/eval.jsonl --output runs/eval/memduet.json
```

评测输出包含逐题答案、候选选择与验收轨迹、耗时，以及两个模型的输出 token 统计，包含验收和被丢弃候选产生的开销。模型请求失败或出现回退轨迹时，默认严格模式会停止评测。保持输入和配置不变，重复执行原命令可接续已完成的问题。

示例配置使用阈值选择；通用配置 `configs/inference.json` 默认使用概率采样选择。两种方式都由 system2 按 `min(1, p(u)/q(u))` 的概率验收候选。文档顺序中最早被接受的候选更新记忆，后续候选使用新记忆重新判断。

代码许可证为 [Apache-2.0](LICENSE)，第三方声明见 [NOTICE](NOTICE)。模型和数据集遵循各自的许可证。

## 引用

```bibtex
@misc{song2026memduet,
  title  = {MemDuet: Learning Dual-Process Memory for Long-Context Reasoning},
  author = {Song, Jiangcheng and Jing, Haodong and Mu, Yulin and
            Tao, Lesong and Wang, Jixin and Cui, Jin and Ren, Tianjing and
            Xie, Runxuan and Ma, Yongqiang and Zheng, Nanning},
  year   = {2026},
  eprint = {2610.19928},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url    = {https://arxiv.org/abs/2610.19928}
}
```

## 项目主页

静态项目页位于 `docs/`，由 GitHub Pages 从 `main` 分支发布。本地预览：

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory docs
```

浏览器打开 `http://127.0.0.1:8765/`。
