## DeepSeek-V2-Lite 基础评测项目

本项目提供一个最小可运行的代码框架，用于：

- **加载** `DeepSeek-V2-Lite-Chat` 模型（Hugging Face `deepseek-ai/DeepSeek-V2-Lite-Chat`）
- 在 **MMLU** 上计算 **accuracy**
- 在 **WMDP** 遗忘集上计算 **4-option multiple-choice accuracy**（理想值 0.25，**越低越好**）
- 在 **RWKU** 遗忘集上计算 **Rouge-L recall**（理想值 0.0，**越低越好**）

### 1. 环境安装

推荐使用 Python 3.10+，在虚拟环境中安装依赖：

```bash
cd /root/autodl-tmp/deepseek_eval
pip install -r requirements.txt
```

> 注意：`DeepSeek-V2-Lite` 模型较大，建议在有 GPU 的环境下运行，并确保磁盘空间充足（数十 GB）。

### 2. 脚本说明

- `eval_deepseek_v2_lite.py`：主评测脚本，包含：
  - 模型加载（`deepseek-ai/DeepSeek-V2-Lite-Chat`）
  - MMLU / WMDP / RWKU 三个数据集的评测逻辑
  - 指标计算与最终结果输出（终端打印 + 写入 JSON 文件）

### 3. 运行示例

默认只抽样各数据集前 100 条数据进行评测，你可以根据显存/算力调整样本数：

```bash
cd /root/autodl-tmp/deepseek_eval

python eval_deepseek_v2_lite.py \
  --model-name deepseek-ai/DeepSeek-V2-Lite-Chat \
  --max-mmlu-samples 100 \
  --max-wmdp-samples 100 \
  --max-rwku-samples 100 \
  --device-map auto \
  --output-json eval_results.json
```

运行结束后，会在当前目录生成 `eval_results.json`，示例结构如下：

```json
{
  "mmlu_accuracy": 0.45,
  "mmlu_correct": 45,
  "mmlu_total": 100,
  "wmdp_accuracy": 0.30,
  "wmdp_correct": 30,
  "wmdp_total": 100,
  "wmdp_ideal_random": 0.25,
  "rwku_rouge_l_recall": 0.05,
  "rwku_num_samples": 100,
  "rwku_ideal": 0.0
}
```

### 4. 数据集说明与自定义

#### 4.1 MMLU

脚本默认使用 Hugging Face 上的：

- `cais/mmlu`，配置名：`all`

无需手动下载，`datasets` 库会自动拉取。

#### 4.2 WMDP（遗忘集，多选题）

脚本中默认假设：

- 数据集名称：`cais/wmdp`（你可以根据实际情况替换为自己的数据集路径 / 本地目录）
- 字段格式（可以按自己数据集改）：
  - `question` 或 `prompt`：题干
  - `choices` 或 `options`：长度为 4 的选项列表
  - `answer`：正确答案（可以是 0–3 的 index，或者 `"A"/"B"/"C"/"D"`）

你也可以通过命令行参数指定自定义数据集：

```bash
python eval_deepseek_v2_lite.py \
  --wmdp-dataset cais/wmdp
```

如果是本地数据集（例如 `./data/wmdp`），同样可以传入：

```bash
python eval_deepseek_v2_lite.py \
  --wmdp-dataset ./data/wmdp
```

#### 4.3 RWKU（遗忘集，生成类）

脚本中默认假设：

- 数据集名称：`cais/rwku`
- 字段格式（可根据实际 RWKU 数据调整）：
  - `prompt` 或 `question`：输入给模型的文本
  - `reference_answer` 或 `answer`：用来计算 Rouge-L recall 的参考文本

同样支持通过参数指定：

```bash
python eval_deepseek_v2_lite.py \
  --rwku-dataset cais/rwku
```

### 5. 指标解读

- **MMLU accuracy**：标准多学科理解基准，**越高越好**。
- **WMDP 4-option accuracy**：4 选 1 多选题准确率，随机水平约为 **0.25**，如果你做“遗忘”实验，则**越低越好**。
- **RWKU Rouge-L recall**：模型输出与参考知识的重叠度，**越低越好**，理想值接近 **0.0**。

### 6. 后续可扩展方向

- 加入批量推理 / 多 GPU 加速（如结合 `vLLM` 服务化部署）
-. 支持更多评测基准（如 TruthfulQA、HaluEval 等）
- 把评测脚本拆成模块化的 `datasets/metrics/models` 结构，方便接更多模型。

