# GRPO / PPO 训练仓库说明

本 README 说明从数据集准备、奖励模型训练、PPO/GRPO 训练到推理测试（`test.py`）的完整流程与常用命令。请根据你本地路径和显存情况调整超参。

**目录**
- 项目概述
- 环境依赖
- 数据集格式与准备
- 训练奖励模型（`1_train_reward_model.py`）
- PPO 训练（`2_train_ppo.py`）
- GRPO 训练（`2_train_grpo.py`）
- 推理 / 对比测试（`test.py`）
- 常见问题与调参建议


**项目概述**
- 本仓库实现了奖励模型训练、PPO 与 GRPO 两种强化学习训练流程，基于 Qwen 系列模型作为基座（可替换为其他因 trust_remote_code 需要的模型）。
- 关键脚本：
  - `1_train_reward_model.py`：训练裁判（reward）模型，输入为成对的（chosen, rejected）样本
  - `2_train_ppo.py`：基于 reward model 的 PPO 训练脚本（含 Critic）
  - `2_train_grpo.py`：GRPO 风格训练脚本（组内相对优势，无需 critic）
  - `test.py`：加载模型并对比输出用于离线评估


**环境依赖**
- Python 3.8+
- 推荐安装包：

```bash
pip install -U torch transformers accelerate tensorboard
```

（根据你的 GPU/CPU 选择合适的 `torch` 版本与安装命令）


**数据集格式与准备**
- prompts.json：一个字符串数组，每项为单条 Prompt，例如：

```json
[
  "请解释什么是量子纠缠？",
  "写一首关于秋天的诗。"
]
```

- reward_data.json：用于训练奖励模型（对比学习）的样本数组，每项包含 `prompt`, `chosen`, `rejected`，示例：

```json
[
  {
    "prompt": "请解释量子纠缠",
    "chosen": "量子纠缠是指...（更好/更完整的回答）",
    "rejected": "纠缠就是两个粒子有关联...（较差回答）"
  }
]
```

- 路径：默认 `1_train_reward_model.py` 中使用 `REWARD_DATA_PATH = './data/reward_data.json'`，`2_train_ppo.py` 与 `2_train_grpo.py` 默认在根目录加载 `prompts.json`（如果你放在 `data/` 下请相应修改脚本中的路径或将文件移动到仓库根目录）。


**训练奖励模型（1_train_reward_model.py）**
- 目的：训练一个判分器（sequence classification）作为 RL 的 reward 模型。
- 关键配置（脚本内可修改）：
  - `MODEL_NAME`：基座模型路径（默认为 `./Qwen2.5-0.5B-Instruct`）
  - `SAVE_PATH`：保存路径（默认 `./my_custom_reward_model`）
  - `REWARD_DATA_PATH`：训练数据路径
  - `MAX_LENGTH`, `BATCH_SIZE`, `EPOCHS`, `LR`
- 训练细节：
  - 使用 pairwise loss：loss = -logsigmoid(r_chosen - r_rejected)
  - 在 loss 计算时强制转换为 `float32` 以减少 fp16/bf16 下的数值不稳定
  - 使用 `gradient_checkpointing_enable()` 与梯度裁剪 `clip_grad_norm_` 来降低显存和稳定训练

- 运行示例：

```bash
python 1_train_reward_model.py
```

- 训练完成后会把模型与 tokenizer 保存到 `SAVE_PATH`，PPO/GRPO 脚本通过 `REWARD_MODEL_PATH` 加载。


**PPO 训练（2_train_ppo.py）**
- 主要变量（脚本顶部）：
  - `SAVE_PATH`：训练后 actor 模型保存路径（默认 `./new_ppo_model`）
  - `LOG_DIR`：TensorBoard 日志目录
  - `REWARD_MODEL_PATH`：指向上一步训练好的奖励模型
  - `ACTOR_MODEL_PATH`：actor（基座）模型路径
- 关键流程：
  1. 从 `prompts.json` 加载 prompts
  2. 使用 actor 生成带采样的回答（`do_sample=True`）形成样本
  3. 使用 reward_model 对回答打分，将得分与 KL 项合成奖励
  4. 使用 Critic 估计 value，计算 GAE 优势并做 PPO update（actor + critic）
- 可调整超参（脚本中可直接修改）：
  - `episodes`、`micro_rollout_batch_size`
  - `max_length`（token 长度截断）、`max_new_tokens`（生成长度）
  - 学习率（actor/critic）、batch_size（训练 DataLoader）
  - `kl_ctl`（脚本内部用于平衡 KL 与 reward）
- 运行示例：

```bash
python 2_train_ppo.py
```

- 训练输出：保存模型到 `SAVE_PATH`，日志写入 `LOG_DIR`，可通过 `tensorboard --logdir runs/` 查看。


**GRPO 训练（2_train_grpo.py）**
- 设计要点：组内相对优势（Group Relative Policy Optimization），不依赖 Critic。
- 脚本顶端常用参数：
  - `GROUP_SIZE`：每个 prompt 生成多少条回答用于组内比较（示例为 8，显存小可设 4）
  - `KL_COEF`：KL 惩罚系数（示例 0.05）
  - `CLIP_EPS`：PPO clip epsilon
  - `SAVE_PATH`, `LOG_DIR`, `REWARD_MODEL_PATH`, `ACTOR_MODEL_PATH`
- 关键流程：
  1. 对每个 prompt 生成 `GROUP_SIZE` 个采样回答
  2. 使用 reward_model 对组内回答打分，计算组内标准化的 advantage = (r - mean)/std
  3. 用 advantage 做策略梯度更新，并加入基于 ref_model 的 KL 惩罚
- 运行示例：

```bash
python 2_train_grpo.py
```

- 建议：若显存受限，降低 `GROUP_SIZE`、缩短 `max_new_tokens`，或者在多卡环境下使用模型并行/`device_map`。


**推理 / 对比测试（test.py）**
- `test.py` 会同时加载 `BASE_MODEL_PATH`（原始基座模型）和 `PPO_MODEL_PATH`（训练后模型，用于对比）
- 运行前请修改脚本顶部的路径：
  - `BASE_MODEL_PATH`：例如 `./Qwen2.5-0.5B-Instruct`
  - `PPO_MODEL_PATH`：例如 `./new_ppo_model` 或 `./grpo_model_final`
- 运行示例：

```bash
python test.py
```

- 输出：对一组测试 prompts，打印基座模型和训练后模型的回答，便于人工比较质量差异。


**常见问题与调参建议**
- OOM（显存不足）：
  - 降低 `GROUP_SIZE` / `batch_size` / `max_new_tokens`
  - 使用 `gradient_checkpointing_enable()` 或混合精度（bf16 / fp16）
- 数值稳定性：
  - 在奖励模型训练里 loss 使用 `float32` 计算，避免 fp16 导致的 NaN
  - 使用梯度裁剪（`clip_grad_norm_`）与较小学习率
- TensorBoard：
  - 日志目录一般为 `runs/`，可以运行：

```bash
tensorboard --logdir runs/
```

- 保存与加载模型：
  - 训练脚本会用 `model.save_pretrained(SAVE_PATH)` 和 `tokenizer.save_pretrained(SAVE_PATH)` 保存模型
  - 推理脚本通过 `AutoModelForCausalLM.from_pretrained(path)` 加载
