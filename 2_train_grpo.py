import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
import copy
import gc
import os
import json
import numpy as np

# ---- 1. 配置 ----
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
model_dtype = torch.bfloat16 if use_bf16 else torch.float16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 路径配置
SAVE_PATH = "./grpo_model_final"          # GRPO 训练后模型保存路径
LOG_DIR = "./runs/grpo_experiment"        # 日志路径
REWARD_MODEL_PATH = "./my_custom_reward_model" # 必须是你训练好的奖励模型
ACTOR_MODEL_PATH = "./Qwen2.5-0.5B-Instruct"   # 必须是你的基座模型路径

# GRPO 特有超参数
GROUP_SIZE = 8        # 关键参数：每个问题生成多少个回答进行对比 (显存小设4，大设8或16)
KL_COEF = 0.05        # KL 散度系数，防止模型跑飞
CLIP_EPS = 0.2        # PPO Clip 参数

print(f"Using Device: {device}")
print(f"Using Dtype: {model_dtype}")
print(f"Group Size: {GROUP_SIZE}")

# --- 2. 类定义 ---


# 数据集 (不变)
class PromptDataset(Dataset):
    def __init__(self, prompts, tokenizer, apply_chat_template=False):
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.final_prompts = []
        for prompt in prompts:
            if apply_chat_template:
                content = [{"role": "user", "content": prompt}]
                prompt = self.tokenizer.apply_chat_template(content, tokenize=False, add_generation_prompt=True)
            else:
                prompt = self.tokenizer.bos_token + prompt
            self.final_prompts.append(prompt)
    def __len__(self): return len(self.final_prompts)
    def __getitem__(self, index): return self.final_prompts[index]

# 经验数据结构 (移除了 Values)
@dataclass
class Experience:
    seqs: torch.Tensor             # [Group, L]
    action_log_probs: torch.Tensor # [Group, L-1]
    advantages: torch.Tensor       # [Group, L-1] (GRPO 核心：基于组内相对分数的优势)
    attention_mask: torch.Tensor   # [Group, L]
    action_mask: torch.Tensor      # [Group, L-1]
    ref_log_probs: torch.Tensor    # [Group, L-1] (用于计算 KL)

class ExperienceBuffer:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = []
    def append(self, experiences):
        batch = []
        for exp in experiences:
            item = {}
            for k, v in exp.__dict__.items():
                if isinstance(v, torch.Tensor):
                    # 移回 CPU 节省显存
                    item[k] = v.detach().cpu()
                else:
                    item[k] = v
            batch.append(item)
        self.buffer.extend(batch)
        if len(self.buffer) >= self.limit: self.buffer = self.buffer[-self.limit:]
    def clear(self):
        self.buffer = []
        gc.collect()
    def __len__(self): return len(self.buffer)
    def __getitem__(self, index): return self.buffer[index]


# --- 3. 核心逻辑修改：GRPO 采样与优势计算 ---

def generate_group_experiences(prompts, actor_model, ref_model, reward_model, tokenizer, reward_tokenizer, device):
    """
    GRPO 核心流程：
    1. 对每个 Prompt，生成 GROUP_SIZE 个回答。
    2. 计算这组回答的 Reward。
    3. 计算组内优势 (Advantage) = (Reward - Mean) / Std。
    """
    actor_model.eval(); ref_model.eval(); reward_model.eval()
    experiences = []
    
    with torch.no_grad():
        for prompt in prompts:
            # 1. 构造输入：复制 prompt GROUP_SIZE 次，形成一个 Batch
            #    Input Shape: [Group, Seq_Len]
            inputs = tokenizer([prompt] * GROUP_SIZE, return_tensors='pt', padding=True).to(device)
            input_ids = inputs.input_ids
            prompt_len = input_ids.shape[1]
            
            # 2. 批量生成 (Group Sampling)
            #    Output Shape: [Group, Total_Len]
            seqs = actor_model.generate(
                **inputs,
                max_new_tokens=64,  # 可以适当调大，看显存 回答的tokens长度,显存情况，256,512
                do_sample=True,     # 必须采样，否则生成的一模一样，GRPO 就失效了
                temperature=0.8,    # 增加一点随机性，让 Group 内回答多样化
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

            # 打印第一个和第二个回答的前50个字对比一下
            print(f"Sample 1: {tokenizer.decode(seqs[0][:50])}")
            print(f"Sample 2: {tokenizer.decode(seqs[1][:50])}")
            
            # 3. 构造 Masks
            attention_mask = (seqs != tokenizer.pad_token_id).long()
            action_mask = torch.zeros_like(attention_mask[:, 1:]) # log_probs 是 L-1 长度
            # 只有生成的 Answer 部分有效
            for i in range(GROUP_SIZE):
                # 找到 prompt 结束后的位置
                action_mask[i, prompt_len-1:] = 1 
                # 找到 eos 位置，eos 之后的不算
                eos_idx = (seqs[i] == tokenizer.eos_token_id).nonzero()
                if len(eos_idx) > 0:
                    first_eos = eos_idx[0].item()
                    if first_eos > prompt_len:
                        action_mask[i, first_eos:] = 0 # eos 之后 mask 为 0
            
            # 4. 计算当前策略的 Log Probs
            logits = actor_model(seqs, attention_mask=attention_mask).logits
            log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
            action_log_probs = log_probs.gather(2, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
            
            # 5. 计算 Ref 模型的 Log Probs (用于 KL)
            ref_logits = ref_model(seqs, attention_mask=attention_mask).logits
            ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
            ref_action_log_probs = ref_log_probs.gather(2, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
            
            # 6. 计算 Reward (裁判打分)
            seq_texts = tokenizer.batch_decode(seqs, skip_special_tokens=True)
            rm_inputs = reward_tokenizer(seq_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            # Reward Model 输出 [Group, 1]
            rewards = reward_model(**rm_inputs).logits.squeeze(-1) # [Group]
            
            # 7. --- GRPO 核心：组内相对优势计算 ---
            # Advantage = (r - mean(r)) / (std(r) + epsilon)
            # 这使得 GRPO 不需要 Critic 模型来做 Baseline
            mean_reward = rewards.mean()
            std_reward = rewards.std() + 1e-8
            advantages_score = (rewards - mean_reward) / std_reward # [Group]
            
            # 将这个 scalar advantage 扩展到序列长度 [Group, L-1]
            # 只有 action_mask 为 1 的地方才有 advantage，其他地方为 0 (或者被 mask 掉)
            advantages = torch.zeros_like(action_log_probs)
            for i in range(GROUP_SIZE):
                advantages[i] = advantages_score[i] # 广播
                
            # 存入经验池
            experiences.append(Experience(
                seqs=seqs,
                action_log_probs=action_log_probs,
                advantages=advantages,
                attention_mask=attention_mask,
                action_mask=action_mask,
                ref_log_probs=ref_action_log_probs
            ))
            
            del inputs, seqs, logits, ref_logits
            torch.cuda.empty_cache()
            
    return experiences

def collate_fn(batch):
    # Padding 逻辑 
    def pad(seqs, val):
        return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=val)
    
    # Batch 里每个元素是一个 Group (shape: [G, L])
    # 为了训练方便，我们将所有 Group 展平 -> [B * G, L]
    flat_seqs = torch.cat([x['seqs'] for x in batch], dim=0)
    flat_log_probs = torch.cat([x['action_log_probs'] for x in batch], dim=0)
    flat_adv = torch.cat([x['advantages'] for x in batch], dim=0)
    flat_att_mask = torch.cat([x['attention_mask'] for x in batch], dim=0)
    flat_act_mask = torch.cat([x['action_mask'] for x in batch], dim=0)
    flat_ref_probs = torch.cat([x['ref_log_probs'] for x in batch], dim=0)
    
    # 注意：generate 出来的长度可能不一致，这里其实应该再 pad 一次
    # 但由于我们是在一个 prompt 里 generate，长度差异通常由 eos 决定，tensor 维度是对齐的
    # 如果 seqs 长度本身不一致（不同 prompt），则需要 pad
    # 简单起见，这里假设 collate 前已经 pad 好了，或者我们在 ExperienceBuffer 里存的时候是 list
    
    # 修正：为了稳健，我们应该在 append 时就 pad 好，或者在这里重新 pad
    # 这里简单处理：假设所有 seqs 已经被 pad 到该 batch 的最大长度 (generate 会自动 pad)
    # 如果跨 prompt batch，generate 长度不同，需要 pad_sequence
    
    return {
        'seqs': flat_seqs,
        'action_log_probs': flat_log_probs,
        'advantages': flat_adv,
        'attention_mask': flat_att_mask,
        'action_mask': flat_act_mask,
        'ref_log_probs': flat_ref_probs
    }

# --- 4. 训练步 ---

def train_step(batch, actor_model, opt_actor, device, step, writer):
    seqs = batch['seqs'].to(device)
    old_log_probs = batch['action_log_probs'].to(device)
    advantages = batch['advantages'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    action_mask = batch['action_mask'].to(device)
    ref_log_probs = batch['ref_log_probs'].to(device)
    
    actor_model.train()
    
    # 重新计算 Log Probs
    logits = actor_model(seqs, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    new_log_probs = log_probs.gather(2, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
    
    # --- GRPO Loss 计算 ---
    # 1. Ratio
    ratio = (new_log_probs - old_log_probs).exp()
    
    # 2. Surrogate Loss (PPO Clip)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS) * advantages
    policy_loss = -torch.min(surr1, surr2)
    
    # 3. KL Penalty (直接加在 Loss 里，而不是 Reward 里)
    # KL = exp(ref) * (ref - new) ??? No, KL(P||Q) = sum p log(p/q)
    # Approx KL = new_log - ref_log
    # GRPO 通常使用: Loss = Policy_Loss + beta * KL
    # 这里我们计算每个 token 的 KL
    kl_div = (new_log_probs - ref_log_probs).exp() * (new_log_probs - ref_log_probs) # D_KL(New || Ref) ?
    # 简化的 KL 惩罚: (new - ref)^2 或者 (new/ref - 1) - log(new/ref)
    # 或者最简单的: log(new) - log(ref)
    kl_penalty = new_log_probs - ref_log_probs
    
    # 组合 Loss
    # mask 掉 padding 部分
    loss_element = policy_loss + KL_COEF * kl_penalty
    loss = (loss_element * action_mask).sum() / (action_mask.sum() + 1e-8)
    
    opt_actor.zero_grad()
    loss.backward()
    
    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(actor_model.parameters(), 1.0)
    opt_actor.step()
    
    # Log
    writer.add_scalar("loss/grpo", loss.item(), step)
    # 计算平均 KL 用于观察
    avg_kl = ((new_log_probs - ref_log_probs) * action_mask).sum() / (action_mask.sum() + 1e-8)
    writer.add_scalar("metric/kl", avg_kl.item(), step)
    
    print(f"step: {step}  grpo_loss: {loss.item():.4f}  avg_kl: {avg_kl.item():.4f}")


# --- 5. 主流程 ---

def train():
    print("Loading Actor Model...")
    actor_model = AutoModelForCausalLM.from_pretrained(ACTOR_MODEL_PATH, torch_dtype=model_dtype, trust_remote_code=True).to(device)
    actor_model.gradient_checkpointing_enable()
    
    # Ref Model
    ref_model = copy.deepcopy(actor_model)
    ref_model.eval(); ref_model.requires_grad_(False)
    
    # Reward Model
    print(f"Loading Reward Model from {REWARD_MODEL_PATH}...")
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH, torch_dtype=model_dtype, num_labels=1, trust_remote_code=True
    ).to(device)
    reward_model.eval(); reward_model.requires_grad_(False)
    
    # --- 关键修复开始 ---
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH)
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
    
    # 1. 设置 Actor Tokenizer
    tokenizer.padding_side = 'left' 
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
        
    # 2. 设置 Reward Tokenizer
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    
    # 3. 显式更新 Reward Model 配置，防止报错
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
    # --- 关键修复结束 ---

    # Optimizer
    optimizer = torch.optim.AdamW(actor_model.parameters(), lr=1e-5)  # 增大学习率
    
    writer = SummaryWriter(LOG_DIR)
    buffer = ExperienceBuffer(limit=20)
    
    # 加载 Prompts
    try:
        with open("./prompts.json", "r", encoding="utf-8") as f:
            prompt_list = json.load(f)
    except:
        prompt_list = ['请解释什么是量子纠缠？', '写一首关于秋天的诗。']

    steps = 0
    episodes = 100
    
    for episode in range(episodes):
        print(f"=== Episode {episode+1} ===")
        for i, prompt in enumerate(prompt_list):
            print(f"Processing prompt {i+1}/{len(prompt_list)} (Group Size {GROUP_SIZE})...")
            
            exps = generate_group_experiences([prompt], actor_model, ref_model, reward_model, tokenizer, reward_tokenizer, device)
            buffer.append(exps)
            
            if len(buffer) > 0:
                batch_data = buffer.buffer[-1]
                batch_map = {
                    'seqs': batch_data['seqs'].to(device),
                    'action_log_probs': batch_data['action_log_probs'].to(device),
                    'advantages': batch_data['advantages'].to(device),
                    'attention_mask': batch_data['attention_mask'].to(device),
                    'action_mask': batch_data['action_mask'].to(device),
                    'ref_log_probs': batch_data['ref_log_probs'].to(device)
                }
                
                train_step(batch_map, actor_model, optimizer, device, steps, writer)
                steps += 1
                
                buffer.clear()
                torch.cuda.empty_cache()

    print(f"Saving GRPO model to {SAVE_PATH}...")
    actor_model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    writer.close()

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        import traceback
        traceback.print_exc()