import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification, AutoTokenizer
from dataclasses import dataclass
import copy
import gc
import os
import json

# --- 配置 ---
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
model_dtype = torch.bfloat16 if use_bf16 else torch.float16
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 路径配置
SAVE_PATH = "./new_ppo_model"          # PPO 训练后模型保存路径
LOG_DIR = "./runs/new_ppo_experiment"        # 日志路径
REWARD_MODEL_PATH = "./my_custom_reward_model" # <--- 修改：指向你刚才训练好的奖励模型路径
ACTOR_MODEL_PATH = "./Qwen2.5-0.5B-Instruct"  # Actor 模型路径

print(f"Using Device: {device}")
print(f"Using Dtype: {model_dtype}")
print(f"Reward Model Path: {REWARD_MODEL_PATH}")

# --- 类定义 ---

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

class Critic(nn.Module):
    def __init__(self, model_name_or_path, dtype):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(model_name_or_path, torch_dtype=dtype, trust_remote_code=True)
        self.value_head = nn.Linear(self.base_model.config.hidden_size, 1).to(dtype=dtype)
    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_state = outputs.last_hidden_state.to(self.value_head.weight.dtype)
        return self.value_head(hidden_state).squeeze(-1)

@dataclass
class Samples:
    seqs: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    num_actions: int
    response_length: torch.Tensor
    total_length: torch.Tensor

@dataclass
class Experience:
    seqs: torch.Tensor
    action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    reward: torch.Tensor
    num_actions: int

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
                    t = v.detach().cpu()
                    if t.dim() == 2 and t.shape[0] == 1: t = t.squeeze(0)
                    item[k] = t
                else: item[k] = v
            batch.append(item)
        self.buffer.extend(batch)
        if len(self.buffer) >= self.limit: self.buffer = self.buffer[-self.limit:]
    def clear(self):
        self.buffer = []
        gc.collect()
    def __len__(self): return len(self.buffer)
    def __getitem__(self, index): return self.buffer[index]

# --- 核心函数 ---

def get_advantages_and_returns(values, rewards, action_mask, gamma, lambd):
    lastgaelam = 0
    advantages_reversed = []
    response_length = rewards.size(1)
    if action_mask is not None:
        values = action_mask * values
        rewards = action_mask * rewards
    for t in reversed(range(response_length)):
        nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
        delta = rewards[:, t] + gamma * nextvalues - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], dim=1)
    returns = advantages + values
    return advantages.detach(), returns

def generate_samples(prompts, model, tokenizer, max_length, max_new_tokens, device):
    samples_list = []
    model.eval()
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors='pt', max_length=max_length, truncation=True).to(device)
        input_ids = inputs.input_ids
        prompt_len = input_ids.shape[1]
        with torch.no_grad():
            seqs = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id, do_sample=True)
        attention_mask = (seqs != tokenizer.pad_token_id).long()
        action_mask = torch.zeros_like(attention_mask)
        action_mask[:, prompt_len:] = attention_mask[:, prompt_len:]
        samples = Samples(seqs=seqs, attention_mask=attention_mask, action_mask=action_mask, num_actions=seqs.shape[1] - prompt_len, response_length=action_mask.float().sum(dim=-1), total_length=attention_mask.float().sum(dim=-1))
        samples_list.append(samples)
    return samples_list

def generate_experiences(samples_list, actor_model, ref_model, critic_model, reward_model, actor_tokenizer, reward_tokenizer, device, kl_ctl=0.1):
    actor_model.eval(); ref_model.eval(); reward_model.eval(); critic_model.eval()
    experiences = []
    with torch.no_grad():
        for samples in samples_list:
            seqs = samples.seqs.to(device)
            attention_mask = samples.attention_mask.to(device)
            action_mask = samples.action_mask.to(device)
            
            logits = actor_model(seqs, attention_mask=attention_mask).logits
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_all = log_probs[:, :-1, :].gather(2, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
            curr_action_mask = action_mask[:, 1:]
            
            ref_logits = ref_model(seqs, attention_mask=attention_mask).logits
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            ref_log_probs_all = ref_log_probs[:, :-1, :].gather(2, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
            
            values = critic_model(seqs, attention_mask)[:, :-1]
            
            seq_texts = actor_tokenizer.batch_decode(seqs, skip_special_tokens=True)
            # 这里的 reward_model 现在是你自己的 Qwen-based 模型
            rm_inputs = reward_tokenizer(seq_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            rm_score = reward_model(**rm_inputs).logits
            
            kl = log_probs_all - ref_log_probs_all
            rewards = -kl_ctl * kl
            for i in range(rewards.shape[0]):
                valid_idx = torch.nonzero(curr_action_mask[i]).squeeze()
                if valid_idx.numel() > 0:
                    last_idx = valid_idx[-1] if valid_idx.dim() > 0 else valid_idx
                    rewards[i, last_idx] += rm_score[i, 0]
            
            advantages, returns = get_advantages_and_returns(values, rewards, curr_action_mask, gamma=0.99, lambd=0.95)
            experiences.append(Experience(seqs=seqs, action_log_probs=log_probs_all, values=values, returns=returns, advantages=advantages, attention_mask=attention_mask, action_mask=curr_action_mask, reward=rm_score, num_actions=samples.num_actions))
    return experiences

def collate_fn(batch):
    def pad_sequence(seq_list, padding_value):
        return torch.nn.utils.rnn.pad_sequence(seq_list, batch_first=True, padding_value=padding_value)
    
    return {
        'seqs': pad_sequence([x['seqs'] for x in batch], 0),
        'action_log_probs': pad_sequence([x['action_log_probs'] for x in batch], 0.0),
        'values': pad_sequence([x['values'] for x in batch], 0.0),
        'returns': pad_sequence([x['returns'] for x in batch], 0.0),
        'advantages': pad_sequence([x['advantages'] for x in batch], 0.0),
        'attention_mask': pad_sequence([x['attention_mask'] for x in batch], 0),
        'action_mask': pad_sequence([x['action_mask'] for x in batch], 0),
    }

def train_step(batch, actor_model, critic_model, opt_actor, opt_critic, device, step, writer):
    seqs = batch['seqs'].to(device)
    old_log_probs = batch['action_log_probs'].to(device)
    returns = batch['returns'].to(device)
    advantages = batch['advantages'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    action_mask = batch['action_mask'].to(device)
    
    # Actor Update
    actor_model.train()
    logits = actor_model(seqs, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits, dim=-1)
    new_log_probs = log_probs[:, :-1, :].gather(2, seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
    
    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
    policy_loss = -torch.min(surr1, surr2)
    policy_loss = (policy_loss * action_mask).sum() / (action_mask.sum() + 1e-8)
    
    opt_actor.zero_grad()
    policy_loss.backward()
    opt_actor.step()
    
    # Critic Update
    critic_model.train()
    new_values = critic_model(seqs, attention_mask)[:, :-1]
    value_loss = (new_values - returns) ** 2
    value_loss = (value_loss * action_mask).sum() / (action_mask.sum() + 1e-8)
    
    opt_critic.zero_grad()
    value_loss.backward()
    opt_critic.step()
    
    # --- Visualization ---
    writer.add_scalar("loss/policy", policy_loss.item(), step)
    writer.add_scalar("loss/value", value_loss.item(), step)
    print(f"step: {step}  policy_loss: {policy_loss.item():.4f}  value_loss: {value_loss.item():.4f}")

def train():
    print("Loading Actor Model...")
    actor_model = AutoModelForCausalLM.from_pretrained(ACTOR_MODEL_PATH, torch_dtype=model_dtype, trust_remote_code=True).to(device)
    actor_model.gradient_checkpointing_enable(); actor_model.config.use_cache = False
    
    ref_model = copy.deepcopy(actor_model); ref_model.eval(); ref_model.requires_grad_(False)
    
    print("Loading Critic Model...")
    critic_model = Critic(ACTOR_MODEL_PATH, model_dtype).to(device)
    critic_model.base_model.gradient_checkpointing_enable(); critic_model.base_model.config.use_cache = False
    
    # --- 修改部分：加载你自己的 Reward Model ---
    print(f"Loading Custom Reward Model from {REWARD_MODEL_PATH}...")
    if not os.path.exists(REWARD_MODEL_PATH):
        raise FileNotFoundError(f"找不到奖励模型路径: {REWARD_MODEL_PATH}，请先运行 train_reward_model.py")

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD_MODEL_PATH, 
        torch_dtype=model_dtype,
        num_labels=1,
        trust_remote_code=True # 因为是 Qwen，需要这个
    ).to(device)
    reward_model.eval(); reward_model.requires_grad_(False)

    actor_tokenizer = AutoTokenizer.from_pretrained(ACTOR_MODEL_PATH)
    # 加载你自己的 Reward Model Tokenizer (虽然也是 Qwen，但最好从保存路径加载)
    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD_MODEL_PATH)
    
    actor_tokenizer.padding_side = 'left'
    if actor_tokenizer.pad_token is None: actor_tokenizer.pad_token = actor_tokenizer.eos_token
    # ----------------------------------------
        
    optimizer_actor = torch.optim.AdamW(actor_model.parameters(), lr=1e-5)
    optimizer_critic = torch.optim.AdamW(critic_model.parameters(), lr=1e-5)
    
    if os.path.exists(LOG_DIR):
        import shutil
        shutil.rmtree(LOG_DIR)
    writer = SummaryWriter(LOG_DIR)
    
    buffer = ExperienceBuffer(limit=50)
    
    print("Loading prompts from prompts.json...")
    try:
        # 这里我改为了根目录，如果你确实放在 data 目录下，请改为 ./data/prompts.json
        with open("./prompts.json", "r", encoding="utf-8") as f:
            prompt_list = json.load(f)
        print(f"Loaded {len(prompt_list)} prompts.")
    except FileNotFoundError:
        print("prompts.json not found, using default list.")
        prompt_list = [
            '请问1+1等于多少？',
            '为什么所有的镜子都是矩形的？',
            '写一首关于秋天的短诗。'
        ]
    
    steps = 0
    episodes = 3 
    micro_rollout_batch_size = 1
    
    for episode in range(episodes):
        print(f"=== Episode {episode+1}/{episodes} ===")
        batch_prompts = [prompt_list[i:i+micro_rollout_batch_size] for i in range(0, len(prompt_list), micro_rollout_batch_size)]
        
        for prompts in batch_prompts:
            samples = generate_samples(prompts, actor_model, actor_tokenizer, 128, 32, device)
            exps = generate_experiences(samples, actor_model, ref_model, critic_model, reward_model, actor_tokenizer, reward_tokenizer, device)
            buffer.append(exps)
            torch.cuda.empty_cache()
            
        if len(buffer) > 0:
            data_dict = collate_fn(buffer.buffer)
            dataset = torch.utils.data.TensorDataset(
                data_dict['seqs'], data_dict['action_log_probs'], 
                data_dict['values'], data_dict['returns'], 
                data_dict['advantages'], data_dict['attention_mask'], 
                data_dict['action_mask']
            )
            train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
            
            for epoch in range(2): 
                for batch in train_loader:
                    batch_map = {'seqs': batch[0], 'action_log_probs': batch[1], 'values': batch[2], 'returns': batch[3], 'advantages': batch[4], 'attention_mask': batch[5], 'action_mask': batch[6]}
                    train_step(batch_map, actor_model, critic_model, optimizer_actor, optimizer_critic, device, steps, writer)
                    steps += 1
            buffer.clear()
            torch.cuda.empty_cache()

    # --- 保存模型 ---
    print(f"Saving model to {SAVE_PATH}...")
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    
    actor_model.save_pretrained(SAVE_PATH)
    actor_tokenizer.save_pretrained(SAVE_PATH)
    print("Model saved successfully.")
    
    writer.close()

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        import traceback
        traceback.print_exc()