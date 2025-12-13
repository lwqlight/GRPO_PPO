import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import os

# --- 配置 ---
# 我们使用 Qwen-0.5B 作为奖励模型的基底（它也需要懂语言才能做裁判）
# 注意：这里我们不再用 Deberta，而是训练一个我们自己的 Qwen 版本裁判
MODEL_NAME = './Qwen2.5-0.5B-Instruct' 
SAVE_PATH = './my_custom_reward_model'
REWARD_DATA_PATH = './data/reward_data.json' # 训练数据路径
MAX_LENGTH = 512
BATCH_SIZE = 1 
EPOCHS = 100
LR = 5e-6  # <--- 修改1：降低学习率，更稳

# <--- 修改2：自动检测最佳精度 ---
if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
    dtype = torch.bfloat16
    print("使用 BF16 精度 (稳定)")
else:
    dtype = torch.float16
    print("使用 FP16 精度 (注意溢出风险)")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 1. 数据集定义 (保持不变) ---
class RewardDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item['prompt']
        chosen = item['chosen']
        rejected = item['rejected']
        
        # 简单拼接格式
        chosen_text = f"User: {prompt}\nAssistant: {chosen}"
        rejected_text = f"User: {prompt}\nAssistant: {rejected}"

        enc_chosen = self.tokenizer(
            chosen_text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt'
        )
        enc_rejected = self.tokenizer(
            rejected_text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt'
        )

        return {
            'input_ids_chosen': enc_chosen['input_ids'].squeeze(0),
            'attention_mask_chosen': enc_chosen['attention_mask'].squeeze(0),
            'input_ids_rejected': enc_rejected['input_ids'].squeeze(0),
            'attention_mask_rejected': enc_rejected['attention_mask'].squeeze(0)
        }

# --- 2. 训练函数 ---
def train():
    print(f"Loading base model from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=1, 
        torch_dtype=dtype, # 使用自动检测的精度
        trust_remote_code=True
    ).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    model.gradient_checkpointing_enable()

    dataset = RewardDataset(REWARD_DATA_PATH, tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    model.train()
    print("Start Training Reward Model...")

    for epoch in range(EPOCHS):
        for step, batch in enumerate(dataloader):
            input_ids_chosen = batch['input_ids_chosen'].to(device)
            mask_chosen = batch['attention_mask_chosen'].to(device)
            input_ids_rejected = batch['input_ids_rejected'].to(device)
            mask_rejected = batch['attention_mask_rejected'].to(device)

            optimizer.zero_grad()

            # 前向传播
            rewards_chosen = model(input_ids_chosen, attention_mask=mask_chosen).logits
            rewards_rejected = model(input_ids_rejected, attention_mask=mask_rejected).logits

            # <--- 修改3：关键修复！强制转为 float32 计算 Loss ---
            # 无论模型本身是什么精度，计算 loss 时必须用 float32，防止 NaN
            r_chosen_f32 = rewards_chosen.float()
            r_rejected_f32 = rewards_rejected.float()
            
            # Loss 计算
            loss = -torch.nn.functional.logsigmoid(r_chosen_f32 - r_rejected_f32).mean()

            # 检查 loss 是否已经是 nan (虽然转了 float32 一般不会，但以防万一)
            if torch.isnan(loss):
                print("Warning: Loss is NaN, skipping this step!")
                optimizer.zero_grad()
                continue

            loss.backward()
            
            # <--- 修改4：梯度裁剪 (防止梯度爆炸) ---
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            if step % 2 == 0:
                print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}")
                diff = (r_chosen_f32 - r_rejected_f32).mean().item()
                print(f"   >> Score Diff: {diff:.4f}")

    print(f"Saving Reward Model to {SAVE_PATH}...")
    model.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("Done!")

if __name__ == '__main__':
    # 训练前清理显存
    torch.cuda.empty_cache()
    try:
        train()
    except Exception as e:
        import traceback
        traceback.print_exc()