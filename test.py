import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 配置 ---
BASE_MODEL_PATH = './Qwen2.5-0.5B-Instruct'  # 原始模型路径
PPO_MODEL_PATH = './ppo_model_final'         # 训练后保存的路径
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model(path, device):
    print(f"Loading model from {path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            path, 
            torch_dtype=torch.float16, # 测试时用 FP16 即可
            device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(path)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model from {path}: {e}")
        return None, None

def generate_response(model, tokenizer, prompt, device):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,
            do_sample=True, # 如果想看确定的变化，可以改为 False
            temperature=0.7,
            top_p=0.9
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def main():
    # 1. 加载原始模型
    base_model, base_tokenizer = load_model(BASE_MODEL_PATH, device)
    
    # 2. 加载 PPO 模型
    ppo_model, ppo_tokenizer = load_model(PPO_MODEL_PATH, device)
    
    if base_model is None or ppo_model is None:
        print("Model loading failed.")
        return

    # 3. 测试 Prompt
    test_prompts = [
    "请用简单的语言解释什么是量子纠缠？",
    "写一首关于程序员深夜加班的七言绝句。",
    "如果这世界上没有了电，人类生活会变成什么样？",
    "请帮我把这句话翻译成英文：'不到长城非好汉'。",
    "作为一名客服，委婉地拒绝客户要求全额退款的无理请求。",
    "列出三个去北京旅游必打卡的景点，并说明理由。",
    "如何用 Python 读取一个 CSV 文件？请给出代码示例。",
    "分析一下《西游记》中孙悟空的性格特点。",
    "我感冒了，头很痛，我该怎么办？",
    "请为一款新型的运动鞋写一段吸引人的广告语。",
    "解释一下为什么天空是蓝色的？",
    "教我做一道简单的西红柿炒鸡蛋。",
    "给你的老板写一封请假条，理由是家里有急事。",
    "人工智能未来会取代人类的工作吗？说说你的看法。",
    "请生成一个强密码，包含字母、数字和符号。"
    ]
    
    print("\n" + "="*50)
    print("STARTING COMPARISON TEST")
    print("="*50 + "\n")

    for prompt in test_prompts:
        print(f"Question: {prompt}")
        print("-" * 30)
        
        # 原始模型回答
        base_resp = generate_response(base_model, base_tokenizer, prompt, device)
        print(f"[Base Model]: {base_resp}")
        
        # PPO 模型回答
        ppo_resp = generate_response(ppo_model, ppo_tokenizer, prompt, device)
        print(f"[PPO Model ]: {ppo_resp}")
        
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()