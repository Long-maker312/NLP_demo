from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings

# 忽略警告信息
warnings.filterwarnings("ignore")

# 使用公开可用且稳定的中文GPT模型
model_name = "uer/gpt2-chinese-cluecorpussmall"  # 原始模型不可用，尝试镜像版本
# 如果上述模型仍不可用，可替换为下面这个
# model_name = "uer/gpt2-chinese-poem"  # 专门用于中文诗歌生成的模型

# 检测是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    # 尝试加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


    # 文本续写函数
    def generate_text(prompt, max_length=80, temperature=0.7):
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


    # 根据学号尾号5对应的续写开头
    prompt = "假如我能隐身一天，我会"
    generated_text = generate_text(prompt)

    print("==== 文本续写结果 ====")
    print(generated_text)

except Exception as e:
    print(f"模型加载失败: {e}")
    print("\n尝试以下解决方案:")
    print("1. 确认网络连接正常，且能够访问Hugging Face")
    print("2. 尝试更换其他公开可用的中文GPT模型")
    print("3. 手动下载模型文件到本地，并指定本地路径加载")