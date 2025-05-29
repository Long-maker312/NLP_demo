# 导入所需库
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 初始化模型和分词器（中文BERT-base模型）
model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 2分类任务（正面/负面）


# 定义情感分类函数
def classify_sentiment(sentence):
    # 数据预处理
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).tolist()[0]

    # 结果映射：0=负面，1=正面
    label_map = {0: "负面", 1: "正面"}
    predicted_label = label_map[torch.argmax(logits).item()]
    confidence = round(probabilities[torch.argmax(logits).item()] * 100, 2)

    return f"句子：{sentence}\n分类结果：{predicted_label}（置信度：{confidence}%）"


# 学号尾号5对应的句子
movie_review = "剧情拖沓冗长，中途几次差点睡着。"
food_review = "食物份量十足，性价比超高，吃得很满足！"

# 执行分类
print("==== 影评情感分类 ====")
print(classify_sentiment(movie_review))
print("\n==== 外卖评价情感分类 ====")
print(classify_sentiment(food_review))