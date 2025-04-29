


# 贝叶斯邮件分类系统

本次作业实现了一个基于朴素贝叶斯算法的邮件分类系统，可以将邮件分为垃圾邮件和普通邮件两类。系统支持两种特征提取模式：高频词模式和TF-IDF模式。

## 核心功能说明

### 1. 文本预处理
- 使用了不同jieba分词工具进行中文分词
- 过滤无效字符和长度为1的词
- 支持自定义词典加载

### 2. 特征提取
系统支持两种特征提取模式：

#### 高频词模式
- 统计所有训练文本中出现频率最高的N个词
- 使用词频作为特征值
- 特征向量维度为N

#### TF-IDF模式
- 计算每个词的TF-IDF值
- 选择TF-IDF值最高的N个词作为特征
- 使用TF-IDF值作为特征值
- 特征向量维度为N

### 3. 分类模型
- 使用sklearn的MultinomialNB实现朴素贝叶斯分类
- 支持模型训练和预测
- 输出分类结果（垃圾邮件/普通邮件）

## 特征模式切换方法

在`classify.py`文件中，可以通过修改特征提取部分的代码来切换特征模式：

### 高频词模式
```python
def get_top_words(top_num):
    """遍历邮件建立词库后返回出现次数最多的词"""
    filename_list = [r'邮件_files/{}.txt'.format(i) for i in range(151)]
    for filename in filename_list:
        all_words.append(get_words(filename))
    freq = Counter(chain(*all_words))
    return [i[0] for i in freq.most_common(top_num)]
```

### TF-IDF模式
```python
from sklearn.feature_extraction.text import TfidfVectorizer

def get_tfidf_features(top_num):
    """使用TF-IDF提取特征"""
    vectorizer = TfidfVectorizer(max_features=top_num)
    X = vectorizer.fit_transform([' '.join(words) for words in all_words])
    return vectorizer.get_feature_names_out()
```

切换方法：
1. 在代码中注释掉当前使用的特征提取函数
2. 取消注释要使用的特征提取函数
3. 修改相应的特征向量构建代码

## 传统数学公式

### 朴素贝叶斯分类器
$$
P(y|x_1, x_2, ..., x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i|y)}{P(x_1, x_2, ..., x_n)}
$$

### TF-IDF计算公式
$$
\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \text{IDF}(t)
$$

其中：
- $\text{TF}(t,d)$ 是词t在文档d中的词频
- $\text{IDF}(t) = \log\frac{N}{1 + \text{DF}(t)}$
- N是文档总数
- $\text{DF}(t)$ 是包含词t的文档数

## 选做作业部分

### 1. 样本平衡处理（SMOTE过采样）

#### 问题背景
在邮件分类任务中，存在严重的样本不平衡问题：
- 垃圾邮件：127条
- 普通邮件：24条
这种不平衡会导致模型偏向于预测多数类（垃圾邮件），影响分类效果。

#### SMOTE过采样实现
系统使用SMOTE（Synthetic Minority Over-sampling Technique）算法来解决样本不平衡问题：

```python
from imblearn.over_sampling import SMOTE

# 使用SMOTE进行过采样
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

#### 实现原理
1. 对少数类样本进行分析
2. 在特征空间中合成新的少数类样本
3. 保持多数类样本不变
4. 最终实现两类样本数量平衡

#### 注意事项
- 仅对训练集进行过采样
- 保持测试集原始分布
- 使用random_state确保结果可复现

### 2. 模型评估指标

#### 分类报告
系统提供详细的分类评估报告，包括：
- 精确率（Precision）：正确预测为正例的样本占所有预测为正例样本的比例
- 召回率（Recall）：正确预测为正例的样本占所有实际为正例样本的比例
- F1分数（F1-Score）：精确率和召回率的调和平均数
- 支持度（Support）：每个类别的样本数量

#### 混淆矩阵
系统自动生成混淆矩阵可视化：
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
```

#### 详细指标计算
```python
from sklearn.metrics import precision_recall_fscore_support

# 计算每个类别的详细指标
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred)

# 输出普通邮件指标
print(f"普通邮件:")
print(f"  精确率: {precision[0]:.4f}")
print(f"  召回率: {recall[0]:.4f}")
print(f"  F1分数: {f1[0]:.4f}")

# 输出垃圾邮件指标
print(f"垃圾邮件:")
print(f"  精确率: {precision[1]:.4f}")
print(f"  召回率: {recall[1]:.4f}")
print(f"  F1分数: {f1[1]:.4f}")
```

#### 评估相关指标计算公式

##### 精确率（Precision）
$$
Precision = \frac{TP}{TP + FP}
$$
- TP：真正例（正确预测为正例）
- FP：假正例（错误预测为正例）

##### 召回率（Recall）
$$
Recall = \frac{TP}{TP + FN}
$$
- FN：假负例（错误预测为负例）

##### F1分数（F1-Score）
$$
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
$$

## 使用说明

1. 安装依赖：
```bash
pip install jieba scikit-learn imbalanced-learn matplotlib seaborn
```

2. 运行分类程序：
```bash
python classify.py
```

3. 运行带评估的版本：
```bash
python classify_evaluation.py
```

4. 查看混淆矩阵：
- 运行程序后会自动生成`confusion_matrix.png`
- 包含详细的分类评估报告
- 可详细显示具体的分类情况
