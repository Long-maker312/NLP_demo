import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def get_words(filename):
    """读取文本并过滤无效字符和长度为1的词"""
    words = []
    with open(filename, 'r', encoding='utf-8') as fr:
        for line in fr:
            line = line.strip()
            # 过滤无效字符
            line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
            # 使用jieba.cut()方法对文本切词处理
            line = cut(line)
            # 过滤长度为1的词
            line = filter(lambda word: len(word) > 1, line)
            words.extend(line)
    return words


def feature_extraction(method="high_freq", top_n=100):
    """
    特征提取，支持高频词和TF-IDF两种方法
    :param method: 特征提取方法，可选"high_freq"或"tfidf"
    :param top_n: 选取的特征数量
    :return: 特征矩阵、标签和向量化器
    """
    filename_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    all_words = []
    for filename in filename_list:
        all_words.append(get_words(filename))

    if method == "high_freq":
        freq = Counter(chain(*all_words))
        top_words = [i[0] for i in freq.most_common(top_n)]
        vector = []
        for words in all_words:
            word_map = list(map(lambda word: words.count(word), top_words))
            vector.append(word_map)
        vector = np.array(vector)
        vectorizer = None
    elif method == "tfidf":
        texts = [' '.join(words) for words in all_words]
        vectorizer = TfidfVectorizer(max_features=top_n)
        vector = vectorizer.fit_transform(texts).toarray()
    else:
        raise ValueError("method must be 'high_freq' or 'tfidf'")

    # 0 - 126.txt为垃圾邮件标记为1；127 - 151.txt为普通邮件标记为0
    labels = np.array([1] * 127 + [0] * 24)
    return vector, labels, vectorizer


def train_model(vector, labels):
    model = MultinomialNB()
    model.fit(vector, labels)
    return model


def predict(model, filename, vectorizer=None, top_words=None, method="high_freq"):
    """对未知邮件分类"""
    words = get_words(filename)
    if method == "high_freq":
        current_vector = np.array(
            tuple(map(lambda word: words.count(word), top_words)))
    elif method == "tfidf":
        text = ' '.join(words)
        current_vector = vectorizer.transform([text]).toarray()
    # 预测结果
    result = model.predict(current_vector.reshape(1, -1))
    return '垃圾邮件' if result == 1 else '普通邮件'


if __name__ == "__main__":
    # 选择特征提取方法，可修改为"tfidf"
    feature_method = "high_freq"
    vector, labels, vectorizer = feature_extraction(method=feature_method, top_n=100)
    model = train_model(vector, labels)

    if feature_method == "high_freq":
        freq = Counter(chain(*[get_words('邮件_files/{}.txt'.format(i)) for i in range(151)]))
        top_words = [i[0] for i in freq.most_common(100)]
    else:
        top_words = None

    print('151.txt分类情况:{}'.format(predict(model, '邮件_files/151.txt', vectorizer, top_words, feature_method)))
    print('152.txt分类情况:{}'.format(predict(model, '邮件_files/152.txt', vectorizer, top_words, feature_method)))
    print('153.txt分类情况:{}'.format(predict(model, '邮件_files/153.txt', vectorizer, top_words, feature_method)))
    print('154.txt分类情况:{}'.format(predict(model, '邮件_files/154.txt', vectorizer, top_words, feature_method)))
    print('155.txt分类情况:{}'.format(predict(model, '邮件_files/155.txt', vectorizer, top_words, feature_method)))