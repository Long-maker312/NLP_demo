# CBOW模型填空题答案

## Vocabulary类
1. ​**mask_token对应的索引通过调用add_token方法赋值给`self.____`属性**  
   ​**答案**: `mask_index`

2. ​**lookup_token方法中，如果`self.unk_index >=0`，则对未登录词返回____**  
   ​**答案**: `self.unk_index`

3. ​**调用`add_many`方法添加多个token时，实际是通过循环调用____方法实现**  
   ​**答案**: `add_token`

---

## CBOWVectorizer类
4. ​**`vectorize`方法中，当`vector_length < 0`时，最终向量长度等于____的长度**  
   ​**答案**: `context_indices`

5. ​**`from_dataframe`方法构建词表时，会遍历DataFrame中____和____两列的内容**  
   ​**答案**: `target`、`context`

6. ​**`out_vector[len(indices):]`的部分填充为`self.cbow_vocab.____`**  
   ​**答案**: `mask_index`

---

## CBOWDataset类
7. ​**`_max_seq_length`通过计算所有`context`列的____的最大值得出**  
   ​**答案**: `长度`

8. ​**`set_split`方法通过`self._lookup_dict`选择对应的____和____**  
   ​**答案**: `contexts`、`targets`

9. ​**`__getitem__`返回的字典中，`y_target`通过查找____列的token得到**  
   ​**答案**: `target`

---

## 模型结构
10. ​**CBOWClassifier的`forward`中，`x_embedded_sum`的计算方式是`embedding(x_in).____(dim=1)`**  
    ​**答案**: `sum`

11. ​**模型输出层`fc1`的`out_features`等于____参数的值**  
    ​**答案**: `num_classes`

---

## 训练流程
12. ​**`generate_batches`函数通过PyTorch的____类实现批量加载**  
    ​**答案**: `DataLoader`

13. ​**训练时`classifier.train()`的作用是启用____和____模式**  
    ​**答案**: `dropout`、`Batch Normalization`

14. ​**反向传播前必须执行____.zero_grad()清空梯度**  
    ​**答案**: `optimizer`

15. ​**`compute_accuracy`中`y_pred_indices`通过____方法获取预测类别**  
    ​**答案**: `argmax`

---

## 训练状态管理
16. ​**`make_train_state`中`early_stopping_best_val`初始化为____**  
    ​**答案**: `inf`

17. ​**`update_train_state`在连续____次验证损失未下降时会触发早停**  
    ​**答案**: `3`

18. ​**当验证损失下降时，`early_stopping_step`会被重置为____**  
    ​**答案**: `0`

---

## 设备与随机种子
19. ​**`set_seed_everywhere`中与CUDA相关的设置是____.manual_seed_all(seed)**  
    ​**答案**: `torch.cuda`

20. ​**`args.device`的值根据____.is_available()确定**  
    ​**答案**: `torch.cuda`

---