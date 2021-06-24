# Kaggle CommonLit Readability Prize

比赛：https://www.kaggle.com/c/commonlitreadabilityprize

实验结果：https://drive.google.com/file/d/1H8sOUMPg9LDu9q4bbpMKobMKdbiQKDeB/view?usp=sharing

# 完成事项

- [x] 获取最后embedding的方式
  - [x] last-avg
  - [x] first-last-avg
  - [x] 倒数n层加权平均
- [x] 增加eval次数
- [x] mlm预训练 
- [x] 损失函数
  - [x] MSE
  - [x] RMSE
  - [x] Gaussian Kl divergence
    - [x] 模型仅输出均值
    - [x] 模型同时输出均值和方差
- [x] layerwise learning rate decay
- [x] 最后n层随机初始化

# 待办事项

- [ ] 提交结果

  暂时因为网络问题没有提交结果
  
- [ ] roberta-large模型
- [ ] 回归转分类
- [ ] pseudo label
- [ ] stochastic weight average
- [ ] 超参数搜索(?)
