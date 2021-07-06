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

# 使用说明

## 目录结构

- project
  - code(github项目)
    - main.py
    - ...
  - data
    - train_data.csv
    - ...

如无特别原因的话，请尽量组织成这种结构，因为yaml文件里用了大量相对路径。

## 安装

pip install -r requirements.txt

请注意requirements.txt里的torch版本可能需要根据自己服务器上的情况进行修改。

## 使用

### 数据预处理

因为使用了k_fold训练，首先对原始数据进行了预处理加入fold信息。

python eda.py --mode=add_fold --config_path=configs/add_fold.yml

只需要执行一次，执行完以后会在add_fold.yml中的data_save_path生成新的csv文件，用这个作为训练文件。

### MLM(Masked Language Model)预训练

这步是将已有的预训练模型在自己的文本上接着做MLM的预训练，是个常用的能提升结果的技巧，一般来说加上比较好。

example:

python run_mlm_no_trainer.py --train_file=../data/train.csv --validation_file=../data/train.csv --model_name_or_path=roberta-base --output_dir=checkpoints/pretrained/roberta_base --per_device_train_batch_size=4 --gradient_accumulation_steps=1 --num_train_epochs=3

修改model_name_or_path的参数来训练不同的模型。可选模型名：https://huggingface.co/transformers/pretrained_models.html

output_dir参数自己设定。

根据模型的大小可能需要调整per_device_train_batch_size和gradient_accumulation_steps参数。
  - actual_batch_size = per_device_train_batch_size x gradient_accumulation_steps

根据不同的模型也需要调整num_train_epochs的参数。可以先用roberta-base训一次记录下perplexity的结果，训练别的模型时可参考roberta-base的perplexity结果来设定训练轮数。

### 训练

example:

单卡

python main.py --config_path=configs/debug_reg.yml --mode=train --gpu=0

多卡

python main.py --config_path=configs/debug_reg.yml --mode=train --gpu=0,1 --dp

config_path换成对应的config。

使用fitlog进行实验记录。fitlog github: https://github.com/fastnlp/fitlog

在code文件夹下命令行执行 fitlog log logs

会在localhost:5000生成一个服务，访问localhost:5000可以查看当前和历史实验的记录。

右侧action的第二个按钮trend较常用，可实时查看损失函数和metric的变化情况。

### 测试

同训练，只需要将mode改为eval。

python main.py --config_path=configs/debug_reg.yml --mode=eval --gpu=0
