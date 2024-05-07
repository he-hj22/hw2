项目结构

- dataset 数据集文件夹
- model_output 模型输出文件夹
  - MLP mlp模型的输出
    - config.txt 记录args和测试集上的表现
    - performance记录在训练过程中模型在训练集和验证集上的表现
- dataset.py 实现TextDataset数据集
- main.py 实现训练和评价模型
- model.py 定义四种模型
- utils.py 实现辅助函数，包括文件读取、曲线绘制等

运行指令

`python main.py --model TextCNN --epochs 20 等`
