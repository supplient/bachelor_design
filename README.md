* 功能点识别模块
    * epoch_checkpoint.py：用于在训练时记录logs以及保存模型
    * Train.ipynb：Fine-tune BERT-CRF模型用
    * Continue_Train.ipynb: 用于继续训练
* 成本估计模型
    * funcomo.py: 成本估计模型的代码
* 工具
    * config: 存放各种设置
    * driver_amount: 用于在colab环境中挂载gdriver
    * cut_and_tag: 用于原始数据的分词、预处理
    * preprocess：预处理，主要是对BERT输入的预处理
    * util：存放各类通用工具，例如log，么，实际上没用上就是了