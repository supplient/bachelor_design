* 功能点识别模块
    * epoch_checkpoint：用于在训练时记录logs以及保存结果
    * Train：Fine-tune BERT-CRF模型用
* 等效需求识别模块
    * char_emb：输入字符，输出字向量
    * SIF：由字向量合成句向量
    * dist_cal：计算两个向量之间的距离
    * equal_generate: 通过翻译过去再翻译回来，生成一句话的等效的一句话
    * EqualTrain：训练等效需求识别模块的参数用
* 工具
    * config: 存放各种设置
    * driver_amount: 用于在colab环境中挂载gdriver
    * preprocess：预处理数据，主要用于BERT输入
    * secret：不包含于github库中。存放各类隐私信息，例如搜狗翻译的用户key
    * util：存放各类通用工具，例如log