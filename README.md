# WebData

垃圾邮件分析
运行命令
python interface.py
根据提示输入需要判断的文本
系统会返回spam 或者unspam


参数效果评估
运行命令
python feature.py 参数文件 输出文件 端点记录位置 输入文件

参数文件格式
模型 spam类别权重 降维模型 降维后维度
(模型可选参数为L, S, R对应 逻辑回归,支持向量机,随机森林)
(降维模型可选NMF, SVD)
例如：用逻辑回归模型分类，用NMF降维成20维,两类权重相同
L 1 NMF 20


