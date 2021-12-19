import numpy as np


class StrongClassifier():
    def __init__(self, weak=[], weakweight=[], weaknum=0, threshold=0, PR=0, FPR=1, index=[]):
        self.weak = weak # 弱分类器列表
        self.weakweight = weakweight # 弱分类器权重
        self.weaknum = weaknum # 弱分类器个数
        self.threshold = threshold # 阈值
        self.PR = PR # 检测率
        self.FPR = FPR # 误检率
        self.index = index # 弱分类器编号


def StrongJudg(StrongClassifier, JudgOut, num, num1):
    '''
    获取单个强分类器检测结果 计算检测率和误检率 JudgOut保存的是300个弱分类器对每个样本的评分
    '''
    JudgFeature = np.zeros(num)
    for i in range(StrongClassifier.weaknum):
        for j in range(num):
            # 计算强分类器预测评分 实质是内部弱分类器评分的加权和
            JudgFeature[j] += StrongClassifier.weakweight[i] * JudgOut[StrongClassifier.index[i], j]
    P = 0 # 检测个数
    FP = 0 # 误检个数
    for i in range(num):
        # 预测为人脸实际也为人脸的情况
        if i < num1 and JudgFeature[i] >= StrongClassifier.threshold:
            P += 1
        # 预测为人脸但实际是非人脸的情况
        elif i >= num1 and JudgFeature[i] >= StrongClassifier.threshold:
            FP += 1
    currentPR = P / num1 # 检测率
    currentFPR = FP / (num - num1) # 误检率
    return currentPR, currentFPR
