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
    获取强分类器检测结果
    '''
    JudgFeature = np.zeros(num)
    for i in range(StrongClassifier.weaknum):
        for j in range(num):
            JudgFeature[j] += StrongClassifier.weakweight[i] * JudgOut[StrongClassifier.index[i], j]
    P = 0
    FP = 0
    for i in range(num):
        if i < num1 and JudgFeature[i] >= StrongClassifier.threshold:
            P += 1
        elif i >= num1 and JudgFeature[i] >= StrongClassifier.threshold:
            FP += 1
    currentPR = P / num1
    currentFPR = FP / (num - num1)
    return currentPR, currentFPR
