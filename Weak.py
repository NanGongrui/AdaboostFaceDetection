from Haar import *


class WeakClassifier():
    def __init__(self, x1=0, x2=0, y1=0, y2=0, s=1, t=1, theta=0, p=1, error=0):
        # 该弱分类器的位置信息
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        # 该弱分类器的st条件
        self.s = s
        self.t = t
        self.theta = theta # 该弱分类器评分阈值
        self.p = p # 该弱分类器的符号项
        self.error = error # 该分类器的错误率


'''
注意下文的初始化训练和后训练的区别
初始化训练指的是根据样本分布的先验知识确定弱分类器的评分阈值和符号项
后训练指的是根据样本的权重确定弱分类器的错误率
'''


def TrainOneWeakClassifier(WeakClassifier, num, num1, Integral):
    '''
    初始化训练一个特定位置特定st条件的弱分类器
    '''
    feature_list = []
    # 计算每个样本的特征值
    for i in range(num):
        feature = CalHaarValue(Integral[i, :, :], WeakClassifier.x1, WeakClassifier.y1, WeakClassifier.x2, WeakClassifier.y2, WeakClassifier.s, WeakClassifier.t)
        feature_list.append(feature)
    # 对这些样本的特征值进行排序(升序排列)
    tmp = sorted(enumerate(feature_list), key=lambda x:x[1])
    sortedfeatureindex = [x[0] for x in tmp]
    sortedfeature = [x[1] for x in tmp]
    t1 = num1 / num # 人脸样本权重和
    t0 = 1 - t1 # 非人脸样本权重和
    s1 = 0 # 此阈值之前(小于该阈值)的人脸样本权重和
    s0 = 0 # 此阈值之前(小于该阈值)的非人脸样本权重和
    error_list = []
    p_list = []
    for i in range(num):
        error = min((s1 + t0 - s0), (s0 + t1 - s1)) # 计算错误率的公式
        error_list.append(error)
        '''
        这里务必说一下个人的理解
        (s1 + t0 - s0) - (s0 + t1 - s1) = 2 * (s1 - s0) + (t0 - t1)
        一般会假设人脸样本和非人脸样本一样多 则t0 = t1
        上式变为 2 * (s1 - s0)
        若error取(s1 + t0 - s0) 即s1 < s0 说明此阈值之前(小于该阈值)的非人脸数量多于人脸 因此符号项为负
        反之若error取(s0 + t1 - s1) 即s1 > s0 说明此阈值之前(小于该阈值)的人脸数量多于非人脸 因此符号项为正 
        '''
        if error == (s1 + t0 - s0):
            p = -1
            p_list.append(p)
        else:
            p = 1
            p_list.append(p)
        # 更新此阈值之前的人脸权重和非人脸权重
        if sortedfeatureindex[i] < num1:
            s1 += 1/num
        else:
            s0 += 1/num
    # 从以每个样本的特征值为阈值所计算的错误率中找到一个最小的错误率 其对应的样本特征值即为该弱分类器的阈值theta 其对应的符号项即为该弱分类器的符号项p
    tmp = min(enumerate(error_list), key=lambda x:x[1])
    minerrorindex = tmp[0]
    WeakClassifier.theta = sortedfeature[minerrorindex]
    WeakClassifier.p = p_list[minerrorindex]
    return WeakClassifier

def TrainSTWeakClassifier(num, num1, Integral, h, w, s, t):
    '''
    初始化训练特定st条件的弱分类器
    '''
    WeakClassifier_list = []
    # 遍历保存每个位置对应的弱分类器
    for y1 in range(0, h - t + 1):
        for x1 in range(0, w - s + 1):
            for y2 in range(y1 + t - 1, h, t):
                for x2 in range(x1 + s - 1, w, s):
                    classifier = WeakClassifier(x1, y1, x2, y2, s, t)
                    classifier = TrainOneWeakClassifier(classifier, num, num1, Integral)
                    WeakClassifier_list.append(classifier)
    return WeakClassifier_list

def TrainAllWeakClassifier(num, num1, Integral, h, w):
    '''
    初始化训练所有弱分类器
    '''
    WeakClassifier_list = []
    # 遍历保存每个st条件对应的弱分类器
    WeakClassifier_list += TrainSTWeakClassifier(num, num1, Integral, h, w, 1, 2)
    WeakClassifier_list += TrainSTWeakClassifier(num, num1, Integral, h, w, 2, 1)
    WeakClassifier_list += TrainSTWeakClassifier(num, num1, Integral, h, w, 1, 3)
    WeakClassifier_list += TrainSTWeakClassifier(num, num1, Integral, h, w, 3, 1)
    WeakClassifier_list += TrainSTWeakClassifier(num, num1, Integral, h, w, 2, 2)
    return WeakClassifier_list
    
def TrainWeak(WeakClassifier, num, num1, Integral, weight):
    '''
    后训练一个特定位置特定st条件的弱分类器 注意和初始化训练区分
    '''
    error = 0
    for i in range(num):
        # 计算该样本的特征值
        feature = CalHaarValue(Integral[i, :, :], WeakClassifier.x1, WeakClassifier.y1, WeakClassifier.x2, WeakClassifier.y2, WeakClassifier.s, WeakClassifier.t)
        # 获取真实标签
        if i < num1:
            y = 1
        else:
            y = 0
        # 这里就利用了该弱分类器的符号项和阈值对样本进行分类 详细解释见TrainOneWeakClassifier函数
        if WeakClassifier.p * feature < WeakClassifier.p * WeakClassifier.theta:
            predicty = 1
        else:
            predicty = 0
        # 错误率加权求和
        error += weight[i] * abs(predicty - y)
    WeakClassifier.error = error
    return WeakClassifier
