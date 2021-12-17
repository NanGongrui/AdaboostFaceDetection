from Haar import *


class WeakClassifier():
    def __init__(self, x1=0, x2=0, y1=0, y2=0, s=1, t=1, theta=0, p=1, error=0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.s = s
        self.t = t
        self.theta = theta
        self.p = p
        self.error = error


def TrainOneWeakClassifier(WeakClassifier, num, num1, Integral):
    '''
    训练一个弱分类器
    '''
    feature_list = []
    for i in range(num):
        feature = CalHaarValue(Integral[i, :, :], WeakClassifier.x1, WeakClassifier.y1, WeakClassifier.x2, WeakClassifier.y2, WeakClassifier.s, WeakClassifier.t)
        feature_list.append(feature)
    tmp = sorted(enumerate(feature_list), key=lambda x:x[1])
    sortedfeatureindex = [x[0] for x in tmp]
    sortedfeature = [x[1] for x in tmp]
    t1 = num1 / num # 人脸样本权重和
    t0 = 1 - t1 # 非人脸样本权重和
    s1 = 0 # 此权重之前的人脸样本权重和
    s0 = 0 # 此权重之前的非人脸样本权重和
    error_list = []
    p_list = []
    for i in range(num):
        error = min((s1 + t0 - s0), (s0 + t1 - s1))
        error_list.append(error)
        if error == (s1 + t0 - s0):
            p = -1
            p_list.append(p)
        else:
            p = 1
            p_list.append(p)
        if sortedfeatureindex[i] < num1:
            s1 += 1/num
        else:
            s0 += 1/num        
    tmp = min(enumerate(error_list), key=lambda x:x[1])
    minerrorindex = [x[0] for x in tmp]
    WeakClassifier.theta = sortedfeature[minerrorindex]
    WeakClassifier.p = p_list[minerrorindex]
    return WeakClassifier

def TrainSTWeakClassifier(num, num1, Integral, h, w, s, t):
    '''
    训练特定st条件的弱分类器
    '''
    WeakClassifier_list = []
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
    训练所有弱分类器
    '''
    WeakClassifier_list = []
    WeakClassifier_list += TrainSTWeakClassifier(num, num1, Integral, h, w, 1, 2)
    WeakClassifier_list += TrainSTWeakClassifier(num, num1, Integral, h, w, 2, 1)
    WeakClassifier_list += TrainSTWeakClassifier(num, num1, Integral, h, w, 1, 3)
    WeakClassifier_list += TrainSTWeakClassifier(num, num1, Integral, h, w, 3, 1)
    WeakClassifier_list += TrainSTWeakClassifier(num, num1, Integral, h, w, 2, 2)
    return WeakClassifier_list
    
def TrainWeak(WeakClassifier, num, num1, Integral, weight):
    error = 0
    for i in range(num):
        feature = CalHaarValue(Integral[i, :, :], WeakClassifier.x1, WeakClassifier.y1, WeakClassifier.x2, WeakClassifier.y2, WeakClassifier.s, WeakClassifier.t)
        if i < num1:
            y = 1
        else:
            y = 0
        if WeakClassifier.p * feature < WeakClassifier.p * WeakClassifier.theta:
            predicty = 1
        else:
            predicty = 0
        error += weight[i] * abs(predicty - y)
    WeakClassifier.error = error
    return WeakClassifier

if __name__ == '__main__':
    pass