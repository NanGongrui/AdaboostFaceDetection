from ReadData import ReadImage
from Haar import *
from Weak import *
from Strong import *
from Box import *


def Train():
    # 1. 读取图片
    images, num, num1 = ReadImage('./faces', './nonfaces')
    H = images.shape[1]
    W = images.shape[2]
    # 2. 将图片转为积分图
    Integral = CalIntegral(images, num, H, W)
    # 3. 初始化训练 确定WeakClassifier结构体的theta和p
    WeakClassifiers, num_of_classifiers = TrainAllWeakClassifier(num, num1, Integral, H, W)
    # 4. 后训练
    weight = np.ones((1, num)) * 1 / num # 初始化样本权重
    T = 300 # 训练迭代次数(选择300个弱分类器)
    OptClassifiers = [] # 保存每轮的最优弱分类器
    ClassifiersWeight = [] # 每轮最优弱分类器权重
    for i in range(T):
        MinError = np.Inf # 最小误差
        MinErrorIndex = 0 # 最小误差对应的弱分类器编号
        # 寻找最优弱分类器
        for j in range(num_of_classifiers):
            WeakClassifiers[j] = TrainWeak(WeakClassifiers[j], num, num1, Integral, weight)
            if MinError >= WeakClassifiers[j].error:
                MinError = WeakClassifiers[j].error
                MinErrorIndex = j
        OptClassifiers.append(WeakClassifiers[MinErrorIndex])
        # 将该弱分类器从分类器列表中删除
        WeakClassifiers.pop(MinErrorIndex)
        num_of_classifiers -= 1
        # 计算该弱分类器的权重
        Beta = OptClassifiers[i].error / (1 - OptClassifiers[i].error)
        ClassifiersWeight.append(np.log(1 / Beta))
        # 更新样本权重
        for j in range(num):
            # 计算特征值
            feature = CalHaarValue(Integral[j, :, :], OptClassifiers[i].x1, OptClassifiers[i].y1, OptClassifiers[i].x2, OptClassifiers[i].y2, OptClassifiers[i].s, OptClassifiers[i].t)
            # 获取真实标签
            if j < num1:
                y = 1
            else:
                y = 0
            # 获取预测标签
            if OptClassifiers[i].p * feature < OptClassifiers[i].p * OptClassifiers[i].theta:
                predicty = 1
            else:
                predicty = 0
            # 样本预测正确就降低该样本的权重
            if predicty == y:
                weight[j] *= Beta
        # 样本权重归一化
        weight = weight / np.sum(weight)
    # 5. 求每个弱分类器对所有样本的预测标签
    JudgOut = np.zeros((T, num))
    for i in range(T):
        for j in range(num):
            feature = CalHaarValue(Integral[j, :, :], OptClassifiers[i].x1, OptClassifiers[i].y1, OptClassifiers[i].x2, OptClassifiers[i].y2, OptClassifiers[i].s, OptClassifiers[i].t)
            # 获取预测标签
            if OptClassifiers[i].p * feature < OptClassifiers[i].p * OptClassifiers[i].theta:
                JudgOut[i, j] = 1
            else:
                JudgOut[i, j] = 0
    # 6. 训练级联分类器
    MinPR = 0.925 # 最小检测率
    MaxFPR = 0.5 # 最大误检率
    WholeFPR = 0.001 # 整个分类器的误检率
    MinWeakNum = 3 # 强分类器中最小弱分类器个数
    TotalcurrentPR = 1 # 当前整个分类器的检测率
    TotalcurrentFPR = 1 # 当前整个分类器的误检率
    i = 0 # 当前强分类器的编号
    weaki = 0 # 当前弱分类器的编号
    lackflag = False # 弱分类器不够用标志
    StrongClassifiers = []
    while(TotalcurrentFPR > WholeFPR or i < 5):
        weaknum = 0
        currentPR = 0
        currentFPR = 1
        tmp = StrongClassifier()
        while(currentPR < MinPR or currentFPR > MaxFPR):
            # 不满足条件 需要给当前强分类器增加弱分类器
            if weaknum == 0: # 该强分类器还未初始化 其中没有弱分类器
                if (weaki + MinWeakNum) > T: # 弱分类器不够用
                    lackflag = True
                    break
                # 用三个弱分类器初始化该强分类器
                tmp.weak = OptClassifiers[weaki:weaki+MinWeakNum]
                tmp.weakweight = ClassifiersWeight[weaki:weaki+MinWeakNum]
                tmp.weaknum = MinWeakNum
                tmp.index = [x for x in range(weaki, weaki+MinWeakNum)]
                weaki += MinWeakNum
                weaknum = MinWeakNum
            else: # 该强分类器已经初始化 其中有弱分类器
                if (weaki + 1) > T: # 弱分类器不够用
                    lackflag = True
                    break
                # 添加一个弱分类器给该强分类器
                tmp.weak = tmp.weak.append(OptClassifiers[weaki])
                tmp.weakweight = tmp.weakweight.append(ClassifiersWeight[weaki])
                tmp.weaknum += 1
                tmp.index = tmp.index.append(weaki)
                weaki += 1
                weaknum += 1
            # 计算更新强分类器的各项性能
            tmp.threshold = 0.5 * np.sum(tmp.weakweight)
            currentPR, currentFPR = StrongJudg(tmp, JudgOut, num, num1)
        if lackflag == True:
            print('没有足够的弱分类器以供使用 程序即将退出')
            break
        tmp.PR = currentPR
        tmp.FPR = currentFPR
        TotalcurrentPR *= tmp.PR
        TotalcurrentFPR *= tmp.FPR
        StrongClassifiers.append(tmp)
    return StrongClassifiers
    
def Test(StrongClassifiers, image_path):
    image = cv.imread(image_path)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    h = image.shape[0]
    w = image.shape[1]
    image = image[np.newaxis, :, :]
    # 计算该测试图片的积分图
    Integral = CalIntegral(image, 1, h, w)
    offset = 5 # 预测窗口每次移动的步长
    times = min(h / 20, w / 20) # 预测窗口的最大放大倍数
    box_list = [] # 用于保存检测框结果
    for t in range(1, times + 1): # 遍历每个放大倍数对应的预测窗口
        for y1 in range(0, h - t * 20 + 1, offset): # 垂直方向的窗口滑动
            for x1 in range(0, w - t * 20 + 1, offset): # 水平方向的窗口滑动
                rank = 0 # 该窗口的评分
                for i in range(len(StrongClassifiers)): # 遍历级联强分类器
                    tmpout = 0 # 强分类器计算出的值 初始化为0
                    for j in range(StrongClassifiers[i].weaknum): # 遍历强分类器中的每个弱分类器
                        # 弱分类器根据当前的放大倍数来放大窗口
                        xx1 = x1 + (StrongClassifiers[i].weak[j].x1) * t
                        yy1 = y1 + (StrongClassifiers[i].weak[j].y1) * t
                        xx2 = xx1 + (StrongClassifiers[i].weak[j].x2 - StrongClassifiers[i].weak[j].x1) * t
                        yy2 = yy1 + (StrongClassifiers[i].weak[j].y2 - StrongClassifiers[i].weak[j].y1) * t
                        # 计算弱分类器特征值
                        feature = CalHaarValue(Integral, xx1, yy1, xx2, yy2, StrongClassifiers[i].weak[j].s, StrongClassifiers[i].weak[j].t)
                        # 弱分类器判断为人脸 窗口变大 特征值的阈值也要扩大
                        if StrongClassifiers[i].weak[j].p * feature < StrongClassifiers[i].weak[j].p * StrongClassifiers[i].weak[j].theta * t * t:
                            tmpout += StrongClassifiers[i].weakweight[j]
                    # 强分类器判断为非人脸
                    if tmpout < StrongClassifiers[i].threshold:
                        break
                    rank += tmpout # 累加级联分类器的最终评分作为该检测框的评分
                    if i == len(StrongClassifiers) - 1: # 所有强分类器都判断为人脸 那么该测试图片大概率是人脸 保存该检测框
                        box_list.append(Box([x1, y1, x1 + offset * t - 1, y1 + offset * t - 1], rank=rank))
    return box_list

def Show(box_list, imagepath):
    '''
    展示测试结果
    '''
    image = cv.imread(imagepath)
    box_list = NMS(box_list, 0.01) # 非极大值抑制去除冗余检测框
    image = drawbox(image, box_list) # 画出检测框
    cv.imshow('face detection', image) # 展示图片
    cv.waitKey(0)


if __name__ == '__main__':
    StrongClassifiers = Train() # 训练
    box_list = Test(StrongClassifiers, './faces_test/1.jpg') # 测试
    Show(box_list, './faces_test/1.jpg') # 展示测试结果
