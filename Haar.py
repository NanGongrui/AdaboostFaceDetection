import numpy as np
import cv2 as cv


def CalIntegral(image, num, h, w):
    '''
    求积分图
    '''
    image = image.astype(int)
    Integral = np.zeros_like(image)
    for i in range(num):
        Integral[i, 0, 0] = image[i, 0, 0] # 第一个元素
        for j in range(1, h): # 最左边一列的元素
            Integral[i, j, 0] = Integral[i, j-1, 0] + image[i, j, 0]
        for j in range(1, w): # 最上面一行的元素
            Integral[i, 0, j] = Integral[i, 0, j-1] + image[i, 0, j]
        for j in range(1, h): # 剩下的元素
            for k in range(1, w):
                Integral[i, j, k] = Integral[i, j-1, k] + Integral[i, j, k-1] + image[i, j, k] - Integral[i, j-1, k-1]
    return Integral

def CalInteValue(Integral, x1, y1, x2, y2):
    '''
    计算一个窗口的积分值
    '''
    if x1 == 0 and y1 == 0:
        InteValue = Integral[y2, x2]
    elif y1 == 0:
        InteValue = Integral[y2, x2] - Integral[y2, x2-1]
    elif x1 == 0:
        InteValue = Integral[y2, x2] - Integral[y2-1, x2]
    else:
        InteValue = Integral[y2, x2] + Integral[y2-1, x2-1] - Integral[y2-1, x2] - Integral[y2, x2-1]
    return InteValue

def CalHaarValue(Integral, x1, y1, x2, y2, s, t):
    '''
    计算不同Haar特征的特征值
    '''
    if s == 1 and t == 2: # 两矩形特征
        m = int(y1 + (y2 - y1) / 2)
        white = CalInteValue(Integral, x1, y1, x2, m)
        black = CalInteValue(Integral, x1, m+1, x2, y2)
        print(white, ' ', black)
        return white - black
    elif s == 2 and t == 1: # 两矩形特征
        m = int(x1 + (x2 - x1) / 2)
        white = CalInteValue(Integral, x1, y1, m, y2)
        black = CalInteValue(Integral, m+1, y1, x2, y2)
        return white - black
    elif s == 1 and t == 3: # 三矩形特征
        m1 = int(y1 + (y2 - y1 + 1) / 3 - 1)
        m2 = int(y1 + 2 * (y2 - y1 + 1) / 3 - 1)
        white1 = CalInteValue(Integral, x1, y1, x2, m1)
        black = CalInteValue(Integral, x1, m1+1, x2, m2)
        white2 = CalInteValue(Integral, x1, m2+1, x2, y2)
        return white1 + white2 - black
    elif s == 3 and t == 1: # 三矩形特征
        m1 = int(x1 + (x2 - x1 + 1) / 3 - 1)
        m2 = int(x1 + 2 * (x2 - x1 + 1) / 3 - 1)
        white1 = CalInteValue(Integral, x1, y1, m1, y2)
        black = CalInteValue(Integral, m1+1, y1, m2, y2)
        white2 = CalInteValue(Integral, m2+1, y1, x2, y2)
        return white1 + white2 - black
    else: # 四矩形特征
        my = int(y1 + (y2 - y1) / 2)
        mx = int(x1 + (x2 - x1) / 2)
        white1 = CalInteValue(Integral, x1, y1, mx, my)
        white2 = CalInteValue(Integral, mx+1, my+1, x2, y2)
        black1 = CalInteValue(Integral, mx+1, y1, x2, my)
        black2 = CalInteValue(Integral, x1, my+1, mx, y2)
        return white1 + white2 - black1 - black2

def CalRectNum(s, t, h, w):
    '''
    计算满足条件的矩形个数
    '''
    num = 0
    for i in range(h - t + 1):
        for j in range(w - s + 1):
            num += int((h - i) / t) * int((w - j) / s)
    return num

def CalRectSum(h, w):
    '''
    计算所有矩形(特征)个数
    '''
    return 2 * CalRectNum(1, 2, h, w) + 2 * CalRectNum(1, 3, h, w) + CalRectNum(2, 2, h, w)


if __name__ == '__main__':
    '''
    用于调试
    '''
    img = cv.imread('./test.png')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    size = img.shape
    img = img[np.newaxis, :, :]
    print(img.shape)
    Integral = CalIntegral(img, 1, size[0], size[1])
    print(Integral[0])
    print(CalHaarValue(Integral[0], 0, 0, 1, 1, 2, 2))
    print(CalRectNum(2, 2, 24, 24))
    print(CalRectSum(24, 24))
