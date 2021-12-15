import cv2 as cv
import numpy as np


def GetIntegralImage(image):
    size = image.shape
    IntegralImage = np.zeros_like(image)
    s = np.zeros_like(image)
    for i in range(0, size[0]):
        for j in range(0, size[1]):
            if i == 0:
                s[i, j] = image[i, j]
            else:
                s[i, j] = s[i - 1, j] + image[i, j]
            if j == 0 and i == 0:
                IntegralImage[i, j] = image[i, j]
            elif j == 0:
                IntegralImage[i, j] = IntegralImage[i - 1, j] + image[i, j]
            else:
                IntegralImage[i, j] = IntegralImage[i, j - 1] + s[i, j]
    return IntegralImage

'''
Haar特征
1. 1    2. 1 -1     3. 1     4. 1 -1 1     5. 1 -1
  -1                  -1                     -1  1
                       1
1 2 边缘特征
3 4 线性特征
5   方向特征
'''

def cal_feature1(integralimage, position):
    x1 = position[0] - 1
    y1 = position[1] - 1
    x2 = position[2]
    y2 = position[3]
    m = int(y1 + (y2 - y1) / 2)
    if x1 < 0 and y1 < 0:
        ii1 = 0
        ii2 = 0
        ii3 = 0
        ii4 = integralimage[m, x2]
        ii5 = 0
        ii6 = integralimage[y2, x2]
    elif x1 < 0:
        ii1 = 0
        ii2 = integralimage[y1, x2]
        ii3 = 0
        ii4 = integralimage[m, x2]
        ii5 = 0
        ii6 = integralimage[y2, x2]
    elif y1 < 0:
        ii1 = 0
        ii2 = 0
        ii3 = integralimage[m, x1]
        ii4 = integralimage[m, x2]
        ii5 = integralimage[y2, x1]
        ii6 = integralimage[y2, x2]
    else:
        ii1 = integralimage[y1, x1]
        ii2 = integralimage[y1, x2]
        ii3 = integralimage[m, x1]
        ii4 = integralimage[m, x2]
        ii5 = integralimage[y2, x1]
        ii6 = integralimage[y2, x2]
    return ii4-ii2-ii3+ii1-ii6+ii4+ii5-ii3

def cal_feature2(integralimage, position):
    x1 = position[0] - 1
    y1 = position[1] - 1
    x2 = position[2]
    y2 = position[3]
    m = int(x1 + (x2 - x1) / 2)
    if x1 < 0 and x2 < 0:
        ii1 = 0
        ii2 = 0
        ii3 = 0
        ii4 = 0
        ii5 = integralimage[y2, m]
        ii6 = integralimage[y2, x2]
    elif x1 < 0:
        ii1 = 0
        ii2 = integralimage[y1, m]
        ii3 = integralimage[y1, x2]
        ii4 = 0
        ii5 = integralimage[y2, m]
        ii6 = integralimage[y2, x2]
    elif y1 < 0:
        ii1 = 0
        ii2 = 0
        ii3 = 0
        ii4 = integralimage[y2, x1]
        ii5 = integralimage[y2, m]
        ii6 = integralimage[y2, x2]
    else:
        ii1 = integralimage[y1, x1]
        ii2 = integralimage[y1, m]
        ii3 = integralimage[y1, x2]
        ii4 = integralimage[y2, x1]
        ii5 = integralimage[y2, m]
        ii6 = integralimage[y2, x2]
    return ii5-ii2-ii4+ii1-ii6+ii3+ii5-ii2

def cal_feature3(integralimage, position):
    x1 = position[0] - 1
    y1 = position[1] - 1
    x2 = position[2]
    y2 = position[3]
    m1 = int(y1 + (y2 - y1) / 3)
    m2 = int(y1 + (y2 - y1) / 3 * 2)
    if x1 < 0 and y1 < 0:
        ii1 = 0
        ii2 = 0
        ii3 = 0
        ii4 = integralimage[m1, x2]
        ii5 = 0
        ii6 = integralimage[m2, x2]
        ii7 = 0
        ii8 = integralimage[y2, x2]
    elif x1 < 0:
        ii1 = 0
        ii2 = integralimage[y1, x2]
        ii3 = 0
        ii4 = integralimage[m1, x2]
        ii5 = 0
        ii6 = integralimage[m2, x2]
        ii7 = 0
        ii8 = integralimage[y2, x2]
    elif y1 < 0:
        ii1 = 0
        ii2 = 0
        ii3 = integralimage[m1, x1]
        ii4 = integralimage[m1, x2]
        ii5 = integralimage[m2, x1]
        ii6 = integralimage[m2, x2]
        ii7 = integralimage[y2, x1]
        ii8 = integralimage[y2, x2]
    else:
        ii1 = integralimage[y1, x1]
        ii2 = integralimage[y1, x2]
        ii3 = integralimage[m1, x1]
        ii4 = integralimage[m1, x2]
        ii5 = integralimage[m2, x1]
        ii6 = integralimage[m2, x2]
        ii7 = integralimage[y2, x1]
        ii8 = integralimage[y2, x2]
    return ii4-ii2-ii3+ii1-ii6+ii4+ii5-ii3+ii8-ii6-ii7+ii5

def cal_feature4(integralimage, position):
    x1 = position[0] - 1
    y1 = position[1] - 1
    x2 = position[2]
    y2 = position[3]
    m1 = int(x1 + (x2 - x1) / 3)
    m2 = int(x1 + (x2 - x1) / 3 * 2)
    if x1 < 0 and y1 < 0:
        ii1 = 0
        ii2 = 0
        ii3 = 0
        ii4 = 0
        ii5 = 0
        ii6 = integralimage[y2, m1]
        ii7 = integralimage[y2, m2]
        ii8 = integralimage[y2, x2]
    elif x1 < 0:
        ii1 = 0
        ii2 = integralimage[y1, m1]
        ii3 = integralimage[y1, m2]
        ii4 = integralimage[y1, x2]
        ii5 = 0
        ii6 = integralimage[y2, m1]
        ii7 = integralimage[y2, m2]
        ii8 = integralimage[y2, x2]
    elif y1 < 0:
        ii1 = 0
        ii2 = 0
        ii3 = 0
        ii4 = 0
        ii5 = integralimage[y2, x1]
        ii6 = integralimage[y2, m1]
        ii7 = integralimage[y2, m2]
        ii8 = integralimage[y2, x2]
    else:
        ii1 = integralimage[y1, x1]
        ii2 = integralimage[y1, m1]
        ii3 = integralimage[y1, m2]
        ii4 = integralimage[y1, x2]
        ii5 = integralimage[y2, x1]
        ii6 = integralimage[y2, m1]
        ii7 = integralimage[y2, m2]
        ii8 = integralimage[y2, x2]
    return ii6-ii2-ii5+ii1-ii7+ii3+ii6-ii2+ii8-ii4-ii7+ii3

def cal_feature5(integralimage, position):
    x1 = position[0] - 1
    y1 = position[1] - 1
    x2 = position[2]
    y2 = position[3]
    mx = int(x1 + (x2 - x1) / 2)
    my = int(y1 + (y2 - y1) / 2)
    if x1 < 0 and y1 < 0:
        ii1 = 0
        ii2 = 0
        ii3 = 0
        ii4 = 0
        ii5 = integralimage[my, mx]
        ii6 = integralimage[my, x2]
        ii7 = 0
        ii8 = integralimage[y2, mx]
        ii9 = integralimage[y2, x2]
    elif x1 < 0:
        ii1 = 0
        ii2 = integralimage[y1, mx]
        ii3 = integralimage[y1, x2]
        ii4 = 0
        ii5 = integralimage[my, mx]
        ii6 = integralimage[my, x2]
        ii7 = 0
        ii8 = integralimage[y2, mx]
        ii9 = integralimage[y2, x2]
    elif y1 < 0:
        ii1 = 0
        ii2 = 0
        ii3 = 0
        ii4 = integralimage[my, x1]
        ii5 = integralimage[my, mx]
        ii6 = integralimage[my, x2]
        ii7 = integralimage[y2, x1]
        ii8 = integralimage[y2, mx]
        ii9 = integralimage[y2, x2]
    else:
        ii1 = integralimage[y1, x1]
        ii2 = integralimage[y1, mx]
        ii3 = integralimage[y1, x2]
        ii4 = integralimage[my, x1]
        ii5 = integralimage[my, mx]
        ii6 = integralimage[my, x2]
        ii7 = integralimage[y2, x1]
        ii8 = integralimage[y2, mx]
        ii9 = integralimage[y2, x2]
    return ii5-ii2-ii4+ii1-ii6+ii3+ii5-ii2-ii8+ii5+ii7-ii4+ii9-ii6-ii8+ii5

def cal_feature(integralimage, position, type):
    if type == 1:
        return cal_feature1(integralimage, position)
    elif type == 2:
        return cal_feature2(integralimage, position)
    elif type == 3:
        return cal_feature3(integralimage, position)
    elif type == 4:
        return cal_feature4(integralimage, position)
    elif type == 5:
        return cal_feature5(integralimage, position)

class HaarClssifier():
    pass

if __name__ == '__main__':
    img = cv.imread('./test.png')
    img1 = np.ones_like(cv.cvtColor(img, cv.COLOR_BGR2GRAY), dtype=np.int32)
    print(img.shape)
    integralimage = GetIntegralImage(img1)
    print(integralimage)
    print(cal_feature(integralimage, [1, 1, 3, 3], 4))