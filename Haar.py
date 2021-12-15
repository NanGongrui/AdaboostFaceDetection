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
    m = int(position[1] + ((position[3] - position[1] + 1) / 2 - 1))
    return 2 * integralimage[m, position[2]] - integralimage[position[3], position[2]]

def cal_feature2(integralimage, position):
    m = int(position[0] + ((position[2] - position[0] + 1) / 2 - 1))
    return 2 * integralimage[position[3], m] - integralimage[position[3], position[2]]

def cal_feature3(integralimage, position):
    m1 = int(position[1] + ((position[3] - position[1] + 1) / 3 - 1))
    m2 = int(position[1] + ((position[3] - position[1] + 1) * 2 / 3 - 1))
    return 2 * integralimage[m1, position[2]] - 2 * integralimage[m2, position[2]] + integralimage[position[3], position[2]]

def cal_feature4(integralimage, position):
    m1 = int(position[0] + ((position[2] - position[0] + 1) / 3 - 1))
    m2 = int(position[0] + ((position[2] - position[0] + 1) * 2 / 3 - 1))
    return 2 * integralimage[position[3], m1] - 2 * integralimage[position[3], m2] + integralimage[position[3], position[2]]

def cal_feature5(integralimage, position):
    my = int(position[1] + ((position[3] - position[1] + 1) / 2 - 1))
    mx = int(position[0] + ((position[2] - position[0] + 1) / 2 - 1))
    return 4 * integralimage[my, mx] - 2 * integralimage[my, position[2]] - 2 * integralimage[position[3], mx] + integralimage[position[3], position[2]]


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
    print(cal_feature(integralimage, [0, 0, 2, 2], 5))