import cv2 as cv
import glob
import os
import numpy as np


def ReadImage(facepath, nonfacepath):
    '''
    读取人脸和非人脸数据
    '''
    face_name_list = glob.glob(facepath + '/*.bmp')
    nonface_name_list = glob.glob(nonfacepath + '/*bmp')
    num1 = len(face_name_list)
    num = num1 + len(nonface_name_list)
    data = np.zeros((num, 20, 20))
    for i, name in enumerate(face_name_list):
        img = cv.imread(name)[:, :, 0]
        data[i, :, :] = img
    for i, name in enumerate(nonface_name_list):
        img = cv.imread(name)[:, :, 0]
        data[i+num1, :, :] = img
    return data, num, num1


if __name__ == '__main__':
    data, num, num1 = ReadImage('./faces', './nonfaces')
    print(data.shape)
    print(num)
    print(num1)