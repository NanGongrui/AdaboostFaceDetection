import cv2 as cv
import glob
import numpy as np


def ReadImage(facepath, nonfacepath):
    '''
    读取人脸和非人脸数据 前一部分是人脸数据 后一部分是非人脸数据 拼接在一起的
    '''
    face_name_list = glob.glob(facepath + '/*.bmp')
    nonface_name_list = glob.glob(nonfacepath + '/*bmp')
    num1 = len(face_name_list) # 人脸数据个数
    num = num1 + len(nonface_name_list) # 总数据个数
    data = np.zeros((num, 20, 20))
    for i, name in enumerate(face_name_list):
        img = cv.imread(name)[:, :, 0] # 默认读取是RGB 但是数据集本身就是灰度图 因此只需要取出任意一个通道数值即可
        data[i, :, :] = img
    # 前一部分是人脸数据 后一部分是非人脸数据 拼接在一起的
    for i, name in enumerate(nonface_name_list):
        img = cv.imread(name)[:, :, 0]
        data[i+num1, :, :] = img
    return data, num, num1
