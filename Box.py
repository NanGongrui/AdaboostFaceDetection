import cv2 as cv


class Box():
    def __init__(self, position=[0, 0, 0, 0], rank=0):
        self.position = position # 确定检测框位置的四个坐标点
        self.rank = rank # 检测框评分
    
    def update(self, position, rank):
        self.position = position
        self.rank = rank


def cal_s(box):
    '''
    计算检测框面积
    '''
    h = box.position[3] - box.position[1]
    w = box.position[2] - box.position[0]
    s = h * w
    return s

def cal_iou(box1, box2):
    '''
    计算两个检测框之间的IOU
    '''
    s1 = cal_s(box1)
    s2 = cal_s(box2)
    # 计算相交矩形的位置信息
    x1 = max(box1.position[0], box2.position[0])
    y1 = max(box1.position[1], box2.position[1])
    x2 = min(box1.position[2], box2.position[2])
    y2 = min(box1.position[3], box2.position[3])
    if (x2 <= x1) or (y2 <= y1): # 如果不相交就返回0
        return 0
    box3 = Box([x1, y1, x2, y2])
    s3 = cal_s(box3)
    # 计算IOU
    iou = s3 / (s1 + s2 - s3)
    return iou


def NMS(box_list, thereshold=0.1):
    '''
    非极大值抑制算法 删去重叠的检测框
    '''
    box_list_len = len(box_list)
    # 先按照评分排序
    box_list = sorted(box_list, key=lambda x: x.rank, reverse=True)
    final_box_list = []
    flag = [False] * box_list_len # 标记一个检测框是否被删去
    for i in range(box_list_len):
        if not flag[i]:
            final_box_list.append(box_list[i])
            for j in range(i + 1, box_list_len):
                if not flag[j]:
                    if(cal_iou(box_list[i], box_list[j]) > thereshold): # IOU大于阈值删去评分低的检测框
                        flag[j] = True
    return final_box_list

def drawbox(image, box_list):
    '''
    绘制检测框
    '''
    for i in range(len(box_list)):
        x1 = box_list[i].position[0]
        y1 = box_list[i].position[1]
        x2 = box_list[i].position[2]
        y2 = box_list[i].position[3]
        cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image
