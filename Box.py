import cv2 as cv


class Box():
    def __init__(self, position=[0, 0, 0, 0], rank=0):
        self.position = position
        self.rank = rank
    
    def update(self, position, rank):
        self.position = position
        self.rank = rank


def cal_s(box):
    h = box.position[3] - box.position[1]
    w = box.position[2] - box.position[0]
    s = h * w
    print(s)
    return s

def cal_iou(box1, box2):
    s1 = cal_s(box1)
    s2 = cal_s(box2)
    x1 = max(box1.position[0], box2.position[0])
    y1 = max(box1.position[1], box2.position[1])
    x2 = min(box1.position[2], box2.position[2])
    y2 = min(box1.position[3], box2.position[3])
    if (x2 <= x1) or (y2 <= y1):
        return 0
    box3 = Box([x1, y1, x2, y2])
    s3 = cal_s(box3)
    iou = s3 / (s1 + s2 - s3)
    return iou


def NMS(box_list, thereshold=0.1):
    box_list_len = len(box_list)
    box_list = sorted(box_list, key=lambda x: x.rank, reverse=True)
    final_box_list = []
    flag = [False] * box_list_len
    for i in range(box_list_len):
        if not flag[i]:
            final_box_list.append(box_list[i])
            for j in range(i + 1, box_list_len):
                if not flag[j]:
                    if(cal_iou(box_list[i], box_list[j]) > thereshold):
                        flag[j] = True
    return final_box_list

def drawbox(image, box_list):
    for i in range(len(box_list)):
        x1 = box_list[i].position[0]
        y1 = box_list[i].position[1]
        x2 = box_list[i].position[2]
        y2 = box_list[i].position[3]
        cv.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return image
