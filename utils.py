from Box import Box
import cv2 as cv


def cal_s(box):
    h = box.position[3] - box.position[1]
    w = box.position[2] - box.position[0]
    s = h * w
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


if __name__ == '__main__':
    box1 = Box([0, 0, 200, 200], 0.6)
    box2 = Box([100, 100, 300, 300], 0.9)
    box3 = Box([200, 200, 400, 600], 0.8)
    box4 = Box([500, 0, 600, 300], 0.4)
    final_box = NMS([box1, box2, box3, box4], 0.01)
    img = cv.imread('./test.png')
    img = drawbox(img, final_box)
    cv.imshow('test', img)
    cv.waitKey(0)

