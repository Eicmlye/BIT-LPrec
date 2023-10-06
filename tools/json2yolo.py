import json
import os
import numpy as np
from copy import deepcopy

from utils.io.cv_img import cv_imread # EM reconstructed
from networks.car_recognition.car_rec import all_file_path

def xywh2yolo(rect,landmarks_sort,img):
    h,w,c =img.shape
    rect[0] = max(0, rect[0])
    rect[1] = max(0, rect[1])
    rect[2] = min(w - 1, rect[2]-rect[0])
    rect[3] = min(h - 1, rect[3]-rect[1])
    annotation = np.zeros((1, 12))
    annotation[0, 0] = (rect[0] + rect[2] / 2) / w  # cx
    annotation[0, 1] = (rect[1] + rect[3] / 2) / h  # cy
    annotation[0, 2] = rect[2] / w  # w
    annotation[0, 3] = rect[3] / h  # h

    annotation[0, 4] = landmarks_sort[0][0] / w  # l0_x
    annotation[0, 5] = landmarks_sort[0][1] / h  # l0_y
    annotation[0, 6] = landmarks_sort[1][0] / w  # l1_x
    annotation[0, 7] = landmarks_sort[1][1] / h  # l1_y
    annotation[0, 8] = landmarks_sort[2][0] / w  # l2_x
    annotation[0, 9] = landmarks_sort[2][1] / h # l2_y
    annotation[0, 10] = landmarks_sort[3][0] / w  # l3_x
    annotation[0, 11] = landmarks_sort[3][1] / h  # l3_y
    # annotation[0, 12] = (landmarks_sort[0][0]+landmarks_sort[1][0])/2 / w  # l4_x
    # annotation[0, 13] = (landmarks_sort[0][1]+landmarks_sort[1][1])/2 / h  # l4_y
    return annotation            
            
if __name__ == "__main__":
    pic_file_list = []
    pic_file = r"./traindata/jsontmp/"
    save_small_path = "small"
    label_file = ['0','1']
    all_file_path(pic_file,pic_file_list)
    count=0
    index = 0
    for pic_ in pic_file_list:
        if not pic_.endswith(".jpg"):
            continue
        count+=1
        img = cv_imread(pic_) # EM modified
        img_name = os.path.basename(pic_)
        txt_name = img_name.replace(".jpg",".txt")
        txt_path = os.path.join(pic_file,txt_name)
        json_file_ = pic_.replace(".jpg",".json")
        if not os.path.exists(json_file_):
            continue
        with open(json_file_, 'r',encoding='utf-8') as a:
            data_dict = json.load(a)
            # print(data_dict['shapes'])
            with open(txt_path,"w") as f:
                for  data_message in data_dict['shapes']:
                    index+=1
                    label=data_message['label']
                    points = data_message['points']
                    pts = np.array(points)
                    # pts=order_points(pts)
                    # new_img = four_point_transform(img,pts)
                    roi_img_name = label+"_"+str(index)+".jpg"
                    save_path=os.path.join(save_small_path,roi_img_name)
                    # cv2.imwrite(save_path,new_img)
                    x_max,y_max = np.max(pts,axis=0)
                    x_min,y_min = np.min(pts,axis=0)
                    rect = [x_min,y_min,x_max,y_max]
                    rect1=deepcopy(rect)
                    annotation=xywh2yolo(rect1,pts,img)
                    print(data_message)
                    label = data_message['label']
                    str_label = label_file.index(label)
                    # str_label = "0 "
                    str_label = str(str_label)+" "
                    for i in range(len(annotation[0])):
                            str_label = str_label + " " + str(annotation[0][i])
                    str_label = str_label.replace('[', '').replace(']', '')
                    str_label = str_label.replace(',', '') + '\n'

                    f.write(str_label)
            print(count,img_name)
                # point=data_message[points]
