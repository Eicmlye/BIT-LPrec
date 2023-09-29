"""
车位信息的判定与更新. 

该方法判定车位的基本要求是, 车辆检测模型能较稳定地检测到车辆存在, 
以保证 ROI 区域和 IoU 值尽可能连续变化. 
其剧烈波动会极大影响该方法的判定准确性.

此外, 若视野内经常堵车, 容易误判产生大量停车位. 此时应增加检测车位的阈值
或手动导入车位数据并关闭车位检测. 
"""

import numpy as np
import cv2
import os
from utils.io.cv_img import cv_imwrite # EM reconstructed

def check_change(pos_1, pos_2, change_dist_thres = 7): 
    assert len(pos_1) == 4
    assert len(pos_2) == 4

    for index in range(4):
        if np.abs(pos_1[index] - pos_2[index]) > change_dist_thres: 
            return True
        
    return False

def avg_pos(pos_1, pos_2):
    assert len(pos_1) == 4
    assert len(pos_2) == 4

    pos = [(coord_1 + coord_2) / 2 for coord_1, coord_2 in zip(pos_1, pos_2)]

    return pos

def get_iou(pos_1, pos_2):
    assert len(pos_1) == 4
    assert len(pos_2) == 4

    area_1 = (pos_1[2] - pos_1[0]) * (pos_1[3] - pos_1[1])
    area_2 = (pos_2[2] - pos_2[0]) * (pos_2[3] - pos_2[1])

    w = max(0, min(pos_1[2], pos_2[2]) - max(pos_1[0], pos_2[0]))
    h = max(0, min(pos_1[3], pos_2[3]) - max(pos_1[1], pos_2[1]))
    intersection = w * h

    iou = intersection / (area_1 + area_2 - intersection)

    return iou

def find_plate(car_roi, plate_list):
    for plate in plate_list:
        plate_roi = plate['rect']
        if car_roi[0] <= plate_roi[0]\
            and car_roi[1] <= plate_roi[1]\
            and car_roi[2] >= plate_roi[2]\
            and car_roi[3] >= plate_roi[3]:
            return plate['plate_no']
    
    return ''

def update_parking_info(curFrame: int, dict_list, fps: int, stopped_pos_info: list, parking_lot_info: list):
    """
    车位判定相关的变量结构:
    
    `stopped_pos_info` 会保存前 1 帧中被判定为停止的车辆信息, 以及其 ROI 坐标首次、最后一次被判定为停止的帧号. 
    其元素结构是: 
        `[car_info, frame_no_1, frame_no_2]`
    由于程序不具备对象追踪能力, 故判定是否为车位时必须逐个比对 ROI 的位移.

    `parking_lot_info` 会保存所有已通过判定的车位 ROI 坐标及该车位产生订单的帧号和对应车牌号.
    其元素结构是: 
        `ROI`, 
        `getin_frames: list[int, plate]`, 
        `occupy_frames: list[int, plate]`,
        `getout_frames: list[int, plate]`,
        `release_frames: list[int, plate]`

    ## Parameters:
    `curFrame`: 当前帧号.

    `dict_list`: 

    `fps`: 视频文件的帧率, 取整. 
    """

    if curFrame == 1:
        for result in dict_list:
            if result['object_no'] == 2:
                stopped_pos_info = [[result, 1, 1]]

        return stopped_pos_info, parking_lot_info

    move_dist_thres = 5 # 任一 ROI 坐标摆动超过该像素值, 则认为有实际移动. 
    distinguish_dist_thres = 10 # 任一 ROI 坐标摆动超过该像素值, 则认为是不同的对象.
    noise_frame_thres = fps * 5 # 持续该帧数内没有车辆被判定为从该区域移动时, 视该区域为模型噪声.
    parking_time_thres = 5 # 持续停止该时长的位置, 视为停车位.
    parking_frame_thres = fps * parking_time_thres

    # 提取车辆和车牌信息
    car_list = []
    plate_list = []
    for result in dict_list: # 遍历检测到的对象
        if result['object_no'] == 2:
            car_list.append(result) # 检测到的车辆信息
        else:
            plate_list.append(result) # 检测到车牌信息

    # 先检查是否有车已经进入已知的车位. 
    for car in car_list:
        for parking in parking_lot_info:
            iou = get_iou(car['rect'], parking[0])
            if iou > 0.7:
                if parking[2][-1][0] > parking[1][-1][0]: # already occupied by this car
                    car_list.remove(car)
                    break
                # got in but not settled down yet
                if curFrame - parking[1][-1][0] > parking_frame_thres: # 视为占用
                        parking[2].append([curFrame, find_plate(car['rect'], plate_list)])
            # Now iou <= 0.7
            elif iou > 0.5:
                if parking[2][-1][0] < parking[1][-1][0]: # 视为入库
                    parking[1].append([curFrame, find_plate(car['rect'], plate_list)])
                else: # 视为出库
                    parking[3].append([curFrame, find_plate(car['rect'], plate_list)])
                car_list.remove(car)
                break

    # 然后检查是否有空车位
    for parking in parking_lot_info:
        if len(parking[3]) == 0:
            continue
        if parking[3][-1][0]> parking[1][-1][0] and parking[3][-1][0] > parking[2][-1][0]\
                and curFrame - parking[3][-1][0] > parking_frame_thres: # 视为空位
            parking[4].append([curFrame, parking[3][-1][1]])

    # 再检测可能的车位位置.
    # 对比当前帧与前一帧的车辆移动状态. 
    for car in car_list:
        isNew = True
        for pos in stopped_pos_info: # 将该车辆 ROI 与前一帧中未移动的车辆 ROI 对比
            if not check_change(car['rect'], pos[0]['rect'], move_dist_thres): # 无移动
                isNew = False
                pos[2] = curFrame # 更新持续停止帧数
                pos[0]['rect'] = avg_pos(car['rect'], pos[0]['rect']) # 平滑位置信息
                break
            elif not check_change(car['rect'], pos[0]['rect'], distinguish_dist_thres): # 是同一车辆移动
                isNew = False
                pos[1] = curFrame # 重新计数
                pos[2] = curFrame # 重新计数
                break

        if isNew: # 将新车辆所在位置加入待判定集中
            stopped_pos_info.append([car, curFrame, curFrame])

    for pos in stopped_pos_info:
        # 持续停止时间足够长的位置, 视为停车位. 
        # 停车判定优先于待判定集更新, 故删除待判定集中的该位置不会增加下次计算量.
        if pos[2] - pos[1] > parking_frame_thres:
            # 添加新车位的同时, 添加入库和占用的订单信息.
            parking_lot_info.append([
                                        pos[0]['rect'], 
                                        [[curFrame - parking_frame_thres, find_plate(pos[0]['rect'], plate_list)]], 
                                        [[curFrame, find_plate(pos[0]['rect'], plate_list)]], 
                                        [], 
                                        []
                                    ])
            stopped_pos_info.remove(pos)
        # 太久没有更新的位置, 视为检测模型的噪点, 删除该位置. 
        elif curFrame - pos[2] > noise_frame_thres:
            stopped_pos_info.remove(pos)

    return stopped_pos_info, parking_lot_info

def save_key_frame(count: int, roi, frame_no: int, status: int, plate: str, capture: cv2.VideoCapture, save_path:str):
    """
    See https://blog.csdn.net/yuejisuo1948/article/details/80734908
    for getting a specific frame of a video.

    ## Parameters:
    `frame_no`: 订单帧帧号.

    `status`: 0-入库, 1-占用, 2-出库, 3-释放.

    `capture`: 

    `save_path`:
    """

    status_list = ['驶入', '占用', '驶出', '空车位']
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no) # 设置要获取的帧号
    ret, img = capture.read() # 返回一个布尔值和一个视频帧. 若帧读取成功, 则返回True.

    if ret:
        cv2.rectangle(img, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (0, 0, 255), 1)

        filename = '_' + plate + '_' + status_list[status] + '.jpg'
        output_path = os.path.join(save_path, filename)
        cv_imwrite(img, output_path)
        print(f"\r\033[1A{count}\t已打印订单 " + filename + "\033[K")