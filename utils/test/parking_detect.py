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
from detect import ObjectInfo

class CandidateParkingPos:
    """
    备选车位项. 
    """

    def __init__(self, obj: ObjectInfo, first_frame: int, last_frame: int):
        self.car_info = obj # 车辆信息
        self.first = first_frame # 首次被判定为停止时的帧号
        self.last = last_frame # 最近一次被判定为停止时的帧号

class Action:
    """
    订单信息.
    """

    def __init__(self, frame: int, plate: str):
        self.frame = frame # 订单帧号
        self.plate = plate # 目标车牌号

class ParkingPos:
    """
    停车位及订单信息
    """

    def __init__(self, roi: list[int], getin: list[Action] = None, occupy: list[Action] = None,
                 getout: list[Action] = None, release: list[Action] = None):
        self.roi = roi # 车位的 ROI 坐标
        self.getin = list[Action]() if getin is None else getin # 驶入订单列表
        self.occupy = list[Action]() if occupy is None else occupy # 占用订单列表
        self.getout = list[Action]() if getout is None else getout # 驶出订单列表
        self.release = list[Action]() if release is None else release # 释放车位订单列表

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

def find_plate(car_roi, plate_list: list[ObjectInfo]):
    for plate in plate_list:
        plate_roi = plate.roi
        if car_roi[0] <= plate_roi[0]\
            and car_roi[1] <= plate_roi[1]\
            and car_roi[2] >= plate_roi[2]\
            and car_roi[3] >= plate_roi[3]:
            return plate.plate
    
    return ''

def update_parking_info(curFrame: int, obj_list: list[ObjectInfo], fps: int, candidates: list[CandidateParkingPos], parkings: list[ParkingPos]):
    """
    `candidates` 会保存前 1 帧中被判定为停止的车辆信息, 以及其 ROI 坐标首次、最后一次被判定为停止的帧号. 
    由于程序不具备对象追踪能力, 故判定是否为车位时必须逐个比对 ROI 的位移.

    `parkings` 会保存所有已通过判定的车位 ROI 坐标及该车位产生订单的帧号和对应车牌号.

    ## Parameters:
    `curFrame`: 当前帧号.

    `obj_list`: 

    `fps`: 视频文件的帧率, 取整. 
    """

    if curFrame == 1:
        for obj in obj_list:
            if obj.type == 2:
                candidates = [CandidateParkingPos(obj, 1, 1)]

        return candidates, parkings

    move_dist_thres = 5 # 任一 ROI 坐标摆动超过该像素值, 则认为有实际移动. 
    distinguish_dist_thres = 10 # 任一 ROI 坐标摆动超过该像素值, 则认为是不同的对象.
    noise_frame_thres = fps * 5 # 持续该帧数内没有车辆被判定为从该区域移动时, 视该区域为模型噪声.
    parking_time_thres = 5 # 持续停止该时长的位置, 视为停车位.
    parking_frame_thres = fps * parking_time_thres

    # 提取车辆和车牌信息
    car_list = list[ObjectInfo]()
    plate_list = list[ObjectInfo]()

    for obj in obj_list: # 遍历检测到的对象
        if obj.type == 2:
            car_list.append(obj) # 检测到的车辆信息
        else:
            plate_list.append(obj) # 检测到车牌信息

    # 先检查是否有车已经进入已知的车位. 
    for car in car_list:
        for parking in parkings:
            iou = get_iou(car.roi, parking.roi)

            if iou > 0.7:
                if parking.occupy[-1].frame > parking.getin[-1].frame: # already occupied by this car
                    car_list.remove(car)
                    break
                # got in but not settled down yet
                if curFrame - parking.getin[-1].frame > parking_frame_thres: # 视为占用
                        parking.occupy.append(Action(curFrame, find_plate(car.roi, plate_list)))
            # Now iou <= 0.7
            elif iou > 0.5:
                if parking.occupy[-1].frame < parking.getin[-1].frame: # 视为入库
                    parking.getin.append(Action(curFrame, find_plate(car.roi, plate_list)))
                else: # 视为出库
                    parking.getout.append(Action(curFrame, find_plate(car.roi, plate_list)))
                car_list.remove(car)
                break

    # 然后检查是否有空车位
    for parking in parkings:
        if len(parking.getout) == 0:
            continue
        if parking.getout[-1].frame > parking.getin[-1].frame\
                and parking.getout[-1].frame > parking.occupy[-1].frame\
                and curFrame - parking.getout[-1].frame > parking_frame_thres: # 视为空位
            parking.release.append(Action(curFrame, parking.getout[-1].plate))

    # 再检测可能的车位位置.
    # 对比当前帧与前一帧的车辆移动状态. 
    for car in car_list:
        isNew = True
        for pos in candidates: # 将该车辆 ROI 与前一帧中未移动的车辆 ROI 对比
            if not check_change(car.roi, pos.car_info.roi, move_dist_thres): # 无移动
                isNew = False
                pos.last = curFrame # 更新持续停止帧数
                pos.car_info.roi = avg_pos(car.roi, pos.car_info.roi) # 平滑位置信息
                break
            elif not check_change(car.roi, pos.car_info.roi, distinguish_dist_thres): # 是同一车辆移动
                isNew = False
                pos.first = curFrame # 重新计数
                pos.last = curFrame # 重新计数
                break

        if isNew: # 将新车辆所在位置加入待判定集中
            candidates.append(CandidateParkingPos(car, curFrame, curFrame))

    for pos in candidates:
        # 持续停止时间足够长的位置, 视为停车位. 
        # 停车判定优先于待判定集更新, 故删除待判定集中的该位置不会增加下次计算量.
        if pos.last - pos.first > parking_frame_thres:
            # 添加新车位的同时, 添加入库和占用的订单信息.
            parkings.append(ParkingPos(
                                pos.car_info.roi, 
                                [Action(curFrame - parking_frame_thres, find_plate(pos.car_info.roi, plate_list))], 
                                [Action(curFrame, find_plate(pos.car_info.roi, plate_list))], 
                                list[Action](), 
                                list[Action]()
                            ))
            candidates.remove(pos)
        # 太久没有更新的位置, 视为检测模型的噪点, 删除该位置. 
        elif curFrame - pos.last > noise_frame_thres:
            candidates.remove(pos)

    return candidates, parkings

def save_key_frame(count: int, roi: list[int], frame_no: int, status: int, plate: str, capture: cv2.VideoCapture, save_path: str):
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
    ret, img = capture.read() # 返回一个布尔值和一个视频帧. 若帧读取成功, 则返回 True.

    if ret:
        # 绘制车位位置
        cv2.rectangle(img, (int(roi[0]), int(roi[1])), (int(roi[2]), int(roi[3])), (0, 0, 255), 1)

        filename = '_' + plate + '_' + status_list[status] + '.jpg'
        output_path = os.path.join(save_path, filename)
        cv_imwrite(img, output_path)
        print(f"\r\033[1A{count}\t已打印订单 " + filename + "\033[K")

def save_action_info(parkings: list[ParkingPos], capture: cv2.VideoCapture, save_path: str):
    print('开始打印订单...', end='\n\n')

    key_frame_count = 0
    for parking in parkings:
        for getin in parking.getin:
            if getin.frame > 0:
                key_frame_count += 1
                save_key_frame(key_frame_count, parking.roi, getin.frame, 0, getin.plate, capture, save_path)
        for occupy in parking.occupy:
            if occupy.frame > 0:
                key_frame_count += 1
                save_key_frame(key_frame_count, parking.roi, occupy.frame, 1, occupy.plate, capture, save_path)
        for getout in parking.getout:
            if getout.frame > 0:
                key_frame_count += 1
                save_key_frame(key_frame_count, parking.roi, getout.frame, 2, getout.plate, capture, save_path)
        for release in parking.release:
            if release.frame > 0:
                key_frame_count += 1
                save_key_frame(key_frame_count, parking.roi, release.frame, 3, release.plate, capture, save_path)

    print('\r\033[1A\033[K\033[1A订单打印完成. \033[K')
