import copy
import numpy as np
import os
import cv2
import torch
from io import TextIOWrapper

from networks.car_recognition.car_rec import get_color_and_score
from utils.formatter.transform import get_split_merge
from networks.plate_recognition.plate_rec import get_plate_result
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from utils.train.datasets import letterbox
from utils.formatter.transform import four_point_transform, scale_coords_landmarks
from utils.formatter.cv_img import cv_imwrite, cv_imaddtext
from utils.formatter.plate_format import rename_special_plate, check_plate_format

class ObjectInfo:
    """
    识别到的对象信息. 
    """

    def __init__(self, class_label: int = None,
                 rect: list[int] = None, conf: float = None,
                 color: str = None, color_conf: float = None,
                 landmarks: list[int] = None, plate: str = None,
                 roi_height: int = None, car_plate_list: list[str] = list[str]()):
        self.type = class_label # 对象类型: 0-单层车牌，1-双层车牌, 2-车辆
        self.roi = rect # ROI 区域坐标
        self.conf = conf # 检测得分
        self.color = color # 对象颜色
        self.color_conf = color_conf # 颜色识别得分

        self.landmarks = landmarks # 车牌角点坐标列表
        self.plate = plate # 车牌号

        self.roi_height = roi_height # 透视标准化后的车牌图像高度

        self.car_plate_list = car_plate_list # 视频中该车识别到的车牌号及识别出该车牌的总帧数

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

    def __init__(self, id: int, roi: list[int], getin: list[Action] = None, occupy: list[Action] = None,
                 getout: list[Action] = None, release: list[Action] = None):
        self.id = id # 车位的唯一识别码
        self.roi = roi # 车位的 ROI 坐标
        self.getin = [Action(-1, "")] if getin is None else getin # 驶入订单列表
        self.occupy = [Action(-1, "")] if occupy is None else occupy # 占用订单列表
        self.getout = [Action(-1, "")] if getout is None else getout # 驶出订单列表
        self.release = [Action(-1, "")] if release is None else release # 释放车位订单列表

    def get_status(self):
        status = np.max([self.getin[-1].frame, self.occupy[-1].frame, self.getout[-1].frame, self.release[-1].frame])

        if status == self.getin[-1].frame:
            return 0
        elif status == self.occupy[-1].frame:
            return 1
        elif status == self.getout[-1].frame:
            return 2
        elif status == self.release[-1].frame:
            return 3
        
        return -1 # impossible return


class Detecter:
    """
    检测及识别图像和视频.
    """

    # static attributes
    landmark_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)] # 车牌角点标识颜色
    roi_colors = [(0, 255, 255), (0, 255, 0), (255, 255, 0)] # ROI 区域标识颜色

    def __init__(self, precision: float, models, device: torch.device, draw_rec: bool, show_color: bool):
        self.precision = precision
        self.models = models
        self.device = device
        self.draw_rec = draw_rec
        self.show_color = show_color

    def detect_recognition_plate(self, orgimg: np.ndarray):
        """
        识别车辆、车牌并获取对象信息.

        ## Parameters:

        `orgimg`: 待识别的图片. 
        """

        # Load model
        conf_thres = 0.3 # 得分阈值
        iou_thres = 0.5 # nms 的 iou 值   
        obj_list = list[ObjectInfo]()
        img0 = copy.deepcopy(orgimg)
        assert orgimg is not None, 'Image Not Found ' 
        h0, w0 = orgimg.shape[:2] # original hw
        r = self.precision / max(h0, w0) # resize image to `precision`
        if r != 1: # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(self.precision, s=self.models['detect'].stride.max()) # check precision  

        img = letterbox(img0, new_shape=imgsz)[0] # 检测前处理, 图片长宽变为 32 倍数, 比如变为 640 * 640
        
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1).copy() # BGR 转为 RGB, 然后将图片的 Height, Width, Color 排列变为 CHW 排列

        # Run inference
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = self.models['detect'](img)[0]

        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)

        # Process detections
        for i, det in enumerate(pred): # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], orgimg.shape).round()
                det[:, 5:13] = scale_coords_landmarks(img.shape[2:], det[:, 5:13], orgimg.shape).round()

                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = det[j, 5:13].view(-1).tolist()
                    class_num = det[j, 13].cpu().numpy()
                    rec_model = self.models['carrec'] if int(class_num) == 2 else self.models['platerec']
                    obj = self._get_rec_landmark(orgimg, xyxy, conf,
                                                 landmarks, int(class_num), rec_model)
                    obj_list.append(obj)

        return obj_list
    
    def visualize_result(self, orgimg: np.ndarray, obj_list: list[ObjectInfo], parkings: list[ParkingPos] = None):
        """
        接收识别结果列表, 输出识别结果信息, 并按要求绘制车牌结果. 

        ## Parameters:
        `orgimg`: 原始图像.

        `obj_list`: 识别到的所有对象及其信息. 

        `parkings`:

        ## Return:
        `orgimg`: 绘制结果后的图像.

        `primary_plate`: 分数最高的车牌号.
        """

        all_plate_info = list[str]() # 本图像中所有车牌识别信息.
        primary_plate = "" # 对输出图片重命名的字符串. 

        for obj in obj_list:
            if obj.type == 2: # car
                orgimg = self._visualize_car(orgimg, obj)
            else: # plate
                orgimg, primary_plate = self._visualize_plate(orgimg, obj, primary_plate, all_plate_info)
        
        orgimg = self._visualize_parkings(orgimg, parkings)
        
        if len(all_plate_info) == 0:
            print('\t\033[31m未识别到有效对象. \033[0m\033[K', end='')
        
        print('\t', end='')
        for item in all_plate_info:
            print(item, end=' \033[K')
        print('')

        return orgimg, primary_plate
    
    def _visualize_car(self, orgimg: np.ndarray, obj: ObjectInfo):
        assert obj.type == 2

        height_area = self._restruct_plate_info(orgimg, obj.type, obj.roi)
        rect_area = obj.roi

        car_color_str = obj.color if self.show_color else ''

        if self.draw_rec:
            orgimg = cv_imaddtext(orgimg, car_color_str,
                                    rect_area[0], rect_area[1],
                                    (0, 255, 0), height_area)
            
            cv2.rectangle(orgimg, # 绘制 ROI
                        (rect_area[0], rect_area[1]),
                        (rect_area[2], rect_area[3]),
                        self.roi_colors[obj.type], 2)
            
        return orgimg
    
    def _visualize_plate(self, orgimg: np.ndarray, obj: ObjectInfo, primary_plate: str, all_plate_info: list[str]):
        assert obj.type in {0, 1}

        rect_area = self._restruct_plate_info(orgimg, obj.type, obj.roi)

        landmarks = obj.landmarks
        plate = obj.plate

        all_plate_info.append(plate + ('' if check_plate_format(plate) else "\033[31m(格式错误)\033[0m"))
        all_plate_info[-1] += ' ' + (obj.color if self.show_color else '') + ("双层" if obj.type == 1 else '')

        if primary_plate == "":
            if '危' not in plate and '险' not in plate and '品' not in plate: # 危险品标志不在题目的考虑范围内
                primary_plate = plate

        if self.draw_rec:
            for i in range(4): # 绘制车牌角点
                cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, self.landmark_colors[i], -1)
            
            labelSize = cv2.getTextSize(plate, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # 获得字体的大小
            if rect_area[0] + labelSize[0][0] > orgimg.shape[1]: # 防止显示的文字越界
                rect_area[0] = int(orgimg.shape[1] - labelSize[0][0])
            orgimg = cv2.rectangle(orgimg, # 画文字框
                                (rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1]))), 
                                (int(rect_area[0] + round(1.2 * labelSize[0][0])), rect_area[1] + labelSize[1]), 
                                (255, 255, 255), cv2.FILLED)
            
            orgimg = cv_imaddtext(orgimg, plate, rect_area[0],
                                    int(rect_area[1] - round(1.6 * labelSize[0][1])),
                                    (0, 0, 0), 21)
            
            cv2.rectangle(orgimg, # 绘制 ROI
                        (rect_area[0], rect_area[1]),
                        (rect_area[2], rect_area[3]),
                        self.roi_colors[obj.type], 2)
            
        return orgimg, primary_plate
    
    def _visualize_parkings(self, orgimg: np.ndarray, parkings: list[ParkingPos] = None):
        if not parkings is None:
            for parking in parkings: 
                cv2.rectangle(orgimg, # 绘制车位 ROI
                              (int(parking.roi[0]), int(parking.roi[1])),
                              (int(parking.roi[2]), int(parking.roi[3])),
                              (0, 0, 255), 2)
                
                rect_area = parking.roi
                status_list = ['驶入', '占用', '驶出', '空车位']
                parking_text = "车位" + str(parking.id) + " " + status_list[parking.get_status()]
                    
                labelSize = cv2.getTextSize(parking_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # 获得字体的大小
                if rect_area[0] + labelSize[0][0] > orgimg.shape[1]: # 防止显示的文字越界
                    rect_area[0] = int(orgimg.shape[1] - labelSize[0][0])
                orgimg = cv2.rectangle(orgimg, # 画文字框
                                        (int(rect_area[0]), int(rect_area[1] - round(1.6 * labelSize[0][1]))),
                                        (int(rect_area[0] + round(1.2 * labelSize[0][0])), int(rect_area[1] + labelSize[1])), 
                                        (0, 0, 255), cv2.FILLED)
                
                orgimg = cv_imaddtext(orgimg, parking_text, rect_area[0],
                                      int(rect_area[1] - round(1.6 * labelSize[0][1])),
                                      (255, 255, 255), 21)
                
        return orgimg
    
    def _get_rec_landmark(self, img: np.ndarray, xyxy: list[int], conf: float,
                          landmarks: list[int], obj_type: int, rec_model):
        """
        获取识别对象的详细信息.

        ## Parameters:
        `img`: 原始图像.

        `xyxy`: 目标区域 ROI 坐标.

        `conf`: 检测得分. 

        `landmarks`: 

        `obj_type`: 对象类型: 0-单层车牌, 1-双层车牌, 2-车辆.

        `device`: 使用 cpu 或 gpu.

        `rec_model`: 加载好的识别模型. 需与 `class_num` 对应, 当 `class_num == 2` 时检测车辆, 
        输入车辆检测模型, 否则输入车牌检测模型.

        ## Return:
        `result`: `ObjectInfo`
        """

        result = None

        x1 = int(xyxy[0])
        y1 = int(xyxy[1])
        x2 = int(xyxy[2])
        y2 = int(xyxy[3])
        
        landmarks_np = np.zeros((4, 2))
        rect = [x1, y1, x2, y2]

        if obj_type == 2: # Car recognition.
            # Now rec_model is car_rec_model.
            car_roi_img = img[y1:y2, x1:x2]
            car_color, color_conf = get_color_and_score(rec_model, car_roi_img, self.device)
            
            result = ObjectInfo(obj_type, rect, conf, car_color, color_conf)

            return result

        # Plate recognition.
        # Now rec_model is plate_rec_model.
        for i in range(4):
            point_x = int(landmarks[2 * i])
            point_y = int(landmarks[2 * i + 1])
            landmarks_np[i] = np.array([point_x, point_y])

        roi_img = four_point_transform(img, landmarks_np) # 车牌图像标准化为 ROI 图.

        if obj_type == 1: # 若为双层车牌, 进行分割后拼接
            roi_img = get_split_merge(roi_img)

        plate_number, plate_color = get_plate_result(roi_img, self.device, rec_model) # 对 ROI 图进行识别.

        # 特殊车牌纠正
        if check_plate_format(plate_number):
            plate_number = rename_special_plate(plate_number)
        
        result = ObjectInfo(obj_type, rect, conf, plate_color,
                            landmarks=landmarks_np.tolist(),
                            plate=plate_number, roi_height=roi_img.shape[0])
        
        return result
    
    def _restruct_plate_info(self, orgimg: np.ndarray, obj_type: int, rect_area: list[int]):
        """
        通过 `rect_area` 计算可视化所需的数据格式. 
        """
        
        area = None

        if obj_type == 2: # car
            area = int((rect_area[3] - rect_area[1]) / 20)
        else: # plate
            x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
            padding_w = 0.05 * w
            padding_h = 0.11 * h

            area = []
            area.append(max(0, int(x - padding_w)))
            area.append(max(0, int(y - padding_h)))
            area.append(min(orgimg.shape[1], int(rect_area[2] + padding_w)))
            area.append(min(orgimg.shape[0], int(rect_area[3] + padding_h)))

        return area
    
    
"""
车位信息的判定与更新. 

该方法判定车位的基本要求是, 车辆检测模型能较稳定地检测到车辆存在, 
以保证 ROI 区域和 IoU 值尽可能连续变化. 
其剧烈波动会极大影响该方法的判定准确性.

此外, 若视野内经常堵车, 容易误判产生大量停车位. 此时应增加检测车位的阈值
或手动导入车位数据并关闭车位检测. 
"""

def check_change(pos_1: list[int], pos_2: list[int], change_dist_thres = 7): 
    assert len(pos_1) == 4
    assert len(pos_2) == 4

    center_1 = [(pos_1[0] + pos_1[2]) / 2, (pos_1[1] + pos_1[3]) / 2]
    center_2 = [(pos_2[0] + pos_2[2]) / 2, (pos_2[1] + pos_2[3]) / 2]

    for index in range(4):
        if np.abs(pos_1[index] - pos_2[index]) > change_dist_thres: 
            return True
        
    for index in range(2):
        if np.abs(center_1[index] - center_2[index]) > change_dist_thres:
            return True
        
    return False

def avg_pos(pos_1: list[int], pos_2: list[int]):
    assert len(pos_1) == 4
    assert len(pos_2) == 4

    pos = [(coord_1 + coord_2) / 2 for coord_1, coord_2 in zip(pos_1, pos_2)]

    return pos

def get_iou(pos_1: list[int], pos_2: list[int]):
    assert len(pos_1) == 4
    assert len(pos_2) == 4

    area_1 = (pos_1[2] - pos_1[0]) * (pos_1[3] - pos_1[1])
    area_2 = (pos_2[2] - pos_2[0]) * (pos_2[3] - pos_2[1])

    w = max(0, min(pos_1[2], pos_2[2]) - max(pos_1[0], pos_2[0]))
    h = max(0, min(pos_1[3], pos_2[3]) - max(pos_1[1], pos_2[1]))
    intersection = w * h

    iou = intersection / (area_1 + area_2 - intersection)

    return iou

def find_plate(car_roi: list[int], plate_list: list[ObjectInfo]):
    for plate in plate_list:
        if car_roi[0] <= plate.roi[0]\
            and car_roi[1] <= plate.roi[1]\
            and car_roi[2] >= plate.roi[2]\
            and car_roi[3] >= plate.roi[3]:
            return plate.plate
    
    return ''

def find_roi(this_plate: str, plate_list: list[ObjectInfo], car_list: list[ObjectInfo]):
    obj = None
    for plate in plate_list:
        if plate.plate == this_plate:
            obj = plate
            break

    if not obj is None:
        for car in car_list:
            if car.roi[0] <= obj.roi[0]\
                and car.roi[1] <= obj.roi[1]\
                and car.roi[2] >= obj.roi[2]\
                and car.roi[3] >= obj.roi[3]:
                return car.roi
    
    return None

def time2str(time_in_msec: int):
    assert time_in_msec >= 0

    day = time_in_msec // 3600 // 24 // 1000

    time_in_msec -= day * 3600 * 24 * 1000
    hr = time_in_msec // 3600 // 1000

    time_in_msec -= hr * 3600 * 1000
    min = time_in_msec // 60 // 1000
    
    time_in_msec -= min * 60 * 1000
    sec = time_in_msec // 1000

    msec = time_in_msec % 1000

    return f"{day}d{hr:02d}{min:02d}{sec:02d}{msec:03d}"

class ParkingLot:
    """
    车位信息管理器. 

    `self.candidates`: 保存前 1 帧中被判定为停止的车辆信息, 以及其 ROI 坐标首次、最后一次被判定为停止的帧号. 
    由于程序不具备对象追踪能力, 故判定是否为车位时必须逐个比对 ROI 的位移.

    `self.parkings`: 保存所有已通过判定的车位 ROI 坐标及该车位产生订单的帧号和对应车牌号.
    """

    # static thresholds
    move_dist_thres = 7 # 任一 ROI 坐标摆动超过该像素值, 则认为有实际移动. 
    distinguish_dist_thres = 15 # 任一 ROI 坐标摆动超过该像素值, 则认为是不同的对象.
    
    parking_time_thres = 5 # 持续停止该时长的位置, 视为停车位.
    noise_time_thres = 5 # 持续该时间内没有车辆被判定为从该区域移动时, 视该区域为模型噪声.
    
    def __init__(self, candidates: list[CandidateParkingPos] = None, parkings: list[ParkingPos] = None):
        self.candidates = list[CandidateParkingPos]() if candidates is None else candidates
        self.parkings = list[ParkingPos]() if parkings is None else parkings

    def _check_parkings(self, cur_frame: int, car_list: list[ObjectInfo], plate_list: list[ObjectInfo], parking_frame_thres: int):
        """
        检查当前车位状态.
        """

        # 先检查是否有车已经进入已知的车位. 
        for car in car_list:
            for parking in self.parkings:
                iou = get_iou(car.roi, parking.roi)

                if iou > 0.7:
                    if parking.get_status() == 1:
                        # 视为该车正占用该车位, 当然也可能这是另一辆车完全挡住了已停入库的车辆, 
                        # 但图像的深度信息难以提取, 故暂时直接视为占用同一车位.
                        # TODO: 相互有完全覆盖关系的车位, 可以考虑按照 iou 值选定其进入的车位. 
                        # 但如何确定这些车位确实是互不相同的新车位仍有难度.
                        car_list.remove(car)
                        break
                    # got in but not settled down yet
                    if parking.get_status() == 0 and cur_frame - parking.getin[-1].frame > parking_frame_thres:
                        # 停留时间超过阈值, 视为占用
                        parking.occupy.append(Action(cur_frame, find_plate(car.roi, plate_list)))
                        car_list.remove(car)
                        break

                # Now iou <= 0.7
                elif iou > 0.5:
                    this_plate = find_plate(car.roi, plate_list)
                    if parking.get_status() == 3: # 视为入库
                        parking.getin.append(Action(cur_frame, this_plate))
                        car_list.remove(car)
                        break
                    
                    if parking.get_status() == 1 and this_plate == parking.occupy[-1].plate: # 视为出库
                        parking.getout.append(Action(cur_frame, this_plate))
                        car_list.remove(car)
                        break

                    # 否则:
                    # 1. 车位处于占用状态且车牌不匹配, 则或者占用车位的车大部分被该车遮挡, 
                    # 或者该车为占用车位的车且其车牌被其他后来车遮挡.
                    # 2. 车位处于驶入或驶出状态, 此时应删除车辆信息.

        # 然后检查是否有空车位
        for parking in self.parkings:
            if parking.get_status() == 2 and cur_frame - parking.getout[-1].frame > parking_frame_thres:
                this_roi = find_roi(parking.getout[-1].plate, plate_list, car_list) # 尝试寻找驶出的车辆是否还能被识别到
                if this_roi is None or get_iou(parking.roi, this_roi) < 0.3: # 视为空位
                    parking.release.append(Action(cur_frame, parking.getout[-1].plate))

    def _update_candidates(self, cur_frame: int, car_list: list[ObjectInfo]):
        """
        更新潜在的车位信息.
        """
        # 检测可能的车位位置.
        # 对比当前帧与前一帧的车辆移动状态. 
        for car in car_list:
            isNew = True
            for pos in self.candidates: # 将该车辆 ROI 与前一帧中未移动的车辆 ROI 对比
                if not check_change(car.roi, pos.car_info.roi, self.move_dist_thres): # 无移动
                    isNew = False
                    pos.last = cur_frame # 更新持续停止帧数
                    pos.car_info.roi = avg_pos(car.roi, pos.car_info.roi) # 平滑位置信息
                    break
                elif not check_change(car.roi, pos.car_info.roi, self.distinguish_dist_thres): # 是同一车辆移动
                    isNew = False
                    pos.first = cur_frame # 重新计数
                    pos.last = cur_frame # 重新计数
                    break

            if isNew: # 将新车辆所在位置加入待判定集中
                self.candidates.append(CandidateParkingPos(car, cur_frame, cur_frame))

    def _update_parkings(self, cur_frame: int, plate_list: list[ObjectInfo], parking_frame_thres: int, noise_frame_thres: int):
        """
        将判定为新车位的位置加入车位集合中. 
        """

        for pos in self.candidates:
            if pos.last - pos.first > parking_frame_thres: # 持续停止时间足够长的位置, 视为停车位. 
                ignore_pos = False

                # 若与已知车位重合度过高, 则忽略该位置. 
                for parking in self.parkings:
                    if get_iou(parking.roi, pos.car_info.roi) >= 0.7:
                        ignore_pos = True
                        break

                # 添加新车位的同时, 添加入库和占用的订单信息.
                if not ignore_pos:
                    self.parkings.append(ParkingPos(
                                            len(self.parkings) + 1, # id
                                            pos.car_info.roi, # roi
                                            [Action(cur_frame - 2 * parking_frame_thres, find_plate(pos.car_info.roi, plate_list))], # getin
                                            [Action(cur_frame, find_plate(pos.car_info.roi, plate_list))] # occupy
                                        ))
                self.candidates.remove(pos)
            # 太久没有更新的位置, 视为检测模型的噪点, 删除该位置. 
            elif cur_frame - pos.last > noise_frame_thres:
                self.candidates.remove(pos)
        
    def update_parking_info(self, cur_frame: int, obj_list: list[ObjectInfo], fps: int):
        """
        ## Parameters:
        `cur_frame`: 当前帧号.

        `obj_list`: 

        `fps`: 视频文件的帧率, 取整. 
        """

        if cur_frame == 1:
            for obj in obj_list:
                if obj.type == 2:
                    self.candidates.append(CandidateParkingPos(obj, 1, 1))

            return
        
        noise_frame_thres = fps * self.noise_time_thres # 持续该帧数内没有车辆被判定为从该区域移动时, 视该区域为模型噪声.
        parking_frame_thres = fps * self.parking_time_thres

        # 提取车辆和车牌信息
        car_list = list[ObjectInfo]()
        plate_list = list[ObjectInfo]()

        for obj in obj_list: # 遍历检测到的对象
            if obj.type == 2:
                car_list.append(obj) # 检测到的车辆信息
            else:
                plate_list.append(obj) # 检测到车牌信息

        self._check_parkings(cur_frame, car_list, plate_list, parking_frame_thres)
        self._update_candidates(cur_frame, car_list)
        self._update_parkings(cur_frame, plate_list, parking_frame_thres, noise_frame_thres)

        return

    def _save_key_frame(self, count: int, parking: ParkingPos, action: Action,
                        status: int, capture: cv2.VideoCapture, save_path: str, f: TextIOWrapper):
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
        capture.set(cv2.CAP_PROP_POS_FRAMES, action.frame) # 设置要获取的帧号
        ret, img = capture.read() # 返回一个布尔值和一个视频帧. 若帧读取成功, 则返回 True.

        if ret:
            timestamp = int(capture.get(cv2.CAP_PROP_POS_MSEC)) # 毫秒为单位的时间戳
            timestamp = time2str(timestamp)

            # 绘制车位位置
            cv2.rectangle(img, (int(parking.roi[0]), int(parking.roi[1])), (int(parking.roi[2]), int(parking.roi[3])), (0, 0, 255), 2)

            filename = '车位' + str(parking.id) + '_' + timestamp + '_' + status_list[status] + '_' + action.plate + '.jpg'
            output_path = os.path.join(save_path, filename)
            cv_imwrite(img, output_path)
            print(f"\r\033[1A{count}\t已打印订单截图 " + filename + "\033[K")
            f.write(f"[{timestamp}] {action.plate} {status_list[status]} 车位{str(parking.id)}\n") # 打印日志

    def save_action_info(self, capture: cv2.VideoCapture, save_path: str, log_path: str):
        print('开始打印订单...', end='\n\n')

        key_frame_count = 0
        with open(log_path, "w+") as f:
            for parking in self.parkings:
                for getin in parking.getin:
                    if getin.frame > 0:
                        key_frame_count += 1
                        self._save_key_frame(key_frame_count, parking, getin, 0, capture, save_path, f)
                for occupy in parking.occupy:
                    if occupy.frame > 0:
                        key_frame_count += 1
                        self._save_key_frame(key_frame_count, parking, occupy, 1, capture, save_path, f)
                for getout in parking.getout:
                    if getout.frame > 0:
                        key_frame_count += 1
                        self._save_key_frame(key_frame_count, parking, getout, 2, capture, save_path, f)
                for release in parking.release:
                    if release.frame > 0:
                        key_frame_count += 1
                        self._save_key_frame(key_frame_count, parking, release, 3, capture, save_path, f)

        print(f'\r\033[1A\033[K\033[1A共计 {key_frame_count} 张订单截图打印完成, 订单日志打印完成. \033[K')
