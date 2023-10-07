import copy
import numpy as np
import cv2
import torch

from networks.car_recognition.car_rec import get_color_and_score
from networks.plate_recognition.double_plate_split_merge import get_split_merge
from networks.plate_recognition.plate_rec import get_plate_result
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from utils.train.datasets import letterbox
from utils.transform.region_transform import four_point_transform, scale_coords_landmarks # EM reconstructed
from utils.io.cv_img import cv_imaddtext
from utils.test.plate_format import rename_special_plate, check_plate_format # EM added

class ObjectInfo:
    """
    识别得到的对象信息. 
    """

    def __init__(self, class_label: int = None,
                 rect: list[int] = None, conf: float = None,
                 color: str = None, color_conf: float = None,
                 landmarks: list[int] = None, plate: str = None,
                 roi_height: int = None):
        self.type = class_label # 对象类型: 0-单层车牌，1-双层车牌, 2-车辆
        self.roi = rect # ROI 区域坐标
        self.conf = conf # 检测得分
        self.color = color # 对象颜色
        self.color_conf = color_conf # 颜色识别得分

        self.landmarks = landmarks # 车牌角点坐标列表
        self.plate = plate # 车牌号

        self.roi_height = roi_height # 透视标准化后的车牌图像高度

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

        img = letterbox(img0, new_shape=imgsz)[0] # 检测前处理, 图片长宽变为 32 倍数, 比如变为 640*640
        
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
                                                landmarks, class_num, rec_model)
                    obj_list.append(obj)

        return obj_list
    
    def visualize_result(self, orgimg: np.ndarray, obj_list: list[ObjectInfo]):
        """
        接收识别结果列表, 输出识别结果信息, 并按要求绘制车牌结果. 

        ## Parameters:
        `orgimg`: 原始图像.

        `obj_list`: 识别到的所有对象及其信息. 

        ## Return:
        `orgimg`: 绘制结果后的图像.

        `primary_plate`: 分数最高的车牌号.
        """

        all_plate_info = list[str]() # 本图像中所有车牌识别信息.
        primary_plate = "" # 对输出图片重命名的字符串. 

        for obj in obj_list:
            if obj.type == 2: # car
                height_area = self._restruct_plate_info(orgimg, obj.type, obj.roi)

                car_color_str = obj.color if self.show_color else ''

                if self.draw_rec:
                    orgimg = cv_imaddtext(orgimg, car_color_str,
                                          obj.roi[0], obj.roi[1],
                                          (0, 255, 0), height_area)
            else: # plate
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
            
            if self.draw_rec: # 绘制对象的 ROI 框
                cv2.rectangle(orgimg,
                            (rect_area[0], rect_area[1]),
                            (rect_area[2], rect_area[3]),
                            self.roi_colors[obj.type], 2)
        
        if len(all_plate_info) == 0:
            print('\t\033[31m未识别到有效对象. \033[0m')
        
        print('\t', end='')
        for item in all_plate_info:
            print(item + ' ', end='')
        print('\033[K')

        return orgimg, primary_plate
    
    def _get_rec_landmark(self, img: np.ndarray, xyxy: list[int], conf: float,
                          landmarks: list[int], class_num: int, rec_model):
        """
        获取识别对象的详细信息.

        ## Parameters:
        `img`: 原始图像.

        `xyxy`: 目标区域 ROI 坐标.

        `conf`: 检测得分. 

        `landmarks`: 

        `class_num`: 对象类型: 0-单层车牌, 1-双层车牌, 2-车辆.

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
        class_label = int(class_num) # 车牌类型: 0-单层车牌, 1-双层车牌, 2-车辆

        if class_label == 2: # Car recognition.
            # Now rec_model is car_rec_model.
            car_roi_img = img[y1:y2, x1:x2]
            car_color, color_conf = get_color_and_score(rec_model, car_roi_img, self.device)
            
            result = ObjectInfo(class_label, rect, conf, car_color, color_conf)

            # result_dict['rect'] = rect # 车辆 ROI 区域
            # result_dict['score'] = conf # 车辆区域检测得分
            # result_dict['object_no'] = class_label
            # result_dict['car_color'] = ''
            # if self.show_color:
            #     result_dict['car_color'] = car_color
            #     result_dict['color_conf'] = color_conf

            return result

        # Plate recognition.
        # Now rec_model is plate_rec_model.
        for i in range(4):
            point_x = int(landmarks[2 * i])
            point_y = int(landmarks[2 * i + 1])
            landmarks_np[i] = np.array([point_x, point_y])

        roi_img = four_point_transform(img, landmarks_np) # 车牌图像标准化为 ROI 图.

        if class_label == 1: # 若为双层车牌, 进行分割后拼接
            roi_img = get_split_merge(roi_img)

        plate_number, plate_color = get_plate_result(roi_img, self.device, rec_model) # 对 ROI 图进行识别.

        # 特殊车牌纠正
        if check_plate_format(plate_number):
            plate_number = rename_special_plate(plate_number)
        
        result = ObjectInfo(class_label, rect, conf, plate_color,
                            landmarks=landmarks_np.tolist(), plate=plate_number, roi_height=roi_img.shape[0])

        # result_dict['rect'] = rect # 车牌 ROI 区域
        # result_dict['detect_conf'] = conf # 检测区域得分
        # result_dict['landmarks'] = landmarks_np.tolist() # 车牌角点坐标
        # result_dict['plate_no'] = plate_number # 车牌号
        # result_dict['roi_height'] = roi_img.shape[0] # 车牌高度
        # result_dict['plate_color'] = ""
        # if self.show_color:
        #     result_dict['plate_color'] = plate_color # 车牌颜色
        #     # result_dict['color_conf'] = color_conf # 颜色得分
        # result_dict['object_no'] = class_label # 对象类型: 0-单层车牌，1-双层车牌
        
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