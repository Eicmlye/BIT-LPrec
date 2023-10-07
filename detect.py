import os
import sys
import torch
import cv2
import copy
import numpy as np
import time

from argparse import Namespace

from utils.io.strmod import FilenameModifier

from models.experimental import attempt_load
from networks.car_recognition.car_rec import get_color_and_score, init_car_rec_model
from networks.plate_recognition.double_plate_split_merge import get_split_merge
from networks.plate_recognition.plate_rec import get_plate_result, init_plate_rec_model
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from utils.train.datasets import letterbox
from utils.transform.region_transform import four_point_transform, scale_coords_landmarks # EM reconstructed
from utils.io.cv_img import cv_imread, cv_imwrite, cv_imaddtext
from utils.io.strmod import control_filename_len # EM added
from utils.test.plate_format import rename_special_plate, check_plate_format # EM added
from utils.test.parking_detect import update_parking_info, save_action_info # EM added
from utils.test.video_eta import video_processing_prompt # EM added

class Detecter:
    """
    对象检测及识别. 
    """

    # static attributes
    plate_corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)] # 车牌角点标识颜色
    roi_colors = [(0, 255, 255), (0, 255, 0), (255, 255, 0)] # ROI 区域标识颜色
    object_type = ['单层车牌', '双层车牌', '汽车']

    image_extensions = ['.jpg', '.png', '.JPG', '.PNG']
    
    def __init__(self, opt: Namespace):
        self.use_gpu, self.device = self._choose_device(opt.try_gpu) # GPU 或 CPU
        self.models = self._load_models({
                                            'detect': opt.detect_model,
                                            'carrec': opt.car_rec_model,
                                            'platerec': opt.plate_rec_model
                                        }) # 加载好的模型字典

        self.precision = opt.img_size # 处理精度

        self.is_video = opt.is_video # 待处理的是否为视频
        # 文件名处理器
        self.name_modifier = FilenameModifier(opt.video_path if self.is_video else opt.image_path,
                                              self.is_video, self.image_extensions, opt.output)

        self.draw_rec = opt.do_draw # 是否绘制 ROI 区域选框
        self.show_color = opt.is_color # 是否输出识别到的颜色属性

        self._show_args(opt) # 打印程序设置

    def run(self):
        """
        检测、识别目标. 
        """
        
        if self.is_video:
            self._process_video()
        else:
            self._process_image()

    def _show_args(self, opt: Namespace):
        """
        输出命令行参数的详细信息. 
        """

        # beginning line
        print("========")

        # models
        print("检测模型: " + opt.detect_model)
        print("车辆识别模型: " + opt.car_rec_model)
        print("车牌识别模型: " + opt.plate_rec_model)

        # I/O
        print(("待识别视频: " if self.is_video else "待识别图片: ") + self.name_modifier.root)

        print("处理结果保存位置: " + self.name_modifier.output_root)

        # hyperparameters
        print("模型超参数设置: ")
        print("\t处理精度(需为 32 的倍数): " + str(self.precision))

        # extra settings
        print("其他设置: ")

        print("\t自动选择了可用的 GPU" if opt.try_gpu == None and self.use_gpu else '', end='')
        print("\t无可用GPU, 自动选择了可用的 CPU" if opt.try_gpu == None and not self.use_gpu else '', end='')
        print("\t以 GPU 模式处理" if opt.try_gpu == True and self.use_gpu else '', end='')
        print("\t无可用GPU, 强制以 CPU 模式处理" if opt.try_gpu == True and not self.use_gpu else '', end='')
        print("\t以 CPU 模式处理" if opt.try_gpu == False else '', end='')
        print('')

        print('\t' + ('' if self.draw_rec else '不') + "绘制识别框")
        print('\t' + ('' if self.show_color else '不') + "输出对象颜色信息")
            
        # ending line
        print("========")
        
    def _load_models(self, model_dict: dict[str, str]):
        """
        加载所有要用到的模型, 并计算、输出其参数总量.
        """

        detect_model = self._init_detect_model(model_dict['detect'])
        car_rec_model = init_car_rec_model(model_dict['carrec'], self.device)
        plate_rec_model = init_plate_rec_model(model_dict['platerec'], self.device)

        # 计算参数量
        total_det = sum(p.numel() for p in detect_model.parameters())
        total_car_rec = sum(p.numel() for p in car_rec_model.parameters())
        total_plate_rec = sum(p.numel() for p in plate_rec_model.parameters())
        print("========")
        print("检测模型参数量: %.2f 万. " % (total_det / 1e4))
        print("车辆识别模型参数量: %.2f 万. " % (total_car_rec / 1e4))
        print("车牌识别模型参数量: %.2f 万. " % (total_plate_rec / 1e4))
        print("========")

        return {'detect': detect_model, 'carrec': car_rec_model, 'platerec': plate_rec_model}
    
    def _choose_device(self, try_gpu: bool):
        """
        根据用户意愿和实际可用资源, 分配 GPU 或 CPU. 

        ## Parameters:

        `try_gpu`: 
            
            - `None`: 根据可用资源, 自动优先选择 GPU.
            - `True`: 有 GPU 就用 GPU, 没有 GPU 就询问用户是否强制结束程序. 
            - `False`: 无论是否有 GPU, 都强制用 CPU. 

        ## Return:

        `is_gpu`: `True`: GPU, `False`: CPU.

        `device`: 获得的资源, `torch.device` 类型. 
        """

        is_gpu = None
        if try_gpu == None:
            is_gpu = torch.cuda.is_available()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif torch.cuda.is_available():
            is_gpu = try_gpu
            device = torch.device("cuda" if try_gpu else "cpu")
        else:
            is_gpu = False
            
            print("无可用 GPU, 是否使用 CPU 处理？ [[y]/n]: ", end='')
            while True: 
                choice = input()
                if choice == '' or choice[0] in ['y', 'Y']:
                    device = torch.device("cpu")
                    break
                elif choice[0] in ['n', 'N']:
                    print("\033[1A无可用 GPU, 用户选择结束处理. \033[K")
                    sys.exit(1)
                else:
                    print("请选择\'y\'(默认)或\'n\': ", end='')

        return is_gpu, device

    def _init_detect_model(self, weights):
        """
        加载检测模型. 
        """

        model = attempt_load(weights, map_location=self.device)  # load FP32 model

        return model
    
    def _process_image(self):
        """
        处理目标图片或者目标目录中的所有图片. 
        """
        
        count = 0 # 处理项目数计数器
        time_begin = time.time()

        for img_path in self.name_modifier.input_list: # 遍历图片文件
            self._process_single_image(count, img_path)
            count += 1

        print(f"处理总用时 {time.time() - time_begin:.2f} s. ")

        return
    
    def _process_single_image(self, count: int, img_path: str):
        """
        处理单张图片. 

        ## Parameters:
        `count`: 图片张数计数器.

        `img_path`: 待处理图片路径.

        `save_path`: 处理结果保存目录路径.
        """

        prompt_img_path = control_filename_len(img_path, 20)
        print(str(count + 1) + '\t', prompt_img_path)

        img = cv_imread(img_path)

        if img is None:
            print('Cannot open image: %s. ' % (img_path))
            return
        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        dict_list = self._detect_recognition_plate(img) # 识别车辆, 检测并识别车牌
        
        ori_img, result_str = self._visualize_result(img, dict_list)

        self.name_modifier.standard_image_filename(count, result_str)
        cv_imwrite(ori_img, self.name_modifier.get_result(count))
    
    def _process_video(self):
        """
        处理视频. 
        """
        
        for fileindex in range(len(self.name_modifier.input_list)):
            video_name = self.name_modifier.input_list[fileindex]
            capture = cv2.VideoCapture(video_name)
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            totalfps = capture.get(cv2.CAP_PROP_FPS) # 帧率
            width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 宽高
            output_path = self.name_modifier.get_result(fileindex)

            out = cv2.VideoWriter(output_path, fourcc, totalfps, (width, height)) # 写入视频

            frame_count = 0
            process_time = 0

            # 车位判定相关的变量结构, 见 update_parking_info()
            stopped_pos_info = []
            parking_lot_info = []

            if capture.isOpened():
                # See https://blog.csdn.net/u011436429/article/details/80604590
                # and https://blog.csdn.net/qq_43797817/article/details/108096827
                # for OpenCV VideoCapture.get() arguments.
                totalFrames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) # 视频文件的帧数

                last_flush = time.time() # 上次刷新预测剩余时间的时刻

                while True:
                    t1 = cv2.getTickCount()
                    ret, img = capture.read()
                    if frame_count > 0 and ret: # 刷新式更新处理进度信息.
                        print('\r\033[3A', end='')
                    if not ret:
                        break
                    
                    frame_count += 1
                    print(f"({frame_count / totalFrames * 100:.2f}%) 第 {frame_count}/{totalFrames} 帧\033[K")
                    dict_list = self._detect_recognition_plate(img)

                    stopped_pos_info, parking_lot_info = update_parking_info(frame_count, dict_list, 
                                                                        int(totalfps), stopped_pos_info, 
                                                                        parking_lot_info)

                    ori_img, _ = self._visualize_result(img, dict_list)

                    t2 = cv2.getTickCount()
                    time_gap = t2 - t1
                    process_time, last_flush = video_processing_prompt(time_gap, totalFrames, frame_count,
                                                                    process_time, last_flush)

                    out.write(ori_img)

            else:
                print("视频加载失败. ")

            print(f"\r\033[1A\033[K\033[1A\033[K\033[1A\033[K共处理 {frame_count} 帧, "\
                f"平均处理帧速率 {totalFrames / process_time:.2f} fps. ")

            # 打印订单图片, 保存到 save_path 目录下.
            save_action_info(parking_lot_info, capture, self.name_modifier.output_root)

            capture.release()
            out.release()
            cv2.destroyAllWindows()

    def _detect_recognition_plate(self, orgimg: np.ndarray):
        """
        识别车辆、车牌并获取对象信息.

        ## Parameters:

        `orgimg`: 待识别的图片. 
        """

        # Load model
        conf_thres = 0.3 # 得分阈值
        iou_thres = 0.5 # nms 的 iou 值   
        dict_list = []
        img0 = copy.deepcopy(orgimg)
        assert orgimg is not None, 'Image Not Found ' 
        h0, w0 = orgimg.shape[:2] # original hw
        r = self.precision / max(h0, w0) # resize image to self.precision
        if r != 1: # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
            img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(self.precision, s=self.models['detect'].stride.max()) # check self.precision  

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
                    result_dict = self._get_rec_landmark(orgimg, xyxy, conf, 
                                                landmarks, class_num, rec_model)
                    dict_list.append(result_dict)

        return dict_list
    
    def _visualize_result(self, orgimg, dict_list):
        """
        接收 `_detect_recognition_plate()` 的处理结果, 输出识别结果信息, 并按要求绘制车牌结果. 

        ## Parameters:
        `orgimg`: 原始图像.

        `dict_list`: 识别到的所有对象及其信息. 其元素是 `_get_rec_landmark()` 返回的 `result_dict`. 
            
            `result_dict`: 

                `class_type`: 
            
                `rect`: 车牌 ROI 坐标.

                `landmarks`: 车牌角点坐标列表.

                `plate_no`: 车牌号字符串.

                `roi_height`:

                `plate_color`: 车牌颜色.

                `object_no`: 对象类型: 0-单层车牌, 1-双层车牌, 2-车辆.
        """

        all_plate_info = "" # 本图像中所有车牌识别信息.
        rename_str = "" # 对输出图片重命名的字符串. 

        for result in dict_list:
            rect_area = result['rect']
            object_no = result['object_no']
            
            if object_no == 2: # car
                height_area = self._restruct_plate_info(orgimg, object_no, rect_area)

                car_color_str = result['car_color'] if self.show_color else ''

                if self.draw_rec:
                    orgimg = cv_imaddtext(orgimg, car_color_str, 
                                        rect_area[0], rect_area[1],
                                        (0, 255, 0), height_area)
            else: # plate
                rect_area = self._restruct_plate_info(orgimg, object_no, rect_area)

                landmarks = result['landmarks']
                plate = result['plate_no']

                if check_plate_format(plate):
                    plate = rename_special_plate(plate)
                    all_plate_info += plate
                else:
                    all_plate_info += plate + "\033[31m(格式错误)\033[0m"

                all_plate_info += " " + (result['plate_color'] if self.show_color else '') + ("双层" if object_no == 1 else '') + " "

                if rename_str == "": # EM added: problem requirement
                    if '危' not in plate and '险' not in plate and '品' not in plate: # 危险品标志不在题目的考虑范围内
                        rename_str = plate

                if self.draw_rec:
                    for i in range(4): # 绘制车牌角点
                        cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, self.plate_corner_colors[i], -1)
                    
                    labelSize = cv2.getTextSize(plate, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # 获得字体的大小
                    if rect_area[0] + labelSize[0][0] > orgimg.shape[1]: # 防止显示的文字越界
                        rect_area[0] = int(orgimg.shape[1] - labelSize[0][0])
                    orgimg = cv2.rectangle(orgimg, # 画文字框
                                        (rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1]))), 
                                        (int(rect_area[0] + round(1.2 * labelSize[0][0])), rect_area[1] + labelSize[1]), 
                                        (255, 255, 255), cv2.FILLED)
                    
                    if len(result) >= 1:
                        orgimg = cv_imaddtext(orgimg, plate, rect_area[0], 
                                            int(rect_area[1] - round(1.6 * labelSize[0][1])), 
                                            (0, 0, 0), 21)
            
            if self.draw_rec: # 绘制对象的 ROI 框
                cv2.rectangle(orgimg,
                            (rect_area[0], rect_area[1]),
                            (rect_area[2], rect_area[3]),
                            self.roi_colors[object_no], 2)
        
        if all_plate_info == '':
            all_plate_info = '\033[31m未识别到有效对象. \033[0m'
        print('\t' + all_plate_info + '\033[K')

        return orgimg, rename_str
    
    def _get_rec_landmark(self, img, xyxy, conf, landmarks, class_num, rec_model):
        """
        1. 获取车辆 ROI 区域.
        2. 获取车牌 ROI 区域和车牌号. 

        ## Parameters:
        `img`: 原始图像.

        `xyxy`: 目标区域 ROI 坐标.

        `conf`: 检测得分. 

        `landmarks`: 

        `class_num`: 对象类型: 0-单层车牌, 1-双层车牌, 2-车辆.

        `device`: 使用 cpu 或 gpu.

        `rec_model`: 加载好的识别模型. 需与 `class_num` 对应, 当 `class_num == 2` 时检测车辆, 
        输入车辆检测模型, 否则输入车牌检测模型.

        `is_color`: 

        ## Return:
        1. 车辆返回值:
        `result_dict`: 

        2. 车牌返回值:
        `result_dict`: 

            `class_type`: 
        
            `rect`: 车牌 ROI 坐标.

            `landmarks`: 车牌角点坐标列表.

            `plate_no`: 车牌号字符串.

            `roi_height`:

            `plate_color`: 车牌颜色.

            `object_no`: 对象类型: 0-单层车牌, 1-双层车牌, 2-车辆.
        """

        result_dict = {}

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

            result_dict['class_type'] = self.object_type[class_label]
            result_dict['rect'] = rect # 车辆 ROI 区域
            result_dict['score'] = conf # 车辆区域检测得分
            result_dict['object_no'] = class_label
            result_dict['car_color'] = ''
            if self.show_color:
                result_dict['car_color'] = car_color
                # result_dict['color_conf'] = color_conf

            return result_dict

        # Plate recognition.
        # Now rec_model is plate_rec_model.
        for i in range(4):
            point_x = int(landmarks[2 * i])
            point_y = int(landmarks[2 * i + 1])
            landmarks_np[i] = np.array([point_x, point_y])

        roi_img = four_point_transform(img, landmarks_np) # 车牌图像标准化为 ROI 图.

        if class_label == 1: # 若为双层车牌, 进行分割后拼接
            roi_img = get_split_merge(roi_img)

        if self.show_color:
            plate_number, plate_color = get_plate_result(roi_img, self.device, rec_model, self.show_color) # 对 ROI 图进行识别.
        else:
            plate_number = get_plate_result(roi_img, self.device, rec_model, self.show_color) 
        
        result_dict['class_type'] = self.object_type[class_label]
        result_dict['rect'] = rect # 车牌 ROI 区域
        result_dict['detect_conf'] = conf # 检测区域得分
        result_dict['landmarks'] = landmarks_np.tolist() # 车牌角点坐标
        result_dict['plate_no'] = plate_number # 车牌号
        result_dict['roi_height'] = roi_img.shape[0] # 车牌高度
        result_dict['plate_color'] = ""
        if self.show_color:
            result_dict['plate_color'] = plate_color # 车牌颜色
            # result_dict['color_conf'] = color_conf # 颜色得分
        result_dict['object_no'] = class_label # 对象类型: 0-单层车牌，1-双层车牌
        
        return result_dict
    
    def _restruct_plate_info(self, orgimg, object_no, rect_area):
        """
        通过 `rect_area` 计算可视化所需的数据格式. 
        """
        
        area = None

        if object_no == 2: # car
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