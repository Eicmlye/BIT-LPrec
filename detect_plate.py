# -*- coding: UTF-8 -*-

"""
直接从原始图像中检测车牌并识别车牌号码等信息. 
"""

import argparse
import time
import os
import cv2
import torch
import copy
import numpy as np

from utils.transform.region_transform import four_point_transform, scale_coords_landmarks # EM reconstructed
from utils.io.cv_img import cv_imread, cv_imwrite, cv2ImgAddText # EM reconstructed
from utils.io.cmd_cursor import activate_cmd_cursor_opr # EM added
from utils.io.modify_filename import get_extension_index, control_filename_len # EM added
from utils.test.parser_arg import show_args # EM added
from utils.test.load import load_models, choose_device # EM added
from utils.test.plate_format import rename_special_plate, check_plate_format # EM added
from utils.test.parking_detect import update_parking_info, save_key_frame # EM added
from utils.test.video_eta import video_processing_prompt # EM added
from utils.train.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from networks.plate_recognition.plate_rec import get_plate_result, allFilePath
from networks.plate_recognition.double_plate_split_merge import get_split_merge
from networks.car_recognition.car_rec import get_color_and_score

clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)] # 车牌角点标识颜色
object_color = [(0, 255, 255), (0, 255, 0), (255, 255, 0)] # ROI 区域标识颜色
class_type = ['单层车牌', '双层车牌', '汽车']

def get_rec_landmark(img, xyxy, conf, landmarks, class_num, 
                     device, rec_model, is_color=False):
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
        car_color, color_conf = get_color_and_score(rec_model, car_roi_img, device)

        result_dict['class_type'] = class_type[class_label]
        result_dict['rect'] = rect # 车辆 ROI 区域
        result_dict['score'] = conf # 车辆区域检测得分
        result_dict['object_no'] = class_label
        result_dict['car_color'] = ''
        if is_color:
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

    if is_color:
        plate_number, plate_color = get_plate_result(roi_img, device, rec_model, is_color) # 对 ROI 图进行识别.
    else:
        plate_number = get_plate_result(roi_img, device, rec_model, is_color) 
    
    result_dict['class_type'] = class_type[class_label]
    result_dict['rect'] = rect # 车牌 ROI 区域
    result_dict['detect_conf'] = conf # 检测区域得分
    result_dict['landmarks'] = landmarks_np.tolist() # 车牌角点坐标
    result_dict['plate_no'] = plate_number # 车牌号
    result_dict['roi_height'] = roi_img.shape[0] # 车牌高度
    result_dict['plate_color'] = ""
    if is_color:
        result_dict['plate_color'] = plate_color # 车牌颜色
        # result_dict['color_conf'] = color_conf # 颜色得分
    result_dict['object_no'] = class_label # 对象类型: 0-单层车牌，1-双层车牌
    
    return result_dict

def detect_recognition_plate(models, orgimg, device, img_size, is_color=False):
    """
    识别车辆、车牌并获取对象信息.

    ## Parameters:

    `models`: 加载好的模型列表, 按顺序依次为:

        `car_rec_model`: 加载好的车辆识别模型.

        `detect_model`: 加载好的车牌检测模型. 

        `plate_rec_model`: 加载好的车牌识别模型.
    """

    # Load model
    conf_thres = 0.3 # 得分阈值
    iou_thres = 0.5 # nms 的 iou 值   
    dict_list = []
    img0 = copy.deepcopy(orgimg)
    assert orgimg is not None, 'Image Not Found ' 
    h0, w0 = orgimg.shape[:2] # original hw
    r = img_size / max(h0, w0) # resize image to img_size
    if r != 1: # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=models[1].stride.max()) # check img_size  

    img = letterbox(img0, new_shape=imgsz)[0] # 检测前处理, 图片长宽变为 32 倍数, 比如变为 640*640
    
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy() # BGR 转为 RGB, 然后将图片的 Height, Width, Color 排列变为 CHW 排列

    # Run inference
    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = models[1](img)[0]

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
                rec_model = models[0] if int(class_num) == 2 else models[2]
                result_dict = get_rec_landmark(orgimg, xyxy, conf, 
                                               landmarks, class_num, device,
                                               rec_model, is_color)
                dict_list.append(result_dict)

    return dict_list

def restruct_plate_info(orgimg, object_no, rect_area):
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

def visualize_result(orgimg, dict_list, do_draw = True, is_color = True):
    """
    接收 `detect_recognition_plate()` 的处理结果, 输出识别结果信息, 并按要求绘制车牌结果. 

    ## Parameters:
    `orgimg`: 原始图像.

    `dict_list`: 识别到的所有对象及其信息. 其元素是 `get_rec_landmark()` 返回的 `result_dict`. 
        
        `result_dict`: 

            `class_type`: 
        
            `rect`: 车牌 ROI 坐标.

            `landmarks`: 车牌角点坐标列表.

            `plate_no`: 车牌号字符串.

            `roi_height`:

            `plate_color`: 车牌颜色.

            `object_no`: 对象类型: 0-单层车牌, 1-双层车牌, 2-车辆.
    
    `do_draw`: 是否在输出中绘制检测框及相关信息.
    """

    all_plate_info = "" # 本图像中所有车牌识别信息.
    rename_str = "" # 对输出图片重命名的字符串. 

    for result in dict_list:
        rect_area = result['rect']
        object_no = result['object_no']
        
        if object_no == 2: # car
            height_area = restruct_plate_info(orgimg, object_no, rect_area)

            car_color_str = result['car_color'] if is_color else ''

            if do_draw:
                orgimg = cv2ImgAddText(orgimg, car_color_str, 
                                       rect_area[0], rect_area[1],
                                       (0, 255, 0), height_area)
        else: # plate
            rect_area = restruct_plate_info(orgimg, object_no, rect_area)

            landmarks = result['landmarks']
            plate = result['plate_no']

            if check_plate_format(plate):
                plate = rename_special_plate(plate)
                all_plate_info += plate
            else:
                all_plate_info += plate + "\033[31m(格式错误)\033[0m"

            all_plate_info += " " + (result['plate_color'] if is_color else '') + ("双层" if object_no == 1 else '') + " "

            if rename_str == "": # EM added: problem requirement
                if '危' not in plate and '险' not in plate and '品' not in plate: # 危险品标志不在题目的考虑范围内
                    rename_str = plate

            if do_draw:
                for i in range(4): # 绘制车牌角点
                    cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)
                
                labelSize = cv2.getTextSize(plate, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # 获得字体的大小
                if rect_area[0] + labelSize[0][0] > orgimg.shape[1]: # 防止显示的文字越界
                    rect_area[0] = int(orgimg.shape[1] - labelSize[0][0])
                orgimg = cv2.rectangle(orgimg, # 画文字框
                                       (rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1]))), 
                                       (int(rect_area[0] + round(1.2 * labelSize[0][0])), rect_area[1] + labelSize[1]), 
                                       (255, 255, 255), cv2.FILLED)
                
                if len(result) >= 1:
                    orgimg = cv2ImgAddText(orgimg, plate, rect_area[0], 
                                           int(rect_area[1] - round(1.6 * labelSize[0][1])), 
                                           (0, 0, 0), 21)
        
        if do_draw: # 绘制对象的 ROI 框
            cv2.rectangle(orgimg,
                        (rect_area[0], rect_area[1]),
                        (rect_area[2], rect_area[3]),
                        object_color[object_no], 2)
    
    if all_plate_info == '':
        all_plate_info = '\033[31m未识别到有效对象. \033[0m'
    print('\t' + all_plate_info + '\033[K')

    return orgimg, rename_str

def process_single_image(count, img_path, device, models,
                         img_size, is_color, do_draw, save_path):
    """
    处理单张图片. 

    ## Parameters:
    `count`: 图片张数计数器.

    `img_path`: 待处理图片路径.

    `device`: 用 cpu 或 gpu 处理.

    `models`: 加载好的模型列表, 按顺序依次为:

        `car_rec_model`: 加载好的车辆识别模型.

        `detect_model`: 加载好的车牌检测模型. 

        `plate_rec_model`: 加载好的车牌识别模型.

    `img_size`:

    `is_color`: 是否识别颜色.

    `do_draw`: 是否绘制识别框.

    `save_path`: 处理结果保存路径.
    """

    prompt_img_path = control_filename_len(img_path, 20)
    print(str(count + 1) + '\t', prompt_img_path)

    img = cv_imread(img_path)

    if img is None:
        print('Cannot open image: %s. ' % (img_path))
        return
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    dict_list = detect_recognition_plate(models, img, device, img_size, is_color) # 识别车辆, 检测并识别车牌
    
    ori_img, result_str = visualize_result(img, dict_list, do_draw, is_color)
    
    img_name = '_' + result_str + '_' + os.path.basename(img_path) # EM modified: problem requirement
    save_img_path = os.path.join(save_path, img_name) # 图片保存的路径
    cv_imwrite(ori_img, save_img_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--car_rec_model', type=str, default='weights/car_rec_color.pth', help='车辆识别模型路径, model.pth')
    parser.add_argument('--detect_model', nargs = '+', type = str, default = 'weights/plate_detect.pt', help = '检测模型路径, model.pt')
    parser.add_argument('--plate_rec_model', type=str, default='weights/plate_rec_color.pth', help='车牌识别模型路径, model.pth')
    
    parser.add_argument('--use_gpu', type=bool, default=None, help="是否使用 GPU, 默认为自动选择")
    parser.add_argument('--img_size', type=int, default=384, help='需为 32 的倍数. 该参数会影响识别阈值, 提高该参数会使低分目标显示出来')

    parser.add_argument('--is_video', action='store_true', help='处理图片还是视频')
    parser.add_argument('--do_draw', action='store_true', help='是否绘制识别框')
    parser.add_argument('--is_color', action='store_true', help='是否识别颜色')
    
    parser.add_argument('--image_path', type=str, default='input/imgs/', help='待识别图片(目录)路径')
    parser.add_argument('--video_path', type=str, default='input/videos/short.mp4', help='待识别视频路径')
    
    parser.add_argument('--output', type=str, default=None, help='处理结果保存位置')

    opt = parser.parse_args()

    device, device_choice = choose_device(opt.use_gpu)
    show_args(opt, device_choice)

    activate_cmd_cursor_opr()

    save_path = opt.output
    if not os.path.exists(save_path): 
        os.mkdir(save_path)

    # models = [car_rec_model, detect_model, plate_rec_model]
    models = load_models(opt, device)
    
    if not opt.is_video: # Image detection and recognition
        count = 0 # 处理项目数计数器

        if os.path.isfile(opt.image_path): # Single image file input
            process_single_image(count, opt.image_path, device, models,
                                 opt.img_size, opt.is_color, opt.do_draw, save_path)
        else: # Directory input.
            time_begin = time.time()

            file_list = []
            allFilePath(opt.image_path, file_list) # 将该目录下的所有图片路径读取到 file_list

            for img_path in file_list: # 遍历图片文件
                # time_b = time.time() # 开始时间

                process_single_image(count, img_path, device, models,
                                     opt.img_size, opt.is_color, opt.do_draw, save_path)
                
                # time_e = time.time()
                # time_gap = time_e - time_b # 计算单张图片识别耗时
                count += 1

            print(f"处理总用时 {time.time() - time_begin:.2f} s. ")
    else: # Video input.
        video_name = opt.video_path
        capture = cv2.VideoCapture(video_name)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        totalfps = capture.get(cv2.CAP_PROP_FPS) # 帧率
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 宽高

        extension = get_extension_index(video_name)
        output_path = os.path.join(save_path, os.path.basename(video_name[:extension] + '_result.mp4'))

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
                img0 = copy.deepcopy(img)
                dict_list = detect_recognition_plate(models, img, device, opt.img_size, opt.is_color)

                stopped_pos_info, parking_lot_info = update_parking_info(frame_count, dict_list, 
                                                                     int(totalfps), stopped_pos_info, 
                                                                     parking_lot_info)

                ori_img, _ = visualize_result(img, dict_list, opt.do_draw)

                t2 = cv2.getTickCount()
                time_gap = t2 - t1
                process_time, last_flush = video_processing_prompt(time_gap, totalFrames, frame_count,
                                                                   process_time, last_flush)

                out.write(ori_img)

        else:
            print("视频加载失败. ")

        print(f"\r\033[1A\033[K\033[1A\033[K\033[1A\033[K共处理 {frame_count} 帧, 平均处理帧速率 {totalFrames / process_time:.2f} fps. ")

        # 打印订单图片, 保存到 save_path 目录下.
        print('开始打印订单...', end='\n\n')
        key_frame_count = 0
        for parking in parking_lot_info:
            for getin in parking[1]:
                if getin[0] > 0:
                    key_frame_count += 1
                    save_key_frame(key_frame_count, parking[0], getin[0], 0, getin[1], capture, save_path)
            for occupy in parking[2]:
                if occupy[0] > 0:
                    key_frame_count += 1
                    save_key_frame(key_frame_count, parking[0], occupy[0], 1, occupy[1], capture, save_path)
            for getout in parking[3]:
                if getout[0] > 0:
                    key_frame_count += 1
                    save_key_frame(key_frame_count, parking[0], getout[0], 2, getout[1], capture, save_path)
            for release in parking[4]:
                if release[0] > 0:
                    key_frame_count += 1
                    save_key_frame(key_frame_count, parking[0], release[0], 3, release[1], capture, save_path)

        capture.release()
        out.release()
        cv2.destroyAllWindows()

        print('\r\033[1A\033[K\033[1A订单打印完成. \033[K')