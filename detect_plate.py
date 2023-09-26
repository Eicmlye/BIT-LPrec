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

from models.experimental import attempt_load
from utils.transform.point_order_transform import four_point_transform # EM reconstructed
from utils.io.cv_img_io import cv_imread, cv_imwrite, cv2ImgAddText # EM reconstructed
from utils.train.datasets import letterbox
from utils.general import check_img_size, non_max_suppression_face, scale_coords
from networks.plate_recognition.plate_rec import get_plate_result, allFilePath, init_plate_rec_model
from networks.plate_recognition.double_plate_split_merge import get_split_merge
from networks.car_recognition.car_rec import get_color_and_score, init_car_rec_model

clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)] # 车牌角点标识颜色
object_color = [(0, 255, 255), (0, 255, 0), (255, 255, 0)]
class_type = ['单层车牌', '双层车牌', '汽车']

def load_model(weights, device): # load detect model
    model = attempt_load(weights, map_location = device)  # load FP32 model

    return model

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    【功能暂不确定】将 ROI 坐标还原为 ROI 区域的角点坐标列表. 
    """

    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7]] -= pad[1]  # y padding
    coords[:, :8] /= gain
    
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4

    return coords # coords is a torch.tensor here. 

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

def detect_recognition_plate(model, orgimg, device, car_rec_model, plate_rec_model, img_size, is_color=False):
    """
    识别车辆、车牌并获取对象信息.
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

    imgsz = check_img_size(img_size, s=model.stride.max()) # check img_size  

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
    pred = model(img)[0]

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
                rec_model = car_rec_model if int(class_num) == 2 else plate_rec_model
                result_dict = get_rec_landmark(orgimg, xyxy, conf, 
                                                     landmarks, class_num, device,
                                                     rec_model, is_color)
                dict_list.append(result_dict)

    return dict_list

def draw_result(orgimg, dict_list, do_draw = True):
    """
    输出识别结果信息, 并按要求绘制车牌结果. 

    ## Parameters:
    `orgimg`: 原始图像.

    `dict_list`: 识别到的所有对象及其信息. 其元素结构是 `get_rec_landmark()` 返回的 `result_dict`. 
        
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

    result_str = "" # 本图像中所有识别信息.
    rename_str = "" # 对输出图片重命名的字符串. 

    for result in dict_list:
        rect_area = result['rect']
        object_no = result['object_no']
        
        if object_no == 2: # car
            height_area = int((rect_area[3] - rect_area[1]) / 20)
            car_color_str = result['car_color']
            orgimg = cv2ImgAddText(orgimg, 
                                   car_color_str, 
                                   rect_area[0],
                                   rect_area[1],
                                   (0,255,0),
                                   height_area)
        else: # plate
            x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
            padding_w = 0.05 * w
            padding_h = 0.11 * h
            rect_area[0] = max(0, int(x - padding_w))
            rect_area[1] = max(0, int(y - padding_h))
            rect_area[2] = min(orgimg.shape[1], int(rect_area[2] + padding_w))
            rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))

            landmarks = result['landmarks']
            result_p = result['plate_no']

            if len(result_p) > 2 and result_p[1] == '0': # EM added: special 'O' character
                result_p = result_p[0] + 'O' + result_p[2:]

            if rename_str == "": # EM added: problem requirement
                if '危' not in result_p and '险' not in result_p and '品' not in result_p: # 不识别危险品标志
                    rename_str = result_p

            result_p += " " + result['plate_color'] + ("双层" if result['object_no'] == 1 else "")
            result_str += result_p + " "

            if do_draw:
                for i in range(4): # 绘制车牌角点
                    cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)
                
                labelSize = cv2.getTextSize(result_p, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) # 获得字体的大小
                if rect_area[0] + labelSize[0][0] > orgimg.shape[1]: # 防止显示的文字越界
                    rect_area[0] = int(orgimg.shape[1] - labelSize[0][0])
                orgimg = cv2.rectangle(orgimg, # 画文字框
                                       (rect_area[0], int(rect_area[1] - round(1.6 * labelSize[0][1]))), 
                                       (int(rect_area[0] + round(1.2 * labelSize[0][0])), rect_area[1] + labelSize[1]), 
                                       (255, 255, 255), 
                                       cv2.FILLED)
                
                if len(result) >= 1:
                    orgimg = cv2ImgAddText(orgimg, 
                                           result['plate_no'], 
                                           rect_area[0], 
                                           int(rect_area[1] - round(1.6 * labelSize[0][1])), 
                                           (0, 0, 0), 
                                           21)
        
        if do_draw:
            cv2.rectangle(orgimg,
                        (rect_area[0],rect_area[1]),
                        (rect_area[2],rect_area[3]),
                        object_color[object_no],
                        2) # 画 ROI 框       
    
    print(result_str + '\033[K')

    return orgimg, rename_str

def process_single_image(count, 
                         img_path,
                         detect_model,
                         device,
                         car_rec_model,
                         plate_rec_model,
                         img_size,
                         is_color,
                         do_draw,
                         save_path
                         ):
    """
    处理单张图片. 

    ## Parameters:
    `count`: 图片张数计数器.

    `img_path`: 待处理图片路径.

    `detect_model`: 加载好的车牌检测模型. 

    `device`: 用 cpu 或 gpu 处理.

    `plate_rec_model`: 加载好的车牌识别模型.

    `img_size`:

    `is_color`: 是否识别颜色.

    `do_draw`: 是否绘制识别框.

    `save_path`: 处理结果保存路径.
    """

    print(count + 1, img_path, end=" ")
    img = cv_imread(img_path)

    if img is None:
        print('Cannot open image: %s. ' % (img_path))
        return
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    dict_list = detect_recognition_plate(detect_model, img, device,
                                         car_rec_model, plate_rec_model,
                                         img_size, is_color) # 识别车辆, 检测以及识别车牌
    
    ori_img, result_str = draw_result(img, dict_list, do_draw) # 将车辆和车牌识别结果画在图上, 并输出车牌字符串
    
    img_name = '_' + result_str + '_' + os.path.basename(img_path) # EM modified: problem requirement
    save_img_path = os.path.join(save_path, img_name) # 图片保存的路径
    cv_imwrite(ori_img, save_img_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('--no_prompt', action='store_true', help='是否开启提示信息')

    parser.add_argument('--detect_model', nargs = '+', type = str, default = 'weights/plate_detect.pt', help = '检测模型路径, model.pt')
    parser.add_argument('--plate_rec_model', type=str, default='weights/plate_rec_color.pth', help='车牌识别模型路径, model.pth')
    parser.add_argument('--car_rec_model', type=str, default='weights/car_rec_color.pth', help='车辆识别模型路径, model.pth')

    parser.add_argument('--is_video', action='store_true', help='处理图片还是视频')
    parser.add_argument('--is_color', action='store_true', help='是否识别颜色')
    parser.add_argument('--do_draw', action='store_true', help='是否绘制识别框')
    
    parser.add_argument('--image_path', type=str, default='input/imgs/', help='待识别图片(目录)路径')
    parser.add_argument('--video_path', type=str, default='input/videos/test_s.mp4', help='待识别视频路径')
    parser.add_argument('--img_size', type=int, default=384, help='需为 32 的倍数. 该参数会影响识别阈值, 提高该参数会使低分目标显示出来')
    
    parser.add_argument('--output', type=str, default='output/', help='处理结果保存位置')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use cpu or gpu
    opt = parser.parse_args()
    print(opt)

    os.system('cd ./') # 激活 \033[ 命令行光标操作转义字符.

    save_path = opt.output
    if not os.path.exists(save_path): 
        os.mkdir(save_path)

    detect_model = load_model(opt.detect_model, device) # init detect model
    plate_rec_model = init_plate_rec_model(opt.plate_rec_model, device) # init rec model
    car_rec_model = init_car_rec_model(opt.car_rec_model, device)

    # 计算参数量
    total_car_rec = sum(p.numel() for p in car_rec_model.parameters())
    total_plate_det = sum(p.numel() for p in detect_model.parameters())
    total_plate_rec = sum(p.numel() for p in plate_rec_model.parameters())
    print("========\n车辆检测模型参数量: %.2f 万. " % (total_car_rec / 1e4))
    print("车牌检测模型参数量: %.2f 万. " % (total_plate_det / 1e4))
    print("车牌识别模型参数量: %.2f 万. \n========" % (total_plate_rec / 1e4))
    
    if not opt.is_video: # Image detection and recognition
        count = 0 # 处理项目数计数器

        if os.path.isfile(opt.image_path): # Single image file input
            process_single_image(count, opt.image_path, detect_model, device, car_rec_model, plate_rec_model,
                                 opt.img_size, opt.is_color, opt.do_draw, save_path)
        else: # Directory input.
            time_all = 0
            time_begin = time.time()

            file_list = []
            allFilePath(opt.image_path, file_list) # 将该目录下的所有图片路径读取到 file_list

            for img_path in file_list: # 遍历图片文件
                time_b = time.time() # 开始时间

                process_single_image(count, img_path, detect_model, device, car_rec_model, plate_rec_model,
                                     opt.img_size, opt.is_color, opt.do_draw, save_path)
                
                time_e = time.time()
                time_gap = time_e - time_b # 计算单个图片识别耗时
                if count:
                    time_all += time_gap 
                count += 1

            print(f"处理总用时 {time.time() - time_begin:.2f} s, 平均处理速率 {time_all / (len(file_list) - 1):.2f} fps. ")
    else: # Video input.
        video_name = opt.video_path
        capture = cv2.VideoCapture(video_name)
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = capture.get(cv2.CAP_PROP_FPS) # 帧数
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 宽高

        extension = 0
        for index in range(len(video_name)):
            rev_ind = len(video_name) - index - 1
            if video_name[rev_ind] == '.':
                extension = rev_ind
                break
        output_path = os.path.basename(video_name[:extension] + '_result.mp4')
        output_path = os.path.join(save_path, output_path)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height)) # 写入视频

        frame_count = 0
        fps_all = 0

        if capture.isOpened():
            # See https://blog.csdn.net/u011436429/article/details/80604590
            # for OpenCV VideoCapture.get() arguments.
            totalFrames = int(capture.get(7)) # 视频文件的帧数

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
                dict_list = detect_recognition_plate(detect_model, img, device, car_rec_model, 
                                                     plate_rec_model, opt.img_size, opt.is_color)
                ori_img, _ = draw_result(img, dict_list, opt.do_draw)
                
                t2 = cv2.getTickCount()
                infer_time = (t2 - t1) / cv2.getTickFrequency()
                fps = 1.0 / infer_time

                eta = (totalFrames - frame_count) * infer_time
                fps_all += fps
                # str_fps = f'Processing fps: {fps:.4f}'
                
                # 写入处理帧信息.
                # cv2.putText(ori_img, str_fps, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                out.write(ori_img)

                # 每处理 15 帧更新一次预测剩余时间.
                print(f"预计剩余时间: {int(eta)} s. \033[K" if not (frame_count % 15) else '')
        else:
            print("视频加载失败. ")

        capture.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"共处理 {frame_count} 帧, 平均处理帧速率 {fps_all / frame_count:.2f} fps. ")