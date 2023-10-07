import sys
import torch
import cv2
import time

from argparse import Namespace

from detect import Detecter
from utils.io.strmod import FilenameModifier

from models.experimental import attempt_load
from networks.car_recognition.car_rec import init_car_rec_model
from networks.plate_recognition.plate_rec import init_plate_rec_model
from utils.io.cv_img import cv_imread, cv_imwrite
from utils.io.strmod import control_filename_len # EM added
from utils.test.parking_detect import ParkingPos, CandidateParkingPos, update_parking_info, save_action_info # EM added
from utils.test.video_eta import video_processing_prompt # EM added

class MediaProcessor:
    """
    对象检测及识别的参数和操作管理. 
    """

    # static attributes
    image_extensions = ['.jpg', '.png', '.JPG', '.PNG']
    
    def __init__(self, opt: Namespace):
        self.use_gpu, self.device = self._choose_device(opt.try_gpu) # GPU 或 CPU
        models = self._load_models({
                                            'detect': opt.detect_model,
                                            'carrec': opt.car_rec_model,
                                            'platerec': opt.plate_rec_model
                                        }) # 加载好的模型字典

        precision = opt.img_size # 处理精度

        self.is_video = opt.is_video # 待处理的是否为视频
        # 文件名处理器
        self.name_modifier = FilenameModifier(opt.video_path if self.is_video else opt.image_path,
                                              self.is_video, self.image_extensions, opt.output)

        draw_rec = opt.do_draw # 是否绘制 ROI 区域选框
        show_color = opt.is_color # 是否输出识别到的颜色属性

        self._show_args(opt) # 打印程序设置

        self.detecter = Detecter(precision, models, self.device, draw_rec, show_color)

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
        print("\t处理精度(需为 32 的倍数): " + str(opt.img_size))

        # extra settings
        print("其他设置: ")

        print("\t自动选择了可用的 GPU" if opt.try_gpu == None and self.use_gpu else '', end='')
        print("\t无可用GPU, 自动选择了可用的 CPU" if opt.try_gpu == None and not self.use_gpu else '', end='')
        print("\t以 GPU 模式处理" if opt.try_gpu == True and self.use_gpu else '', end='')
        print("\t无可用GPU, 强制以 CPU 模式处理" if opt.try_gpu == True and not self.use_gpu else '', end='')
        print("\t以 CPU 模式处理" if opt.try_gpu == False else '', end='')
        print('')

        print('\t' + ('' if opt.do_draw else '不') + "绘制识别框")
        print('\t' + ('' if opt.is_color else '不') + "输出对象颜色信息")
            
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
        obj_list = self.detecter.detect_recognition_plate(img) # 识别车辆, 检测并识别车牌
        
        ori_img, primary_plate = self.detecter.visualize_result(img, obj_list)

        self.name_modifier.standard_image_filename(count, primary_plate)
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

            if capture.isOpened():
                # See https://blog.csdn.net/u011436429/article/details/80604590
                # and https://blog.csdn.net/qq_43797817/article/details/108096827
                # for OpenCV VideoCapture.get() arguments.
                totalFrames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) # 视频文件的帧数

                last_flush = time.time() # 上次刷新预测剩余时间的时刻
                
                candidates = list[CandidateParkingPos]()
                parkings = list[ParkingPos]()

                while True:
                    t1 = cv2.getTickCount()
                    ret, img = capture.read()
                    if frame_count > 0 and ret: # 刷新式更新处理进度信息.
                        print('\r\033[3A', end='')
                    if not ret:
                        break
                    
                    frame_count += 1
                    print(f"({frame_count / totalFrames * 100:.2f}%) 第 {frame_count}/{totalFrames} 帧\033[K")
                    obj_list = self.detecter.detect_recognition_plate(img)

                    candidates, parkings = update_parking_info(frame_count, obj_list,
                                                             int(totalfps), candidates, parkings)

                    ori_img, _ = self.detecter.visualize_result(img, obj_list)

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
            save_action_info(parkings, capture, self.name_modifier.output_root)

            capture.release()
            out.release()
            cv2.destroyAllWindows()
    
