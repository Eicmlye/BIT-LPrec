"""
加载测试模型时可能用到的函数.
"""

import torch
import sys

from argparse import Namespace
from models.experimental import attempt_load
from networks.plate_recognition.plate_rec import init_plate_rec_model
from networks.car_recognition.car_rec import init_car_rec_model

def init_detect_model(weights, device):
    """
    加载检测模型. 
    """
    model = attempt_load(weights, map_location = device)  # load FP32 model

    return model

def load_models(opt: Namespace, device):
    """
    加载所有要用到的模型, 并计算、输出其参数总量.
    """

    car_rec_model = init_car_rec_model(opt.car_rec_model, device)
    detect_model = init_detect_model(opt.detect_model, device) # init detect model
    plate_rec_model = init_plate_rec_model(opt.plate_rec_model, device) # init rec model

    # 计算参数量
    total_car_rec = sum(p.numel() for p in car_rec_model.parameters())
    total_plate_det = sum(p.numel() for p in detect_model.parameters())
    total_plate_rec = sum(p.numel() for p in plate_rec_model.parameters())
    print("========")
    print("车辆检测模型参数量: %.2f 万. " % (total_car_rec / 1e4))
    print("车牌检测模型参数量: %.2f 万. " % (total_plate_det / 1e4))
    print("车牌识别模型参数量: %.2f 万. " % (total_plate_rec / 1e4))
    print("========")

    return car_rec_model, detect_model, plate_rec_model

def choose_device(use_gpu: bool):
    """
    根据用户意愿和实际可用资源, 分配 GPU 或 CPU. 

    ## Parameters:

    `use_gpu`: 
        
        - `None`: 根据可用资源, 自动优先选择 GPU.
        - `True`: 有 GPU 就用 GPU, 没有 GPU 就询问用户是否强制结束程序. 
        - `False`: 无论是否有 GPU, 都强制用 CPU. 

    ## Return:

    `device`: 获得的资源, `torch.device` 类型. 

    `device_choice`: `True`: GPU, `False`: CPU.
    """

    device_choice = None
    if use_gpu == None:
        device_choice = torch.cuda.is_available()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif torch.cuda.is_available():
        device_choice = use_gpu
        device = torch.device("cuda" if use_gpu else "cpu")
    else:
        device_choice = False
        
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

    return device, device_choice