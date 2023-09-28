"""
命令行参数的格式化输出.
"""

import os
import argparse

from utils.io.modify_filename import get_extension_index

def show_args(opt: argparse.Namespace, device_choice: bool):
    """
    输出命令行参数的详细信息. 
    """

    print("========")
    print("车辆识别模型: " + opt.car_rec_model)
    print("车牌检测模型: " + opt.detect_model)
    print("车牌识别模型: " + opt.plate_rec_model)

    print(("待识别视频: " + opt.video_path) if opt.is_video else ("待识别图片: " + opt.image_path))\

    output_path = opt.output
    if output_path == None:
        output_path = 'output/'
        if opt.is_video:
            video_name = opt.video_path
            extension = get_extension_index(video_name)

            output_path = os.path.join(output_path, os.path.basename(video_name[:extension])) + '/'

    opt.output = output_path
    print("处理结果保存位置: " + output_path)

    print("模型超参数设置: ")
    print("\t处理精度(需为 32 的倍数): " + str(opt.img_size))

    # extra settings
    print("其他设置: ")

    print("\t自动选择了可用的 GPU" if opt.use_gpu == None and device_choice else '', end='')
    print("\t无可用GPU, 自动选择了可用的 CPU" if opt.use_gpu == None and not device_choice else '', end='')
    print("\t以 GPU 模式处理" if opt.use_gpu == True and device_choice else '', end='')
    print("\t无可用GPU, 强制以 CPU 模式处理" if opt.use_gpu == True and not device_choice else '', end='')
    print("\t以 CPU 模式处理" if opt.use_gpu == False else '', end='')
    print('')
    
    print('\t' + ('' if opt.do_draw else '不') + "绘制识别框")
    print('\t' + ('' if opt.is_color else '不') + "输出对象颜色信息")
        
    # ending line
    print("========")