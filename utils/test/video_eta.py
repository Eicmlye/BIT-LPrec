"""
视频处理的预计剩余时间. 
"""

import cv2
import time

def video_processing_prompt(time_gap, totalFrames, curFrame, process_time, last_flush):
    """
    更新处理所用的总时间, 并打印预计剩余时间. 
    """
    
    infer_time = time_gap / cv2.getTickFrequency()

    eta = (totalFrames - curFrame) * infer_time
    process_time += infer_time
    
    # 每秒更新一次预测剩余时间.
    do_flush = (time.time() - last_flush >= 1)
    print(f"已处理 {int(process_time)} s, 预计还需 {int(eta)} s. \033[K" if do_flush else '')
    last_flush = time.time() if do_flush else last_flush

    return process_time, last_flush