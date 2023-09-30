"""
视频处理的预计剩余时间. 
"""

import cv2
import time

def time2str(time_in_sec: int):
    assert time_in_sec >= 0

    day = time_in_sec // 3600 // 24

    result = ""
    if day > 30:
        result += f"至少 30 天"
        return result
    
    if day > 0:
        result += f"{day} 天 "

    time_in_sec -= day * 3600 * 24
    hr = time_in_sec // 3600

    time_in_sec -= hr * 3600
    min = time_in_sec // 60
    
    time_in_sec -= min * 60
    sec = time_in_sec % 60

    if hr > 0:
        result += f"{hr:02d}:{min:02d}:{sec:02d}"
    elif min > 0:
        result += f"{min:02d}:{sec:02d}"
    elif sec >= 0:
        result += f"{sec} s"

    return result

def video_processing_prompt(time_gap, totalFrames, curFrame, process_time, last_flush):
    """
    更新处理所用的总时间, 并打印预计剩余时间. 
    """
    
    infer_time = time_gap / cv2.getTickFrequency()

    eta = (totalFrames - curFrame) * infer_time
    process_time += infer_time
    
    # 每秒更新一次预测剩余时间.
    do_flush = (time.time() - last_flush >= 1)
    print(("已处理 " + time2str(int(process_time)) + ", 预计还需 " + time2str(int(eta)) + ". \033[K") if do_flush else '')
    last_flush = time.time() if do_flush else last_flush

    return process_time, last_flush