# EM reconstructed.

"""
解决 OpenCV 的文字 I/O 不兼容中文字符的诸多问题. 
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def cv_imread(path):
    """
    读取路径含中文字符的图片. 
    """
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img

def cv_imwrite(img, path, filetype='.jpg'):
    """
    输出文件名含中文字符的图片, 防止其文件名乱码. 
    """

    # EM modified. Solved bug for bad GBK encoding in output names.
    # https://www.zhihu.com/question/47184512/answer/136012000
    cv2.imencode(filetype, img)[1].tofile(path)

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    """
    向 OpenCV 格式的图片上嵌入中文文本, 并防止其乱码. 

    本质是使用 `pillow` 包作为中介. 
    """

    if (isinstance(img, np.ndarray)): # 判断是否 OpenCV 图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "fonts/platech.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)