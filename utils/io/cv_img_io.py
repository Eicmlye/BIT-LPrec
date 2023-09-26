# EM reconstructed.

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def cv_imread(path): # 可以读取中文路径的图片
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img

def cv_imwrite(img, path, filetype='.jpg'):
    # EM modified. Solved bug for bad GBK encoding in output names.
    # https://www.zhihu.com/question/47184512/answer/136012000
    cv2.imencode(filetype, img)[1].tofile(path)

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)): # 判断是否 OpenCV 图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "fonts/platech.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)