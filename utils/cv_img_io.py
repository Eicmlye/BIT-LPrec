# EM reconstructed.

import cv2
import numpy as np

def cv_imread(path): # 可以读取中文路径的图片
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
    return img

def cv_imwrite(img, path, filetype='.jpg'):
    # EM modified. Solved bug for bad GBK encoding in output names.
    # https://www.zhihu.com/question/47184512/answer/136012000
    cv2.imencode(filetype, img)[1].tofile(path)