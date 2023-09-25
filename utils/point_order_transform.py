# EM reconstructed.

import numpy as np
import cv2

def order_points(pts):
    """
    将目标框的四角关键点坐标按从左上角开始顺时针排列.

    ## Parameters:
    `pts`: `4*2` 的 `numpy.ndarray`, 目标框的四角关键点坐标列表, 对顺序没有要求. 

    ## Return:
    `rect`: 按左上角、右上角、右下角、左下角的顺序排列的关键点坐标列表.
    """
    rect = np.zeros((4, 2), dtype = "float32")

    sum = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(sum)] # top left
    rect[2] = pts[np.argmax(sum)] # bottom right

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)] # top right
    rect[3] = pts[np.argmax(diff)] # bottom left

    return rect

def four_point_transform(image, pts):
    """
    将截得的车牌图像透视变换为标准化正视图. 

    ## Parameters:
    `image`: 原始图像.

    `pts`: 这里要求 `pts` 已经被 `order_points()` 标准化排序.

    ## Return:
    `warped`: 标准化大小的正视图, 保持了透视信息. 
    """
    
    rect = pts.astype('float32')
    (tl, tr, br, bl) = rect

    # See https://theailearner.com/tag/cv2-getperspectivetransform/
    # for OpenCV perspective transformation.
    bottom_width = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    top_width = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(bottom_width), int(top_width))

    right_height = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    left_height = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(right_height), int(left_height))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped