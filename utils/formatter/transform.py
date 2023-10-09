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

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    将 `img1` 中的角点坐标按比例缩放至 `img0` 中. 
    """

    # Rescale coords (xyxy) from img1_shape to img0_shape
    # EM note: this coords is NOT xyxy, it's landmarks instead.
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

def get_split_merge(img):
    """
    将双行车牌整形拼裁为单行车牌.
    """
    h, w, c = img.shape
    img_upper = img[0: int(5 / 12 * h), :]
    img_lower = img[int(1 / 3 * h): , :]
    img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))
    new_img = np.hstack((img_upper, img_lower))
    
    return new_img