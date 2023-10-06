import cv2
import numpy as np
from utils.io.cv_img import cv_imread # EM reconstructed

def get_split_merge(img):
    h, w, c = img.shape
    img_upper = img[0: int(5 / 12 * h), :]
    img_lower = img[int(1 / 3 * h): , :]
    img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))
    new_img = np.hstack((img_upper, img_lower))
    
    return new_img

if __name__ == "__main__":
    img = cv_imread("double_plate/tmp8078.png")
    new_img = get_split_merge(img)
    cv2.imwrite("double_plate/new.jpg", new_img)
