import torch
import cv2
import numpy as np
import os
import time
import sys

from networks.plate_recognition.plate_rec_net import plateNet_ocr_color
from utils.components.strmod import get_all_file_path

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
plate_name = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
color_list = ['黑色', '蓝色', '绿色', '白色', '黄色']
mean_value, std_value = (0.588, 0.193)
    
def init_plate_rec_model(model_path: str, device: torch.device):
    # print(print(sys.path))
    # model_path = "plate_recognition/model/checkpoint_61_acc_0.9715.pth"
    check_point = torch.load(model_path, map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point['cfg']
    model_path = os.sep.join([sys.path[0], model_path])
    model = plateNet_ocr_color(num_classes=len(plate_name), export=True, 
                               cfg=cfg, color_num=len(color_list))
   
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model

def image_processing(img, device: torch.device):
    img = cv2.resize(img, (168, 48))
    img = np.reshape(img, (48, 168, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    return img

def get_plate_result(img: torch.Tensor, device: torch.device, model: plateNet_ocr_color):
    input_ = image_processing(img, device)
    
    preds, color_preds = model(input_)
    # print(preds) # EM DEBUG
    preds = preds.argmax(dim=2) # 找出概率最大的那个字符
    color_preds = color_preds.argmax(dim=-1)
    
    preds = preds.view(-1).detach().cpu().numpy()
    # print(preds) # EM DEBUG
    color_preds = color_preds.item()
    new_preds = decode_plate(preds)
    # print(new_preds) # EM DEBUG
    # input() # EM DEBUG
    plate = ""
    for i in new_preds[0]: # EM modified. Added '[0]'
        plate += plate_name[i]
        
    return plate, color_list[color_preds] # 返回车牌牌号和颜色

def decode_plate(preds: torch.Tensor):
    pre = 0
    newPreds = []
    index = []

    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
            index.append(i)
        pre = preds[i]
    return newPreds, index

# model = init_model(device)
if __name__ == '__main__':
   model_path = r"weights/plate_rec_color.pth"
   image_path = "images/tmp2424.png"
   testPath = r"/mnt/Gpan/Mydata/pytorchPorject/CRNN/crnn_plate_recognition/images"
   fileList = []
   get_all_file_path(testPath, fileList, ['.jpg', '.png', '.JPG'])

   is_color = False
   model = init_plate_rec_model(device, model_path)
   right = 0
   begin = time.time()
   
   for imge_path in fileList:
        img = cv2.imread(imge_path)
        if is_color:
            plate, _, plate_color, _ = get_plate_result(img, device, model, is_color)
            print(plate)
        else:
            plate, _ = get_plate_result(img, device, model, is_color)
            print(plate, imge_path)
        
  
        
