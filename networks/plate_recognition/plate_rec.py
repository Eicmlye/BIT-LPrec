import torch
import cv2
import numpy as np
import os
import time
import sys

from networks.plate_recognition.plateNet import plateNet_ocr_color

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
color_list = ['黑色', '蓝色', '绿色', '白色', '黄色']
mean_value, std_value = (0.588, 0.193)

def allFilePath(rootPath, allFileList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            if temp.endswith('.jpg') or temp.endswith('.png') or temp.endswith('.JPG'):
                allFileList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp), allFileList)

def decodePlate(preds):
    pre = 0
    newPreds = []
    index = []

    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
            index.append(i)
        pre = preds[i]
    return newPreds, index

def image_processing(img, device):
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

def get_plate_result(img, device, model, is_color=False):
    input = image_processing(img, device)
    
    preds, color_preds = model(input)
    preds = preds.argmax(dim=2) # 找出概率最大的那个字符
    color_preds = color_preds.argmax(dim=-1)
    
    preds = preds.view(-1).detach().cpu().numpy()
    color_preds = color_preds.item()
    newPreds = decodePlate(preds)
    plate = ""
    for i in newPreds[0]: # EM modified. Added '[0]'
        plate += plateName[i]
        
    if is_color:
        return plate, color_list[color_preds] # 返回车牌牌号和颜色
    else:
        return plate
    
def init_plate_rec_model(model_path, device):
    # print( print(sys.path))
    # model_path ="plate_recognition/model/checkpoint_61_acc_0.9715.pth"
    check_point = torch.load(model_path,map_location=device)
    model_state=check_point['state_dict']
    cfg=check_point['cfg']
    model_path = os.sep.join([sys.path[0],model_path])
    model = plateNet_ocr_color(num_classes=len(plateName), export=True, 
                               cfg=cfg, color_num=len(color_list))
   
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model

# model = init_model(device)
if __name__ == '__main__':
   model_path = r"weights/plate_rec_color.pth"
   image_path = "images/tmp2424.png"
   testPath = r"/mnt/Gpan/Mydata/pytorchPorject/CRNN/crnn_plate_recognition/images"
   fileList = []
   allFilePath(testPath, fileList)

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
        
  
        
