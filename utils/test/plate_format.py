"""
车牌格式的检查和纠正.
"""

provinces = "京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新"
department = "学警港澳挂使领民航"
digits = "0123456789"
letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"

def rename_special_plate(plate: str):
    """
    特殊车牌纠正. 

    京O牌为警用. 
    """

    if plate[0] == '京' and plate[1] == '0':
        plate = plate[0] + 'O' + plate[2:]

    return plate

def check_plate_format(plate: str):
    """
    检查识别出的车牌的基本格式是否正确. 
    """

    # 车牌号长度检查
    if len(plate) not in [7, 8]:
        return False
    
    # 首字为省份信息
    if plate[0] not in provinces:
        return False
    
    plate = rename_special_plate(plate)
    # 第二位为归属信息
    if plate[1] not in letters:
        return False
    
    # 除第二位外, 至多 2 个字母
    # 考虑到有挂字车、港澳车等, 不应检查其它字符数量
    count_letter = 0
    for index in range(2, len(plate)):
        count_letter += 1 if plate[index] in letters else 0
        if count_letter > 2:
            return False
        
    return True