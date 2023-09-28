import os

def get_extension_index(filename: str):
    """
    捕获拓展名的分割点位置.
    """

    for index in range(len(filename)):
        rev_ind = len(filename) - index - 1
        if filename[rev_ind] == '.':
            return rev_ind
        
def control_filename_len(path: str, maxlen: int = 20):
    if (maxlen <= 10):
        print("Control length too short (lower than 11). ")
        return path

    short_path = os.path.basename(path)
    if len(short_path) > maxlen:
        extension = get_extension_index(short_path)

        # 省略号占 3 字符, 拓展名占 len(short_path) - extension 字符
        prefix_len = int((maxlen - 3 - (len(short_path) - extension)) / 2)

        short_path = short_path[:prefix_len] + '...' + short_path[extension - prefix_len:]
        
    return short_path