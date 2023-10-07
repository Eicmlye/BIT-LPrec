import os

class OutputItem:
    """
    一个待处理文件对应的输出文件项.
    """

    def __init__(self, is_video, input_filebasename: str):
        # self.basename
        self._basename = input_filebasename

        # self.files
        self._files = {}
        if is_video: # video file
            name_without_extension = self._basename[:get_extension_index(self._basename)]
            
            self._files['result'] = name_without_extension + "_result.mp4"
            self._files['log'] = name_without_extension + ".log"
        else: # image file
            self._files['result'] = self._basename
    
    def get_result(self):
        return self._files['result']
    
    def get_log(self):
        return self._files['log']

    def standard_image_filename(self, rec_result: str):
        """
        图像文件名格式: _识别结果_原文件名.原拓展名
        """
        self._files['result'] = '_' + rec_result + '_' + os.path.basename(self.get_result())

class OutputFilelist:
    """
    输出文件名列表.
    """

    def __init__(self, output_root: str, is_video: bool, input_filelist: list[str]):
        # self.output_root
        self._output_root = output_root

        # self.contents
        self._contents = list[OutputItem]()

        for filename in input_filelist:
            item = OutputItem(is_video, os.path.basename(filename))
            self._contents.append(item)
    
    def get_result(self, fileindex: int):
        return self._output_root + self._contents[fileindex].get_result()
    
    def get_log(self, fileindex: int):
        return self._output_root + self._contents[fileindex].get_log()

    def standard_image_filename(self, fileindex: int, rec_result: str):
        """
        图像文件名格式: _识别结果_原文件名.原拓展名
        """
        self._contents[fileindex].standard_image_filename(rec_result)

        return

class FilenameModifier:
    """
    修改文件名的格式.
    """

    def __init__(self, input_filename: str, is_video: bool, extension: list[str] = None, provide_output: str = None):
        # self.root
        self.root = input_filename # 待处理文件的根目录
        # self.input_list
        self.input_list = [] # 待处理文件列表
        get_all_file_path(self.root, self.input_list, extension)

        # self.output_root
        self.output_root = self._standard_output_dirname(is_video, provide_output) # 输出位置根目录
        # self.output_list
        self.output_list = OutputFilelist(self.output_root, is_video, self.input_list) # 输出文件名列表, 顺序与 `self.input_list` 对应
    
    def get_result(self, fileindex: int):
        """
        返回对应的处理结果文件路径. 
        """
        
        return self.output_list.get_result(fileindex)
    
    def get_log(self, fileindex: int):
        """
        对于视频文件, 返回其日志文件路径.
        """
        
        return self.output_list.get_log(fileindex)

    def _standard_output_dirname(self, is_video: bool, provide_output: str = None):
        """
        生成输出文件的根目录路径.
        """

        if provide_output is None: # 默认输出位置
            provide_output = 'output/' # 图片默认直接输出到 output/

        if is_video: # 视频及订单输出到该目录下与原视频同名的子目录中
            video_name = self.root
            extension = get_extension_index(video_name)
            provide_output = os.path.join(provide_output, os.path.basename(video_name[:extension])) + '/'

        if not provide_output.endswith('/') and not provide_output.endswith('\\'):
            provide_output += '/'

        if not os.path.exists(provide_output): # 创建输出目录
            os.mkdir(provide_output)

        return provide_output

    def standard_image_filename(self, fileindex: int, rec_result: str):
        """
        图像文件名格式: _识别结果_原文件名.原拓展名
        """
        self.output_list.standard_image_filename(fileindex, rec_result)
        
        return

def get_all_file_path(root_path: str, all_file_list: list[str], extension: list[str] = None):
    """
    递归地取出 `root_path` 中所有以 `extension` 的元素结尾的文件路径, 并追加到 `all_file_list` 中.
    """
    
    if os.path.isfile(root_path): # 根目录为文件时, 直接处理该文件
        all_file_list.append(root_path)

        return

    # 根目录非文件时, 处理其中的所有文件和子目录
    file_list = os.listdir(root_path) # 获取目录中的所有文件和子目录路径

    # 不指定特定拓展名, 则将所有文件路径都取出
    if extension is None:
        for temp in file_list:
            if os.path.isfile(os.path.join(root_path, temp)):
                all_file_list.append(os.path.join(root_path, temp))
            else: # 递归取出子目录中的文件路径
                get_all_file_path(os.path.join(root_path, temp), all_file_list)
        
        return

    # 指定拓展名, 则只取出相应拓展名的文件路径
    for temp in file_list:
        if os.path.isfile(os.path.join(root_path, temp)):
            for ext in extension:
                if (temp.endswith(ext)):
                    all_file_list.append(os.path.join(root_path, temp))
                    break
        else: # 递归取出子目录中的文件路径
            get_all_file_path(os.path.join(root_path, temp), all_file_list)

def get_extension_index(filename: str):
    """
    捕获拓展名的分割点位置.
    """

    for index in range(len(filename)):
        rev_ind = len(filename) - index - 1
        if filename[rev_ind] == '.':
            return rev_ind
        
    raise RuntimeError("No extension found. ")
        
def control_filename_len(path: str, maxlen: int = 20):
    """
    将文件名字符串的长度控制在一定范围内, 若超长则省略中间的部分内容. 
    """
    
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