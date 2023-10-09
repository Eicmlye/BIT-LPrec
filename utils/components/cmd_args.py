import argparse

class Parser:
    """
    命令行参数解析. 
    """
    
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--detect_model', nargs = '+', type = str, default = 'weights/plate_detect.pt', help = '检测模型路径, model.pt')
        parser.add_argument('--car_rec_model', type=str, default='weights/car_rec_color.pth', help='车辆识别模型路径, model.pth')
        parser.add_argument('--plate_rec_model', type=str, default='weights/plate_rec_color.pth', help='车牌识别模型路径, model.pth')
        
        parser.add_argument('--try_gpu', type=bool, default=None, help="是否使用 GPU, 默认为自动选择")
        parser.add_argument('--img_size', type=int, default=384, help='需为 32 的倍数. 该参数会影响识别阈值, 提高该参数会使低分目标显示出来')

        parser.add_argument('--is_video', action='store_true', help='处理图片还是视频')
        parser.add_argument('--do_draw', action='store_true', help='是否绘制识别框')
        parser.add_argument('--is_color', action='store_true', help='是否识别颜色')
        
        parser.add_argument('--image_path', type=str, default='input/imgs/', help='待识别图片(目录)路径')
        parser.add_argument('--video_path', type=str, default='input/videos/short.mp4', help='待识别视频路径')
        
        parser.add_argument('--output', type=str, default=None, help='处理结果保存位置')

        self.opt = parser.parse_args()