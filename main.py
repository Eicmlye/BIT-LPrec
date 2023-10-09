# -*- coding: UTF-8 -*-

"""
直接从原始图像中检测车辆、车牌并识别车牌号码等信息. 
"""

from utils.components.monitor import MediaProcessor

from utils.formatter.cmd_cursor import activate_cmd_cursor_opr
from utils.components.cmd_args import Parser

if __name__ == '__main__':
    activate_cmd_cursor_opr()
    
    parser = Parser()
    processor = MediaProcessor(parser.opt)

    processor.run()