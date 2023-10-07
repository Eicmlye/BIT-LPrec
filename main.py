# -*- coding: UTF-8 -*-

"""
直接从原始图像中检测车牌并识别车牌号码等信息. 
"""

from monitor import MediaProcessor

from utils.io.cmd_cursor import activate_cmd_cursor_opr # EM added
from utils.test.cmd_args import Parser # EM added

if __name__ == '__main__':
    parser = Parser()
    processor = MediaProcessor(parser.opt)

    activate_cmd_cursor_opr()

    processor.run()