# -*- coding: UTF-8 -*-

"""
直接从原始图像中检测车牌并识别车牌号码等信息. 
"""

from detect import Detecter

from utils.io.cmd_cursor import activate_cmd_cursor_opr # EM added
from utils.test.cmd_args import Parser # EM added

if __name__ == '__main__':
    parser = Parser()
    detecter = Detecter(parser.opt)

    activate_cmd_cursor_opr()

    detecter.run()