"""
命令行输出光标的控制. 
"""

import os

def activate_cmd_cursor_opr():
    # See https://blog.csdn.net/qq_51427262/article/details/128571536
    # for necessity of this line. 
    # See https://blog.csdn.net/yuhai738639/article/details/79221835
    # for \033[ usages. 
    os.system('cd ./') # 激活 \033[ 命令行光标操作转义字符.