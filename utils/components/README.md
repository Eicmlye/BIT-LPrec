## `utils/components/`

本目录包含程序实际使用时的功能组件. 

### `utils/components/cmd_args.py`

`class Parser` 负责命令行参数的解析. 

### `utils/components/monitor.py`

`class MediaProcessor` 接收 `Parser` 解析的参数信息, 创建对应的处理资源, 并控制图像和视频处理的用户接口.

### `utils/components/detect.py`

`class Detecter` 负责图像和视频的实际处理实现与结果可视化. 

### `utils/components/strmod.py`

`class FilenameModifier` 负责输出图像、视频和日志文件名的控制, 以及视频中潜在车位的检测、记录及车位订单信息的生成. 