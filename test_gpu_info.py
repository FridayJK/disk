import torch, h5py, imageio, os, argparse
import numpy as np
import pynvml

os.environ["CUDA_VISIBLE_DEVICES"] = "1"




pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(1) # 指定显卡号
meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)

print(meminfo.total/1024**2) #总的显存大小（float）
print(meminfo.used/1024**2)  #已用显存大小（float）
print(meminfo.free/1024**2)  #剩余显存大小（float）