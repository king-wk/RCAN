# this file for testing the parameters and flops of the model in specific resolution and scale

from thop import profile
import torch

import utility
import data
import model
import loss
from option import args

model_name = "RCAN"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 检测GPU
# (3840 2160) (1920 1080) (1280 720) (960 540)
# (1920 1080) (960 540) (640 360) (480 270)
width = 480
height = 270
input = torch.randn(1,3,width,height).to(device)
print(args)
checkpoint = utility.checkpoint(args)
model = model.Model(args, checkpoint)
flops, params = profile(model, inputs=(input, 0, 10))
print("%s|param %.2f|FLOPS %.2f"%(model_name, params / (1000 ** 2), flops / (1000 ** 3)))
