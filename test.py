import numpy as np
import torch
import torch.utils.data as data
import os
from PIL import Image
import random

# semantic_image = Image.open('../data/train/label/0.png').point(lambda i: i * 80).convert('RGB')
semantic_image = Image.open('../data/train/label/0.png').convert('RGB')  # (76,76,76)
w,h = semantic_image.size
#遍历的到图像中所有像素点的RGB颜色值
for row in range(w):
	for cloumn in range(h):
			print(semantic_image.getpixel((row,cloumn)))
	#getpixel函数是用来获取图像中某一点的像素的RGB颜色值，getpixel的参数是一个像素点的坐标。
    #对于图象的不同的模式，getpixel函数返回的值不同。
semantic_image.show()