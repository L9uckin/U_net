import numpy as np
import torch
import torch.utils.data as data
import os
from PIL import Image
import random


def is_image_file(filename):  # 定义一个判断是否是图片的函数
    #获取所有图片
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def trans_to_tensor(pic):  # 定义一个转变图像格式的函数
    if isinstance(pic, np.ndarray):#pic是否是np.ndarray类型
        #转置 将图片数组转换为张量
        img = torch.from_numpy(pic.transpose((2, 0, 1)))  # transpose和reshape区别巨大
        return img.float().div(255)
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    if pic.mode == 'YCbCr':
        #通道数
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    img = img.transpose(0, 1).transpose(0, 2).contiguous()#把tensor变成在内存中连续分布的形式
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def data_augment(img1, img2, flip=1, ROTATE_90=1, ROTATE_180=1, ROTATE_270=1, add_noise=1):
    n = flip + ROTATE_90 + ROTATE_180 + ROTATE_270 + add_noise
    a = random.random()
    if flip == 1:
        #旋转 转置
        img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
        img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
    if ROTATE_90 == 1:
        # 顺时针旋转90度
        img1 = img1.transpose(Image.ROTATE_90)
        img2 = img2.transpose(Image.ROTATE_90)
    if ROTATE_180 == 1:
        # 顺时针旋转180度
        img1 = img1.transpose(Image.ROTATE_180)
        img2 = img2.transpose(Image.ROTATE_180)
    if ROTATE_270 == 1:
        # 顺时针旋转270度
        img1 = img1.transpose(Image.ROTATE_270)
        img2 = img2.transpose(Image.ROTATE_270)
    if add_noise == 1:
        #？？增加噪声，为什么pass
        pass

## 2020/10/26
import torchvision.transforms as transforms
mean_std = ([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
# 将input变成tensor
input_transform = transforms.Compose([
    transforms.ToTensor(),      ##如果是numpy或者pil image格式，会将[0,255]转为[0,1]，并且(hwc)转为(chw)
    transforms.Normalize(*mean_std)     #[0,1]  ---> 符合imagenet的范围[-2.117,2.248][,][,]
])
# 将label变成tensor
def function_label(x):
    if x > 0: return 1
    else: return 0
class RGBToGray(object):
    #彩色转换为灰度图像
    def __call__(self, mask):
        mask = mask.convert("L")#L为灰色图像
        mask = mask.point(function_label) #点操作 0  1  图
        return mask
class MaskToTensor(object):
    #转换为张量
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()
#串联多个操作
target_transform = transforms.Compose([
    RGBToGray(),
    MaskToTensor()
])
#？？？？
palette = [0, 0, 0, 255, 255, 255]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    #将图像转变为8位彩色图像 fromarray就是实现array到image的转换
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)#自定义一个随机调色板着色
    return new_mask
def fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    #重建矩阵
    hist = np.bincount(
        num_classes * label_true[mask].astype(int) +
        label_pred[mask], minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist
## end 2020/10/26

#训练数据集
class train_dataset(data.Dataset):
    def __init__(self, data_path='', size_w=256, size_h=256, flip=0):
        #初始化
        super(train_dataset, self).__init__()
        self.list = [x for x in os.listdir(data_path + '/src/') if is_image_file(x)]#返回路径下所有文件判断是否是图像
        self.data_path = data_path
        self.size_w = size_w
        self.size_h = size_h
        self.flip = flip

    def __getitem__(self, index):
        #文件路径拼接
        initial_path = os.path.join(self.data_path + '/src/', self.list[index])
        semantic_path = os.path.join(self.data_path + '/label/', self.list[index])
        assert os.path.exists(semantic_path)
        try:
            initial_image = Image.open(initial_path).convert('RGB')
            #点运算 数据增强百分之80 转变为RGB图像
            semantic_image = Image.open(semantic_path).point(lambda i: i * 80).convert('RGB')
        except OSError:
            return None, None, None

        #使用双线性插值调整images的size
        initial_image = initial_image.resize((self.size_w, self.size_h), Image.BILINEAR)
        semantic_image = semantic_image.resize((self.size_w, self.size_h), Image.BILINEAR)

        #旋转
        if self.flip == 1:
            a = random.random()
            if a < 1 / 3:
                initial_image = initial_image.transpose(Image.FLIP_LEFT_RIGHT)
                semantic_image = semantic_image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                if a < 2 / 3:
                    initial_image = initial_image.transpose(Image.ROTATE_90)
                    semantic_image = semantic_image.transpose(Image.ROTATE_90)

        # initial_image = trans_to_tensor(initial_image)  # 0到1之间
        # initial_image = initial_image.mul_(2).add_(-1)  # -1到1之间
        # semantic_image = trans_to_tensor(semantic_image)
        # semantic_image = semantic_image.mul_(2).add_(-1)

        #上面定义的函数
        initial_image = input_transform(initial_image)
        semantic_image = target_transform(semantic_image)

        return initial_image, semantic_image, self.list[index]

    def __len__(self):
        return len(self.list)