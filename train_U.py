from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torch.autograd import Variable
import time
import numpy as np
from numpy import *
from data_loader.dataset import train_dataset, colorize_mask, fast_hist
from models.u_net import UNet
from models.seg_net import Segnet
from tensorboardX import SummaryWriter
from PIL import Image
import numpy as np

#创建一个ArgumentParaser对象
parser = argparse.ArgumentParser(description='Training a UNet model')
#添加参数
#‘’命名 类型int 默认值4 每次训练多少张
parser.add_argument('--batch_size', type=int, default=4, help='equivalent to instance normalization with batch_size=1')
#输入数据通道为3
parser.add_argument('--input_nc', type=int, default=3)
#输出通道数为2
parser.add_argument('--output_nc', type=int, default=2, help='equivalent to num class')
#全部数据被训练的次数
parser.add_argument('--niter', type=int, default=10, help='number of epochs to train for')
#学习率设置为0.0001
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
##学习率自适应算法采用adam
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
#能够调用cuda
parser.add_argument('--cuda', type=bool, default=True, help='enables cuda. default=True')
##指定生成随机数seed
parser.add_argument('--manual_seed', type=int, help='manual seed')
#加载数据时需要使用的cpu线程数

parser.add_argument('--num_workers', type=int, default=2, help='how many threads of cpu to use while loading data')
#将图像的宽度按此大小缩放
parser.add_argument('--size_w', type=int, default=256, help='scale image to this size')
#将图像的高度按此大小缩放
parser.add_argument('--size_h', type=int, default=256, help='scale image to this size')

#是否翻转图像
parser.add_argument('--flip', type=int, default=0, help='1 for flipping image randomly, 0 for not')
##预训练网络的路径
parser.add_argument('--net', type=str, default='', help='path to pre-trained network')
#图像数据训练路径
parser.add_argument('--data_path', default='./data/train', help='path to training images')
#模型输出图片以及检查点保存路径
parser.add_argument('--outf', default='./checkpoint/Unet', help='folder to output images and model checkpoints')

# 保存路径
parser.add_argument('--save_epoch', default=1, help='path to save model')


# 测试的数据 多少次保存一次
parser.add_argument('--test_step', default=300, help='path to val images')

# 日志多少次保存一次
parser.add_argument('--log_step', default=1, help='path to val images')

#GPU数量
parser.add_argument('--num_GPU', default=1, help='number of GPU')
#实例 增加的属性直接使用即可
opt = parser.parse_args()
try:
    #创建目录
    os.makedirs(opt.outf)
    os.makedirs(opt.outf + '/model/')
except OSError:
    pass
#如果没有设置随机种子，返回一个从1-10000之间的随机数
if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)

#提升计算速度
cudnn.benchmark = True

print(opt)
print("Random Seed: ", opt.manual_seed)

##获取训练数据
train_datatset_ = train_dataset(opt.data_path, opt.size_w, opt.size_h, opt.flip)

#把训练数据分成多个小组，数据初始化
train_loader = torch.utils.data.DataLoader(dataset=train_datatset_, batch_size=opt.batch_size, shuffle=True,
                                           num_workers=opt.num_workers)

#参数初始化
def weights_init(m):
# m作为一个形参，原则上可以传递很多的内容，为了实现多实参传递，每一个模型要给出自己名字 所以这句话就是返回m的名字。
    class_name = m.__class__.__name__
#find()函数，实现查找classname中是否含有conv字符，没有返回-1；有返回0.
    if class_name.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)#m.weight.data表示需要初始化的权重。.normal_()表示随机初始化采用正态分布，均值为0.0，标准差为0.02.

        m.bias.data.fill_(0)#表示将偏差定义为0

    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

#Uet模型
net = UNet(opt.input_nc, opt.output_nc)

#如果有预训练网络，加载
if opt.net != '':
    net.load_state_dict(torch.load(opt.netG))
else:
    net.apply(weights_init) #apply函数会递归地搜索网络内的所有module并把参数表示的函数应用到所有的module上。

#用cuda进行训练
if opt.cuda:
    net.cuda()
##用多个GPU进行加速训练
if opt.num_GPU > 1:
    net = nn.DataParallel(net)

###########   LOSS & OPTIMIZER   ##########
# criterion = nn.BCELoss()

#损失函数
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)#返回N个样本的loss的平均 忽略255标签loss

#实现Adam算法
#net.parameters()获取网络中的所有参数，lr学习率之前自定义，beat用于计算梯度以及梯度平方的运行平均值的系数
optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

###########   GLOBAL VARIABLES   ###########
# 初始化数据图像
initial_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)
# 数据上的标签
semantic_image = torch.FloatTensor(opt.batch_size, opt.input_nc, opt.size_w, opt.size_h)

#自动求导
initial_image = Variable(initial_image)
semantic_image = Variable(semantic_image)

#  调用cuda？
if opt.cuda:
    initial_image = initial_image.cuda()
    semantic_image = semantic_image.cuda()
    criterion = criterion.cuda()

if __name__ == '__main__':
    #打开文件
    log = open('./checkpoint/Unet/train_Unet_log.txt', 'w')
    #计算运行时间
    start = time.time()
    #网络训练
    net.train()
    #返回一个0矩阵
    hist = np.zeros((opt.output_nc, opt.output_nc))
    #迭代训练
    for epoch in range(1, opt.niter + 1):
        #迭代初试数据
        loader = iter(train_loader)
        #从0开始按照batchsize步长增加，直到数据结束。
        for i in range(0, train_datatset_.__len__(), opt.batch_size):
            #每隔一个batchsize 进行一次next
            initial_image_, semantic_image_, name = loader.next()# ？下一条记录
            # 为什么要copy 虽然一样
            initial_image.resize_(initial_image_.size()).copy_(initial_image_)#调整图像大小

            semantic_image.resize_(semantic_image_.size()).copy_(semantic_image_)

            semantic_image_pred = net(initial_image) #分割前？


            # initial_image = initial_image.view(-1)

            # semantic_image_pred = semantic_image_pred.view(-1)


            ### loss ###
            # from IPython import embed;embed()
            #判断是否相等
            assert semantic_image_pred.size()[2:] == semantic_image.size()[1:]
            loss = criterion(semantic_image_pred, semantic_image.long())# 计算损失函数

            optimizer.zero_grad()#梯度初始化为零，把loss关于weight的导数变成0
            loss.backward()#反向传播求梯度
            optimizer.step()#更新所有参数

            ### evaluate ###
            #squeeze(1)n行1列向量降成一维  squeeze(0) 1*n向量降成1维
            predictions = semantic_image_pred.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()

            gts = semantic_image.data[:].squeeze_(0).cpu().numpy()

            hist += fast_hist(label_pred=predictions.flatten(), label_true=gts.flatten(),
                              num_classes=opt.output_nc)

            #计算准确率
            train_acc = np.diag(hist).sum() / hist.sum()

            ########### Logging ##########

            # 自定义那块log_step 和 test_step jasldfjklajsdhgkjhdakf;lakdjf;lsh
            if i % opt.log_step == 0:
                #每隔log-step次写一次日志
                print('[%d/%d][%d/%d] Loss: %.4f TrainAcc: %.4f' %
                      (epoch, opt.niter, i, len(train_loader) * opt.batch_size, loss.item(), train_acc))
                log.write('[%d/%d][%d/%d] Loss: %.4f TrainAcc: %.4f' %
                          (epoch, opt.niter, i, len(train_loader) * opt.batch_size, loss.item(), train_acc))
            if i % opt.test_step == 0:
                #每隔test-step次保存一次测试数据
                gt = semantic_image[0].cpu().numpy().astype(np.uint8)
                gt_color = colorize_mask(gt)
                predictions = semantic_image_pred.data.max(1)[1].squeeze_(1).cpu().numpy()
                prediction = predictions[0]
                predictions_color = colorize_mask(prediction)
                #保存图片
                width, height = opt.size_w, opt.size_h
                save_image = Image.new('RGB', (width * 2, height))
                save_image.paste(gt_color, box=(0 * width, 0 * height))
                save_image.paste(predictions_color, box=(1 * width, 0 * height))
                save_image.save(opt.outf + '/epoch_%03d_%03d_gt_pred.png' % (epoch, i))
        #结束内循环
        if epoch % opt.save_epoch == 0:
            torch.save(net.state_dict(), '%s/model/netG_%s.pth' % (opt.outf, str(epoch)))

    end = time.time()
    torch.save(net.state_dict(), '%s/model/netG_final.pth' % opt.outf)

    print('Program processed ', end - start, 's, ', (end - start) / 60, 'min, ', (end - start) / 3600, 'h')
    log.close()
