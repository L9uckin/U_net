# coding:utf8

import os
import cv2
import numpy as np
import random
from tqdm import tqdm
import operator


def is_image_file(filename):  #判断是否事图像函数
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".tif", ".tiff"])


def is_image_damaged(cvimage):
    white_pixel_count = 0
    height, weight, channel = cvimage.shape
    # for row in list(range(0,height)):
    #     for col in list(range(0,weight)):
    #         if cvimage[row][col][2] == 255:
    #             white_pixel_count += 1
    #             if white_pixel_count > 0.2*height*weight:
    #                 return True
    # return False
    one_channel = np.sum(cvimage, axis=2)#axis=2相当于Z轴上的数之和
    white_pixel_count = len(one_channel[one_channel == 255 * 3])  # 计算白色像素的数量
    if white_pixel_count > 0.08 * height * weight: #？？？为什么0.08
        return True
    return False


def gamma_transform(img, gamma):
    #映射表必须为0 - 255
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]
    #numpy数组默认数据类型为int32，需要将数据类型转换成opencv图像适合使用的无符号8位整型uint8，
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
    return cv2.LUT(img, gamma_table) #映射


def random_gamma_transform(img, gamma_vari):
    log_gamma_vari = np.log(gamma_vari)#计算以e为底的对数
    #随机数
    alpha = np.random.uniform(-log_gamma_vari, log_gamma_vari)
    #返回e的幂次方
    gamma = np.exp(alpha)
    return gamma_transform(img, gamma)


def rotate(xb, yb, angle):
    M_rotate = cv2.getRotationMatrix2D((img_w / 2, img_h / 2), angle, 1) #获得图像绕着中心点的旋转矩阵
    xb = cv2.warpAffine(xb, M_rotate, (img_w, img_h)) #仿射变换 平移 输出尺寸不变
    yb = cv2.warpAffine(yb, M_rotate, (img_w, img_h))
    return xb, yb


def blur(img):
    img = cv2.blur(img, (3, 3));#  均值滤波  获取内核区域下所有像素的平均值
    return img


def add_noise(img):
    for i in range(200):  # 添加点噪声
        temp_x = np.random.randint(0, img.shape[0])
        temp_y = np.random.randint(0, img.shape[1])
        img[temp_x][temp_y] = 255
    return img

#数据增强规则
def data_augment(xb, yb):
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 90)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 180)
    if np.random.random() < 0.25:
        xb, yb = rotate(xb, yb, 270)
    if np.random.random() < 0.25:
        xb = cv2.flip(xb, 1)  # flipcode > 0：Flip along the y axis
        yb = cv2.flip(yb, 1)

    if np.random.random() < 0.25:
        xb = random_gamma_transform(xb, 1.0)

    if np.random.random() < 0.25:
        xb = blur(xb)

    if np.random.random() < 0.2:
        xb = add_noise(xb)

    return xb, yb

# 参数不明白
def creat_dataset(image_sets, img_w, img_h, image_num=5000, mode='normal'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        src_img = cv2.imread('./Training Set/Input images/' + image_sets[i])  # 3 channels
        # label_img = cv2.imread('./Training Set/Target maps/' + image_sets[i].replace('tiff','tif'))  # 3 channels
        label_img_gray = cv2.imread('./Training Set/Target maps/' + image_sets[i].replace('tiff', 'tif'),
                                    cv2.IMREAD_GRAYSCALE) #读入灰度图片 # single channel
        #获取图片的长宽
        X_height, X_width, _ = src_img.shape
        while count < image_each:
            random_width = random.randint(0, X_width - img_w - 1)
            random_height = random.randint(0, X_height - img_h - 1)
            src_roi = src_img[random_height: random_height + img_h, random_width: random_width + img_w, :]
            # if is_image_damaged(src_roi):
            #     continue
            # label_roi = label_img[random_height: random_height + img_h, random_width: random_width + img_w]
            label_roi_gray = label_img_gray[random_height: random_height + img_h, random_width: random_width + img_w]
            if mode == 'augment':
                src_roi, label_roi_gray = data_augment(src_roi, label_roi_gray)

            # cv2.imwrite(('./Training Set/visualize/%d.png' % g_count),label_roi)
            cv2.imwrite(('./Training Set/src/%d.png' % g_count), src_roi)
            cv2.imwrite(('./Training Set/label/%d.png' % g_count), label_roi_gray)
            count += 1
            g_count += 1


if __name__ == '__main__':
    img_w = 256
    img_h = 256
    src_data_path = './Training Set/Input images/'
    label_data_path = './Training Set/Target maps/'

    # data_path = './Training Set/'
    # image_sets = [x for x in os.listdir(data_path + '/Input images/') if is_image_file(x)]
    # print(len(image_sets))
    image_sets2 = [x for x in os.listdir(src_data_path) if is_image_file(x)]
    # print(len(image_sets2))
    creat_dataset(image_sets=image_sets2, img_w=img_w, img_h=img_h)