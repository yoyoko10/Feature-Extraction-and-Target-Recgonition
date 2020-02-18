import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_ubyte,transform


def canny_extrction(path):

    img = cv2.imread(path, 1)
    # cv2.imshow('original',img)
    # flag=-1时，8位深度，原通道
    # flag=0，8位深度，1通道
    # flag=1,   8位深度  ，3通道
    # flag=2，原深度，1通道
    # flag=3,  原深度，3通道
    # flag=4，8位深度 ，3通道
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = transform.resize(img, [224,224])
    # cv2.imshow('original',img)
    # 图像降噪
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = img*255
    img = img.astype(np.uint8)
    # print(type(img),img.shape,np.max(img),type(img[1][1]))
    # Canny边缘检测
    canny = cv2.Canny(img, 15, 45, 3)
    # print(canny.shape,np.max(canny))

    return canny

def main():
    path = 'D:\phython_work\cifar\cifar-10-batches-py/train/8_18660.jpg'
    result = canny_extrction(path)

    cv2.namedWindow('canny',0)
    cv2.imshow('canny',result)
    cv2.waitKey()


if __name__ == '__main__':
    main()








