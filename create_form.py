import  os, glob
import  random, csv
from GLCM_2 import convelution
from Canny import canny_extrction
import skimage.io
import numpy as np
from PIL import Image

def create_csv(root,filename):
    images = []
    images += glob.glob(os.path.join(root,'*jpg'))
    print(len(images))
    print(images)

    with open(os.path.join(root,filename), mode='w',newline='') as f:
        writer = csv.writer(f)
        for img in images:
            writer.writerow([img])
        print('writen into csv file:',filename)

    return (print('finished'))

def main():
    filename = 'train.csv'
    document = 'train_canny'
    root = 'D:/phython_work/cifar/cifar-10-batches-py/train'
    if not os.path.exists(os.path.join(root, filename)):
        create_csv(root,filename)

    if not os.path.exists(root.replace('train',document)):

        os.mkdir(root.replace('train',document))

        with open(os.path.join(root,filename)) as f:
            reader = csv.reader(f)
            for path in reader:
                path1 = "".join(path[0])    # 列表信息转字符串
                path1.replace('\\','/')
                # result = convelution(path1,4)
                result = canny_extrction(path1)
                savename = path1.replace('train',document)
                skimage.io.imsave(savename, result)

if __name__ == '__main__':
    main()
