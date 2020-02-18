import  torch
import  os, glob
import  random, csv
import  numpy as np
from    torch.utils.data import Dataset, DataLoader
from    torchvision import transforms
from    PIL import Image


class Cifar(Dataset):

    def __init__(self, root, resize, mode):
        super(Cifar, self).__init__()
        self.root = root
        self.resize = resize
        # image, label
        if mode =='train':
            self.images, self.labels = self.load_csv('train.csv')
            self.images = self.images[int(0.5 * len(self.images)):]
            self.labels = self.labels[int(0.5 * len(self.labels)):]
        if mode =='val':
            self.images, self.labels = self.load_csv('test.csv')
            self.images = self.images[int(0.5 * len(self.images)):]
            self.labels = self.labels[int(0.5 * len(self.labels)):]
        elif mode == 'test':
            self.images, self.labels = self.load_csv('test.csv')
            self.images = self.images[:int(0.5 * len(self.images))]
            self.labels = self.labels[:int(0.5 * len(self.labels))]


    def load_csv(self, filename):
        # 如果 filename这个文件不存在，就开始创建添加一个
        if not os.path.exists(os.path.join(self.root, filename)):
            images = []       # 建立一个images的列表，里面的元素是每个图片的路径
            images += glob.glob(os.path.join(self.root, '*.jpg'))
            print(len(images))
            random.shuffle(images)
            # 创建一个命名为“filename”的csv文件
            with open(os.path.join(self.root, filename), mode='w', newline='') as f:
                writer = csv.writer(f)
                for img in images: # 'pokemon\\bulbasaur\\00000000.png'
                    name = img.split(os.sep)[-1]   # name = bulbasaur
                    label = name[0]    # 获取name对应的标签
                    writer.writerow([img, label])   # 将图片的文件路径和对应的类型标签存贮到csv文件中
                print('writen into csv file:', filename)

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                # 'pokemon\\bulbasaur\\00000000.png', 0, [att]
                img = row[0]
                label = row[1]
                label = int(label)

                # att = np.asarray(att)              # 先将列表转换为矩阵形式，矩阵内元素这时还是str类型
                # att = att.astype(np.float)        # 将str转换为float才能再变为tensor
                images.append(img)
                labels.append(label)

        assert len(images) == len(labels)
        state = np.random.get_state()
        np.random.shuffle(images)

        np.random.set_state(state)
        np.random.shuffle(labels)

        return images, labels


    def __len__(self):

        return len(self.images)


    def denormalize(self, x_hat):

        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        mean = 0.47328657379180195
        std = 0.2098421003177017

        # x_hat = (x-mean)/std
        # x = x_hat*std = mean
        # x: [c, h, w]
        # mean: [3] => [3, 1, 1]
        # mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        # std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
        # print(mean.shape, std.shape)
        x = x_hat * std + mean

        return x


    def __getitem__(self, idx):

        img, label = self.images[idx], self.labels[idx]

        # tf = transforms.Compose([
        #     lambda x:Image.open(x).convert('RGB'), # string path= > image data
        #     transforms.Resize((int(self.resize*1.25), int(self.resize*1.25))),
        #     transforms.RandomRotation(15),
        #     transforms.CenterCrop(self.resize),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])
        # ])

        tf = transforms.Compose([
            lambda x: Image.open(x),  # string path= > image data
            transforms.Resize((int(self.resize * 1.25), int(self.resize * 1.25))),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize((0.47328657379180195,), (0.2098421003177017,))
        ])

        img = tf(img)
        label = torch.tensor(label)
        # att = att.unsqueeze(0)


        return img, label





def main():

    import  visdom
    import  time
    import  torchvision

    viz = visdom.Visdom()

    # tf = transforms.Compose([
    #                 transforms.Resize((64,64)),
    #                 transforms.ToTensor(),
    # ])
    # db = torchvision.datasets.ImageFolder(root='pokemon', transform=tf)
    # loader = DataLoader(db, batch_size=32, shuffle=True)
    #
    # print(db.class_to_idx)
    #
    # for x,y in loader:
    #     viz.images(x, nrow=8, win='batch', opts=dict(title='batch'))
    #     viz.text(str(y.numpy()), win='label', opts=dict(title='batch-y'))
    #
    #     time.sleep(10)


    db = Cifar('D:/phython_work/cifar/cifar-10-batches-py/train_canny', 224,'train')
    print(type(db))

    x, y = next(iter(db))
    print('sample:', x.shape, y.shape, x, y)

    viz.image(db.denormalize(x), win='sample_x', opts=dict(title='sample_x'))

    loader = DataLoader(db, batch_size=30,  num_workers=4)

    for x, y in loader:
        viz.images(db.denormalize(x), nrow=6, win='batch', opts=dict(title='batch'))
        viz.text(str(y.numpy()), win='label', opts=dict(title='batch-label'))

        time.sleep(10)



if __name__ == '__main__':
    main()