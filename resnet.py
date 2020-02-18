###author:xiaoheimiao
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#定义make_layer
def make_layer(in_channel, out_channel, block_num, stride=1):
    shortcut = nn.Sequential(
        nn.Conv2d(in_channel, out_channel, 1, stride),
        nn.BatchNorm2d(out_channel))
    layers = list()
    layers.append(ResBlock(in_channel, out_channel, stride, shortcut))

    for i in range(1, block_num):
        layers.append(ResBlock(out_channel, out_channel))
    return nn.Sequential(*layers)


#定义ResBlock，见Resnet Learning图
class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, shortcut=None):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)


# 堆叠Resnet，见上表所示结构
class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64),       # 3表示输入图片的通道数
            nn.ReLU(True), nn.MaxPool2d(3, 2, 1))
        self.layer1 = make_layer(64, 64, 2)
        self.layer2 = make_layer(64, 128, 2, stride=2)
        self.layer3 = make_layer(128, 256, 2, stride=2)
        self.layer4 = make_layer(256, 512, 2, stride=2)
        self.avg = nn.AvgPool2d(7)
        self.classifier = nn.Sequential(nn.Linear(512, 10))

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out


#训练函数
def net_train():
    net.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 将输入传入GPU
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 将梯度置零
        optimizer.zero_grad()

        # 前向传播-计算误差-反向传播-优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 计算误差并显示
        running_loss += loss.item()
        if i % 127 == 0:  # print every mini-batches
            print(
                '[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 128))
            running_loss = 0.0

    print('Training Epoch Finished')


#测试函数
def net_test():
    correct = 0
    total = 0
    # 关闭梯度
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' %
          (100 * correct / total))
    return


#数据集函数
def net_dataloader(root, train_transform, test_transform):
    trainset = torchvision.datasets.CIFAR10(
        root, train=True, transform=train_transform, download=False)
    testset = torchvision.datasets.CIFAR10(
        root, train=False, transform=test_transform, download=False)
    trainloader = DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4)
    testloader = DataLoader(
        testset, batch_size=16, shuffle=False, num_workers=4)
    print('Initializing Dataset...')
    return trainloader, testloader


# main
if __name__ == "__main__":
    # 创建实例并送入GPU
    net = Resnet().to(device)
    # 选择误差
    criterion = nn.CrossEntropyLoss()
    # 选择优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    # 数据位置
    root = './pydata/data/'
    # 数据处理
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 创建数据loader
    trainloader, testloader = net_dataloader(root, train_transform,
                                             test_transform)
    # run
    n_epoch = 5  #改变epoch
    for epoch in range(n_epoch):
        print('Training...')
        net_train()  #每个epoch训练一次，测试一次
        print('Testing...')
        net_test()