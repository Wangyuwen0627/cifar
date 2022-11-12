from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
from ResNet18.net import ResNet18
from DenseNet.net import DenseNet
import time
import datetime

def format_time(time):
    elapsed_rounded = int(round((time)))
    # 格式化为 hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

def drawAcc(train_acc_list, test_acc_list):
    fig, ax = plt.subplots(figsize=(9, 3), dpi=200, facecolor = "#EFE9E6")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(ls="--", lw=0.5, color="#4E616C")
    X = pd.Series(range(len(train_acc_list)))
    # X = np.linspace(0, len(train_acc_list), 1024)
    print(X)
    print(train_acc_list)
    print(test_acc_list)
    ax.plot(X, train_acc_list, marker="o", mfc="white", ms=2)
    ax.plot(X, test_acc_list, marker="o", mfc="white", ms=2)
    ax.spines["bottom"].set_edgecolor("#4E616C")
    plt.savefig('/home/wyf/cifar/results/resnet18/resnet18_acc_bs_128.png')  # 保存图片
    plt.show()


def drawLoss(train_loss_list):
    fig, ax = plt.subplots(figsize=(9, 3), dpi=200, facecolor = "#EFE9E6")
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(ls="--", lw=0.5, color="#4E616C")
    X = pd.Series(range(len(train_loss_list)))
    ax.plot(X, train_loss_list, marker="o", mfc = "white", ms = 1)
    xmajor = ticker.MultipleLocator(10)
    xminor = ticker.MultipleLocator(5)
    # xticks_ = ax.xaxis.set_ticklabels([x for x in range(0, len(X) + 3, 4)])
    # This last line outputs
    # [-1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35]
    # and we mark the tickers every two positions.
    # ax.xaxis.set_tick_params(length=5, color="#4E616C", labelcolor="#4E616C", labelsize=6)
    # ax.yaxis.set_tick_params(length=5, color="#4E616C", labelcolor="#4E616C", labelsize=6)
    ax.xaxis.set_minor_locator(xminor)
    ax.xaxis.set_major_locator(xmajor)
    ax.tick_params(axis="x", direction="in", color="#4E616C", labelcolor="#4E616C", labelsize=6, which="minor",
                   length=4)
    ax.tick_params(axis="x", direction="out", color="#4E616C", labelcolor="#4E616C", labelsize=6, which="major",
                   length=5)
    ax.spines["bottom"].set_edgecolor("#4E616C")
    plt.savefig('/home/wyf/cifar/results/resnet18/resnet18_loss_bs_128.png')  # 保存图片
    plt.show()

# 定义是否使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./weight/resnet18', help='folder to output images and weight checkpoints')  # 输出结果保存路径
parser.add_argument('--net', default='./weight/resnet18/Resnet18.pth', help="path to net (to continue training)")  # 恢复训练时的模型路径
args = parser.parse_args()

# 超参数设置
EPOCH = 150  # 遍历数据集次数
pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 128  # 批处理尺寸(batch_size)
LR = 0.01  # 学习率

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),  # R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)  # 训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True,
                                          num_workers=2)  # 生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
# net = ResNet18().to(device)
net = DenseNet(growthRate=12, depth=100, reduction=0.5,
               bottleneck=True, nClasses=10).to(device)

# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  # 损失函数为交叉熵，多用于多分类问题
# optimizer = optim.Adam(net.parameters(), lr=LR, momentum=0.9,
#                       weight_decay=5e-4)  # 优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9,
                      weight_decay=5e-4)
# 训练
if __name__ == "__main__":
    best_acc = 85  # 2 初始化best test accuracy
    print("Start Training, Resnet-18!")  # 定义遍历数据集的次数
    train_acc_list = []
    train_loss_list = []
    test_acc_list = []
    t0 = time.time()
    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        net.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(trainloader, 0):
            # 准备数据
            length = len(trainloader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # 每训练1个batch打印一次loss和准确率
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                    % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
        print('训练分类准确率为：%.3f%%' % (100 * correct / total))
        train_acc = 100. * correct / total
        train_acc_list.append(train_acc.cpu())
        train_loss = sum_loss / len(trainloader)
        train_loss_list.append(train_loss)
        print("train_loss", train_loss)
        # 每训练完一个epoch测试一下准确率
        print("Waiting Test!")
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                net.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类 (outputs.data的索引号)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('测试分类准确率为：%.3f%%' % (100 * correct / total))
            acc = 100. * correct / total
            test_acc_list.append(acc.cpu())
            # 将每次测试结果实时写入acc.txt文件中
            print("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
            # 记录最佳测试分类准确率并写入best_acc.txt文件中
            if acc > best_acc:
                print("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                print('Saving weight......')
                torch.save(net.state_dict(), '%s/resnet18_best_bs_128.pth' % (args.outf))
                best_acc = acc
    t1 = time.time()
    training_time = t1 - t0
    drawLoss(train_loss_list)
    drawAcc(train_acc_list, test_acc_list)
    print("Training Finished, TotalEPOCH=%d" % EPOCH)
    print("best test_acc:", max(test_acc_list))
    training_time = format_time(training_time)
    print("training time:", training_time)