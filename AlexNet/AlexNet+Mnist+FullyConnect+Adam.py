# NY 本程序用AlexNet + Fully Connected layer 实现 MNIST识别
# NY by NY Wang 2019.9, Canada
# 注意事项：如果是第一次运行，需要将NY Part3中两个download变量改为True，用来下载MNIST数据，否则False

# =========================================================================
# 本次改动：1采用Adam优化器，2减少了卷积层和全连接层进行尝试（在定义self.classifier中） #
# 效果： 修改后，第二个epoch，loss降到0.01了
# =========================================================================


# from .utils import load_state_dict_from_url
# NY 原本的上一行语句无法执行，用下一行替换.
from torch.hub import load_state_dict_from_url
import torch
import torch.utils.data
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

# NY Part1：定义网络结构
# __all__ = ['AlexNet', 'alexnet']
# NY 因为不使用系统自带AlexNet，所以上一行屏蔽掉

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        # NY num_classes 改为10，即最后输出10个类别
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # NY 由于MNIST尺寸为28X28，所以以下的层数和卷积参数需要修改
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),

            # 以下两行暂时去掉，相当于减少了一个卷积层
            # nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # NY 上面一行是pytorch官方AlexNet版本给的，暂时不用
        self.classifier = nn.Sequential(
            nn.Dropout(),
            # nn.Linear(256 * 3 * 3, 1024),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(1024, 512),
            # 注意，删除上四行，加入下面一行,相当于去掉了一个全连接层
            nn.Linear(256 * 3 * 3, 512),  #这是新增行
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        # NY 上一行暂时不用
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


# NY Part2：定义数据变换
transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomGrayscale(),
    transforms.ToTensor(),
])

transform2 = transforms.Compose([
    transforms.ToTensor()
])

# NY Part3：数据的下载和装入设置
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform1)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True,
                                          num_workers=0)  # windows下num_workers设置为0，不然有bug
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# NY Part4：产生一个网络
net = alexnet()

# NY Part5：定义损失函数
loss_criterion = nn.CrossEntropyLoss()

# NY Part6：定义优化器
# optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
# NY 本次试用adam，而不是用SGD
optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# NY Part7: 定义device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# NY Part8: 开始训练
print("Start Training!")
start_time=time.time()

num_epochs = 20  # 训练次数
for epoch in range(num_epochs):
    running_loss = 0
    batch_size = 100

    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print('[%d, %5d] loss:%.4f' % (epoch + 1, (i + 1) * 100, loss.item()))

end_time=time.time()
print("Finished Traning")

# NY Part9:计算训练时间
train_time=end_time-start_time
print("time consumed:{:.0f}m {:.0f}s".format(train_time//60, train_time%60))

# NY Part10:保存训练模型
torch.save(net, 'MNIST.pkl')
net = torch.load('MNIST.pkl')

# NY Part11:开始识别测试
with torch.no_grad():
    # 在接下来的代码中，所有Tensor的requires_grad都会被设置为False
    correct = 0
    total = 0

    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        out = net(images)
        _, predicted = torch.max(out.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images:{}%'.format(100 * correct / total))  # 输出识别准确率
