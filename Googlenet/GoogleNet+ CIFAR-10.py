# NY 本程序用GoogleNet 实现 Cifar10的训练和识别
# NY by NY Wang 2019.9, Canada
# 注意事项1：注意是否使用googlenet的两个辅助分支。具体看52行注释。
# 注意事项2：如果要修改batchsize，修改的地方有四处，lines：26,28,47,60。

import torch
import torch.utils.data
import torchvision
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms as transforms
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# NY Part1：生成一个googlenet实例
net = models.googlenet(pretrained=False)

# NY Part2：定义数据变换
transform1 = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.ToTensor()])
transform2 = transforms.Compose([transforms.ToTensor()])

# NY Part3：数据的下载和装入设置
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform1)
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=100, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform2)
testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=100, shuffle=False)

# NY Part4：定义损失函数
loss_criterion = nn.CrossEntropyLoss()

# NY Part5：定义优化器
optimizer = optim.Adam(net.parameters(),lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# NY Part6: 定义device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# NY Part7: 开始训练
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
        # 下面这一句非常关键，因为我们没有使用辅助分支。参考资料：https://discuss.pytorch.org/t/question-about-googlenet/44896/5
        outputs = outputs.logits
        loss = loss_criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print('[%d, %5d] loss:%.4f' % (epoch + 1, (i + 1) * 100, loss.item()))

end_time=time.time()
print("Finished Traning")

# NY Part8:计算训练时间
train_time=end_time-start_time
print("time consumed:{:.0f}m {:.0f}s".format(train_time//60, train_time%60))

# NY Part9:保存训练模型
torch.save(net, 'CIFAR-10.pkl')

# NY Part10:导入已保存的训练模型
net = torch.load('CIFAR-10.pkl')

# NY Part11:开始识别测试
net.eval()    #调整为测试模式
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
