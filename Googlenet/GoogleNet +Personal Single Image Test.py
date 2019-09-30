# 本程序已经测试运行成功，用训练好的模型，可以预测电脑中的单张图片
# 用法：在第15行input_image = Image.open后面，参数改为图片名称
# NY 2019， Canada

import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

model = models.googlenet(pretrained=True, progress=True)
model.eval()
model.load_state_dict(torch.load('model_para.pth')) # 加载训练好的模型文件的参数

# 读入图片，并进行格式转换：
input_image = Image.open('./test_image/yellow-river.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
# 上面经过转换后输出的input格式是[C,H,W],网络输入还需要增加一维批量大小Batch
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
# 上面是给数据增加一维，输出的img格式为[1,C,H,W]

# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)   # 将图片输入网络得到输出

# print(output[0])  暂不输出1000个值
print(torch.nn.functional.softmax(output[0], dim=0))  # 计算softmax，即该图片属于各类的概率
print(torch.max(output[0],0))  # 找到最大概率对应的索引号，该图片即为该索引号对应的类别