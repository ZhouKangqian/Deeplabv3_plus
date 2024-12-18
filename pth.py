import torch

pthfile = r'E:\DeepLabv3+\bubbliiiing\deeplabv3-plus-pytorch-main\model_data\deeplab_mobilenetv2.pth'
net = torch.load(pthfile)
print(net)
