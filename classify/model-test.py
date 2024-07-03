import torch
from torchvision import models
import res18_senet
# import res18
resume = 'E:/PycharmProjects/grt - SENET/training/detector/results/res18_senet-20181113-183106/100.ckpt'
checkpoint = torch.load(resume)

model = res18_senet.Net()
checkpoint = torch.load('E:/PycharmProjects/grt - SENET/training/detector/results/res18_senet-20181113-183106/100.ckpt')
model.load_state_dict(checkpoint['state_dict'])

# resnet = models.resnet18(pretrained=True)

print(list(model.children())[:-15])

