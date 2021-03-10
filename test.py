import argparse

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision
import numpy as np
import time
from models import *

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('-weights', type=str, required=True, help='adress to model file')
args = parser.parse_args()

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

testset = torchvision.datasets.CIFAR10(
    root='/media/mdisk/yushu/dataset/CIFAR_10', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=True, num_workers=2)

net = DLA().to(device)
checkpoint = torch.load(args.weights)
net.load_state_dict(({k.replace('module.',''):v for k,v in checkpoint['net'].items()}))

net.eval()

correct_1 = 0.0
correct_5 = 0.0
total = 0
# error = 0

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

for n_iter, (image, label) in enumerate(testloader):
    #print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

    image = image.cuda()
    label = label.cuda()


    output = net(image)
    _, pred = output.topk(5, 1, largest=True, sorted=True)

    label = label.view(label.size(0), -1).expand_as(pred)
    correct = pred.eq(label).float()

    # compute top 5
    correct_5 += correct[:, :5].sum()

    # compute top1
    correct_1 += correct[:, :1].sum()
    # if correct[:, :1].sum() == 0:
    #     img = torchvision.utils.make_grid(image.cpu()).numpy()
    #     plt.imshow(np.transpose(img, (1, 2, 0)))
    #     print('label'+str(label)+'pred'+str(pred))
    #     plt.show()
    # if correct[:, :1].sum() == 0:
    #     img = torchvision.utils.make_grid(image.cpu()).numpy()
    #     plt.imshow(np.transpose(img, (1, 2, 0)))
    #     plt.title('label:'+str(classes[label.cpu().numpy()[0][0]])+'\tpred:'+str(classes[pred.cpu().numpy()[0][0]]))
    #     plt.show()
    #     error += 1
    #     if error== 10 :
    #         break

print()
print("Top 1 err: ", 1 - correct_1 / len(testloader.dataset))
print("Top 5 err: ", 1 - correct_5 / len(testloader.dataset))
print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))

