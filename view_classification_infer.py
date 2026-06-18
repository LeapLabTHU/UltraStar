import os
import torch
import torch.nn as nn
import random
import cv2
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 2048),
            nn.Dropout(0.4),
            nn.GELU(),
            nn.Linear(2048, 512),
            nn.Dropout(0.4),
            nn.GELU(),
            nn.Linear(512, 10)
        )


    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1) 
        x = self.fc(x)

        return x


class Data_aug():
    def __init__(self, img_size=(224, 224, 3)):
        self.mask = np.zeros(img_size, dtype=np.uint8)

    def __call__(self, img):
        img = np.array(img)
        cv2.ellipse(self.mask, center=(112, 8), axes=(208, 208), angle=0, startAngle=55,
                    endAngle=125, color=(1, 1, 1), thickness=-1)
        img = img * self.mask
        img = Image.fromarray(np.array(img))
        return img

def inference(data_path, model, transform):
    for exam in tqdm(sorted(os.listdir(data_path)), total=len(os.listdir(data_path))):
        exam_path = os.path.join(data_path, exam)
        for im_name in sorted(os.listdir(exam_path)):
            if not im_name.endswith('.jpg'):
                continue
            img_path = os.path.join(exam_path, im_name)
            base_path = exam_path.replace('image', 'quality')
            os.makedirs(base_path, exist_ok=True)
            img = Image.open(img_path)
            img = transform(img)
            img = img.unsqueeze(0).cuda()
            with torch.no_grad():
                ori_output = model(img)
            out = torch.sigmoid(ori_output / 4).tolist()[0]
            result_path = os.path.join(base_path, im_name.replace('.jpg', '.txt'))
            with open(result_path, 'w') as f:
                f.write(str(out))

if __name__ == '__main__':
    test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            Data_aug(img_size=(224, 224, 3)),
            transforms.ToTensor(),
            transforms.Normalize([0.193, 0.193, 0.193], [0.224, 0.224, 0.224])
        ])

    model = ResNet34()
    model.load_state_dict(torch.load('qs_model.pth', map_location='cpu'))
    model = model.cuda()
    model.eval()
    data_path = 'train/images'
    inference(data_path, model, test_transform)