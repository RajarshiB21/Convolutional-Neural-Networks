import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample =None, stride=1):
        super(block, self).__init__()
        #identity_downsample is a conv layer which we might need to do if we have hanged the input size or the number of channels

        self.expansion = 4#since the number of channels after a block is always four times after it entered
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):

    #Layers is a list telling us how many times we want to use the block
    #For resnet 50 it is [3, 4, 6, 3]


    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding= 1)

        #Resnet layer
        self.layer1 = self.make_layer(block, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self.make_layer(block, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self.make_layer(block, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self.make_layer(block, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


    def make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        #If we have changed the number of channels or changed the size of the image then we use identity downsample
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels,intermediate_channels * 4,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )
        self.in_channels =intermediate_channels*4#64*4 = 256

        for i in range(num_residual_blocks-1):
            #-1 since we already did the first block above
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)

def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(block, [3,4,6,3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(block, [3,4,23,3], img_channels, num_classes)

def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(block, [3,8,36,3], img_channels, num_classes)


def test():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = ResNet50()
    x = torch.randn(2, 3, 224, 224)
    y = net(x).to(device)
    print(y.shape)


if __name__ == '__main__':
    test()






