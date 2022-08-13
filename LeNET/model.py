import torch
import torch.nn as nn

#LENET Architecture
# 1x32x32 input > 5x5, s-1, p-0 -> avg pool s-2, p=0 -> 5x5, s=1, p=0 -> avg pool s=2, p=0
# -> Conv 5x5 to 120 channels x Linear 120 to 84 -> Linear 84 x Linear 10


class LeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(LeNet, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0), #Num_examples x 120 x 1 x 1 -- > num_examples x 120
        )
        self.fcs = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)
        return x


device = 'cuda' if torch.cuda.is_available() else 'cpu'
x = torch.randn(64, 1, 32, 32).to(device)
model = LeNet().to(device)
print(model(x).shape)

