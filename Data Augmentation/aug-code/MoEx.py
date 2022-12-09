# https://github.com/Boyiliee/MoEx
import torch
import torch.nn as nn

class PONO(nn.Module):
    def __init__(self, input_size=None, affine=True, eps=1e-5):
        super(PONO, self).__init__()
        self.input_size = input_size
        self.eps = eps
        self.affine = affine

        if affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta
        return x, mean, std

class MomentShortcut(nn.Module):
    def __init__(self, beta=None, gamma=None):
        super(MomentShortcut, self).__init__()
        self.gamma, self.beta = gamma, beta

    def forward(self, x, beta=None, gamma=None):
        beta = self.beta if beta is None else beta
        gamma = self.gamma if gamma is None else gamma
        if gamma is not None:
            x.mul_(gamma)
        if beta is not None:
            x.add_(beta)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(64, 2)
        self.ms = MomentShortcut()
        self.pono = PONO(affine=False)

    def forward(self, x, input2=None):
        x = self.conv1(x)
        x = self.bn1(x)

        if input2 is not None:
            x2 = self.conv1(input2)
            x2 = self.bn1(x2)
            x, _, _ = self.pono(x)
            x2, mean, std = self.pono(x2)
            x = self.ms(x, mean, std)
        return x


if __name__ == "__main__":
    x1 = torch.randn(1, 3, 20, 20)
    x2 = torch.randn(1, 3, 20, 20)
    model = Net()
    output = model(x1, x2)