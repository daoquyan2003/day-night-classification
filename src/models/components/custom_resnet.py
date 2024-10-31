import torch
from torch import nn
from torchvision.models import resnet18

class CustomResNet18(nn.Module):
    def __init__(self, output_size=1):
        super(CustomResNet18, self).__init__()
        self.resnet18 = resnet18(weights='DEFAULT')

        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, output_size)

    def forward(self, x):
        """Forward pass through the network.

        :param x: A tensor of images to be classified.
        :return: A tensor of predictions.
        """
        x = self.resnet18(x)
        return x

if __name__ == '__main__':
    model = CustomResNet18()
    random_tensor = torch.randn(1, 3, 256, 256)
    print(model(random_tensor).shape)
