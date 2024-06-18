import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, width * height)
        energy = torch.bmm(query, key)
        attention = F.softmax(energy, dim=-1)
        value = self.value_conv(x).view(batch_size, -1, width * height)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualDenseBlock, self).__init__()
        self.layer1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.layer4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.layer5 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.layer1(x)
        out = F.relu(out)
        out = self.layer2(out)
        out = F.relu(out)
        out = self.layer3(out)
        out = F.relu(out)
        out = self.layer4(out)
        out = F.relu(out)
        out = self.layer5(out)
        return out * 0.2 + x

class ComplexCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.attention = SelfAttention(128)
        self.rrdb = ResidualDenseBlock(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 64 * 64, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.attention(x)
        x = self.rrdb(x)
        x = x.view(-1, 128 * 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComplexCNN().to(device)
    summary(model, (3, 256, 256))
