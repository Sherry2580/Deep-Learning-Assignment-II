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
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)



class ComplexCNN(nn.Module):
    def __init__(self, num_classes=50):
        super(ComplexCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.sa = SpatialAttention()
        self.attention = SelfAttention(128)
        self.rrdb = ResidualDenseBlock(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 1 * 1, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.4)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        input image size: (batch_size, 3, 256, 256)
        """
        # conv1
        # (batch_size, 3, 256, 256) -> (batch_size, 64, 256, 256)
        # pool1
        # (batch_size, 64, 256, 256) -> (batch_size, 64, 128, 128)
        x = self.pool(F.relu(self.conv1(x)))  # [64, 128, 128]
        # conv2
        # (batch_size, 64, 128, 128) -> (batch_size, 128, 128, 128)
        # pool2
        # (batch_size, 128, 128, 128) -> (batch_size, 128, 64, 64)
        x = self.pool(F.relu(self.conv2(x)))  # [128, 64, 64]
        # x = self.attention(x) # out of memory
        x = self.sa(x) * x  # [128, 64, 64]
        x = self.rrdb(x)  # [128, 64, 64]
        # dropout
        x = self.dropout(x)
        # global average pooling
        # (batch_size, 128, 64, 64) -> (batch_size, 128, 1, 1)
        x = self.global_avg_pool(x)
        # Flatten
        # (batch_size, 128, 1, 1) -> (batch_size, 128 * 1 * 1)
        x = x.view(-1, 128 * 1 * 1)

        # x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ComplexCNN().to(device)
    summary(model, (3, 256, 256))