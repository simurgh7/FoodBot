import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, out_channels3, out_channels4, dropout_rate=0.2):
        super(InceptionBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size=1)
        self.conv2_1 = nn.Conv2d(in_channels, out_channels2[0], kernel_size=1)
        self.conv2_2 = nn.Conv2d(out_channels2[0], out_channels2[1], kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(in_channels, out_channels3[0], kernel_size=1)
        self.conv3_2 = nn.Conv2d(out_channels3[0], out_channels3[1], kernel_size=5, padding=2)
        self.conv4_1 = nn.Conv2d(in_channels, out_channels4[0], kernel_size=1)
        self.conv4_2 = nn.Conv2d(out_channels4[0], out_channels4[1], kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2_2(F.relu(self.conv2_1(x))))
        x3 = F.relu(self.conv3_2(F.relu(self.conv3_1(x))))
        x4 = F.relu(self.conv4_2(F.relu(self.conv4_1(x))))
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.dropout(x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class StateCNN(nn.Module):
    def __init__(self, num_classes=11, init_option="random", dropout_rate=0.2):
        super(StateCNN, self).__init__()
        self.init_option = init_option
        self.dropout_rate = dropout_rate
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2)
        self.batchnorm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(128)

        self.inception1 = InceptionBlock(in_channels=128, out_channels1=64, out_channels2=(32, 64), out_channels3=(32, 64), out_channels4=(32, 64), dropout_rate=0.3)
        self.residual1 = ResidualBlock(256, 256, stride=2, dropout_rate=0.3)
        self.ir_batchnorm1 = nn.BatchNorm2d(256)

        self.inception2 = InceptionBlock(in_channels=256, out_channels1=128, out_channels2=(64, 128), out_channels3=(64, 128), out_channels4=(64, 128), dropout_rate=0.3)
        self.residual2 = ResidualBlock(512, 512, stride=2, dropout_rate=0.3)
        self.ir_batchnorm2 = nn.BatchNorm2d(512)

        self.fc1 = nn.Linear(512*8*8, 512)  # Adjusted for 256x256 input size
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
        self.init_weights()
    
    def init_weights(self):
        if self.init_option == 'xavier':
            for layer in self.children():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
        if self.init_option == 'random':
            for layer in self.children():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        x = self.relu(self.batchnorm1(self.conv1(x)))
        x = self.maxpool(x)
        #print(x.shape)
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.maxpool(x)
        #print(x.shape)
        x = self.relu(self.batchnorm4(self.conv4(x)))
        x = self.inception1(x)
        x = self.residual1(x)
        x = self.ir_batchnorm1(x)
        x = self.maxpool(x)
        #print(x.shape)
        x = self.inception2(x)
        x = self.residual2(x)
        x = self.ir_batchnorm2(x)
        x = self.maxpool(x)
        #print(x.shape)
        x = x.view(-1, 512*8*8)  # Adjusted for 256x256 input size
        print(x.shape)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)  # Apply dropout
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x
