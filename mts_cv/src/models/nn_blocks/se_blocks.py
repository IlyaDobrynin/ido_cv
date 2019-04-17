import torch
from torch import nn


class SEBlock(nn.Module):
    
    def __init__(self, channels, reduction):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SCSEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCSEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.channel_excitation = nn.Sequential(nn.Linear(channel, int(channel // reduction)),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(int(channel // reduction), channel),
                                                nn.Sigmoid())
        
        self.spatial_se = nn.Sequential(nn.Conv2d(channel, 1, kernel_size=1,
                                                  stride=1, padding=0, bias=False),
                                        nn.Sigmoid())
    
    def forward(self, x):
        bahs, chs, _, _ = x.size()
        
        # Returns a new tensor with the same data as the self tensor but of a different size.
        chn_se = self.avg_pool(x).view(bahs, chs)
        chn_se = self.channel_excitation(chn_se).view(bahs, chs, 1, 1)
        chn_se = torch.mul(x, chn_se)
        
        spa_se = self.spatial_se(x)
        spa_se = torch.mul(x, spa_se)
        out = torch.add(chn_se, 1, spa_se)
        # print(out.shape)
        return out


class ConcurrentSEModule(nn.Module):
    
    def __init__(self, channels, reduction=2):
        super(ConcurrentSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        
        self.fc = nn.Conv2d(channels, 1, kernel_size=1)
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x_c = self.avg_pool(x)
        x_c = self.fc1(x_c)
        x_c = self.relu(x_c)
        x_c = self.fc2(x_c)
        x_c = self.sigmoid(x_c)
        
        x_s = self.fc(x)
        x_s = self.sigmoid(x_s)
        
        return x * x_c + x * x_s