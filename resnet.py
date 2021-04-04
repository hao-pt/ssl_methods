import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np



class BottleNeck(nn.Module):
    def __init__(self, in_ch, out_ch, times) -> None:
        super(BottleNeck, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.Sequential(
            nn.Conv2d(in_ch, in_ch//2, kernel_size=1, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//2, in_ch//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch//2, out_ch, kernel_size=3, stride=1, padding=1),
        )

        self.remaining_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_ch, out_ch//2, kernel_size=1, stride=1, padding=1),
                nn.Conv2d(out_ch//2, in_ch//2, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(out_ch//2, out_ch, kernel_size=3, stride=1, padding=1),
            ) for _ in range(times-1)
        ])

    def forward(self, input):
        x = self.layers(input)
        for block in self.remaining_layers:
            x_clone = x.clone()
            x = block(x) + x_clone
            self.relu(x)
        
        return x

class ShortCut(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super(ShortCut, self).__init__()

        # shake-shake paper
        # self.stride = stride
        # self.conv1 = nn.Conv2d(in_ch, out_ch//2, 1, stride=1, padding=0, bias=False)
        # self.conv2 = nn.Conv2d(in_ch, out_ch//2, 1, stride=1, padding=0, bias=False)
        # self.bn = nn.BatchNorm2d(out_ch)

        # mean teacher paper
        self.conv = nn.Conv2d(in_ch*2, out_ch, 
            1, stride=1, padding=0, groups=2)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, input): 
        # shake-shake paper
        # x = F.relu(input)

        # # flow 1
        # x1 = F.avg_pool2d(x, 1, self.stride)
        # x1 = self.conv1(x)

        # # flow 2
        # x2 = F.avg_pool2d(F.pad(x, (-1,1,-1,1)), 1, self.stride)
        # x2 = self.conv2(x)
        # x = torch.cat((x1, x2), dim=1)
        
        # return self.bn(x)

        # mean teacher paper
        # split x into 2 sub-tensor: (1) even x,y (2) odd x,y 
        x = torch.cat((input[...,0::2, 0::2], input[...,1::2, 1::2]), dim=1)
        x = F.relu(x)
        x = self.conv(x)
        x = self.bn(x)

        return x
        

class ShakeBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(ShakeBlock, self).__init__()
        
        self.convs_brach1 = self._make_branch(in_ch, out_ch, stride)
        self.convs_brach2 = self._make_branch(in_ch, out_ch, stride)

        self.shortcut = None if in_ch == out_ch else ShortCut(in_ch, out_ch, stride)
        
        self.downsample = downsample

    def forward(self, input):

        x1 = self.convs_brach1(input)
        x2 = self.convs_brach2(input)

        x = ShakeShake.apply(x1, x2, self.training)
        residual = input if not self.shortcut else self.shortcut(input)

        return x+residual

    def _make_branch(self, in_ch, out_ch, stride):
        return nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, 
                stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, 
                stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

class ShakeShake(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, training):
        if training:
            alpha = torch.rand(x1.size(0)).to(x1.device)
            alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x1) # reshape alpha
        else:
            alpha = 0.5

        x = x1*alpha + (1-alpha)*x2

        return x

    @staticmethod
    def backward(ctx, grad):
        beta = torch.rand(grad.size(0)).to(grad.device)
        beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad) # reshape beta
        
        beta = Variable(beta)

        return beta*grad, (1-beta)*grad, None
        
class ShakeResNet(nn.Module):
    def __init__(self, in_ch, out_chs, depth, num_classes):
        super(ShakeResNet, self).__init__()
        # first layer
        self.conv1 = nn.Conv2d(in_ch, out_chs[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.num_units = (depth-2)//6 # number of residual units in one conv block

        self.conv2_x = self._make_layer(out_chs[0], out_chs[1], 1)
        self.conv3_x = self._make_layer(out_chs[1], out_chs[2], 2)
        self.conv4_x = self._make_layer(out_chs[2], out_chs[3], 2)

        self.avg_pool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(out_chs[-1], num_classes)

        self._init_weights()
    
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)

        # x = F.relu(x)
        x = self.avg_pool(x)
        # flatten
        x = x.view(x.size(0), -1) # (#batches, -1)
        
        return self.fc1(x)

    def _make_layer(self, in_ch, out_ch, stride):
        stage = []
        # create residual blocks
        for i in range(self.num_units):
            stage.append(ShakeBlock(in_ch, out_ch, stride))
            in_ch, stride = out_ch, 1

        return nn.Sequential(*stage)
            
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d): # init conv weights
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


