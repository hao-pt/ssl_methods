import torch
from torch import nn
from torch.nn import functional as F

class Basic(nn.Module):
    """
    x -> conv3x3-dropout-conv3x3 -> out
    |                            |
    -----------shortcut-----------

    where each conv follows BN-Relu-Conv.
    """
    def __init__(self, in_ch, out_ch, stride=1, dropout_rate=0.0):
        super(Basic, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, 
                    stride=stride, padding=1, bias=False),
        )

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 
                    stride=1, padding=1, bias=False),
        )
        
        self.dropout = nn.Dropout(p=dropout_rate)
        self.shortcut = nn.Sequential() 
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.dropout(self.conv1(x))
        out = self.conv2(out)
        residual = self.shortcut(x)

        return residual+out

class WideResNet(nn.Module):
    """
    Architecture:
    ---
    Conv1: 16,3x3
    Conv2: [16xk,3x3
            16xk,3x3] x n
    Conv3: [32xk,3x3
            32xk,3x3] x n
    Conv4: [64xk,3x3
            64xk,3x3] x n
    avg_pool: 8x8

    fc: num_classes
    """
    def __init__(self, depth, widening_factor, dropout_rate=0., num_classes=100):
        """
        Args:
        ---
        - depth (int): number of conv layers
        - widening_factor (int): expand channels for each conv layer
        - droupout_rate (float)
        - num_classes (int)
        """
        super(WideResNet, self).__init__()
        # depth = 6*n+4 where n is number of conv layers per block
        assert (depth-4)%6 == 0
        self.num_units = (depth-4)//6
        self.width_list = [16, 16*widening_factor, 32*widening_factor, 64*widening_factor]

        self.conv1 = nn.Conv2d(3, self.width_list[0], kernel_size=3, 
            stride=1, padding=1, bias=False)
        self.conv2_group = self._make_layer(self.width_list[0], self.width_list[1], 
            stride=1, dropout_rate=dropout_rate)
        self.conv3_group = self._make_layer(self.width_list[1], self.width_list[2], 
            stride=2, dropout_rate=dropout_rate)
        self.conv4_group = self._make_layer(self.width_list[2], self.width_list[3], 
            stride=2, dropout_rate=dropout_rate)

        self.avg_pool = nn.AvgPool2d(8)
        self.bn = nn.BatchNorm2d(self.width_list[3])
        self.fc = nn.Linear(self.width_list[3], num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_group(x)
        x = self.conv3_group(x)
        x = self.conv4_group(x)

        x = F.relu(self.bn(x))
        x = self.avg_pool(x)
        # flatten
        x = x.view(x.size(0), -1) # (#batches, -1)
        
        return self.fc(x)

    def _make_layer(self, in_channels, out_channels, stride=1, dropout_rate=0.0):
        """
        Create n basic blocks for current conv group

        Note: Downsample (stride!=1) is performed only in first block.
        """
        layers = []
        for i in range(self.num_units):
            layers.append(Basic(in_channels, out_channels, stride, dropout_rate))
            in_channels, stride = out_channels, 1

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) # kaiming Initialization
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

def wrn_40_2(**kwargs):
    model = WideResNet(depth=40, widening_factor=2, **kwargs)
    return model

def wrn_40_1(**kwargs):
    model = WideResNet(depth=40, widening_factor=1, **kwargs)
    return model

def wrn_16_2(**kwargs):
    model = WideResNet(depth=16, widening_factor=2, **kwargs)
    return model

def wrn_16_1(**kwargs):
    model = WideResNet(depth=16, widening_factor=1, **kwargs)
    return model

if __name__ == "__main__":
    model = wrn_16_2(num_classes=100)
    print(model)
    x = torch.randn((4,3,32,32))
    output = model(x)
    print(output.shape)
            

        
        

        