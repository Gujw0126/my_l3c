import torch
import torch.nn as nn
__all__=[
    "conv3x3",
    "conv5x5",
    "conv1x1",
    "residual_block",
    "AtrousConv",
    "EncUnit",
    "DecUnit",
    "DecUnitFirst",
    "EncUnitLast",
    "MeanShift"
]

def conv3x3(in_ch:int, out_ch:int)->nn.Module:
    return nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)


def conv5x5(in_ch:int, out_ch:int)->nn.Module:
    return nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=5, stride=2, padding=2)

def conv1x1(in_ch:int, out_ch:int)->nn.Module:
    return nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1)


class residual_block(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = conv3x3(in_ch,in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_ch, in_ch)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + x
        return out


class AtrousConv(nn.Module):
    """
    AtrousConv, in_ch->3*in_ch
    rate=1,2,4
    """
    def __init__(self, in_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, dilation=1, padding=1)
        self.conv2 = nn.Conv2d(in_ch, in_ch, 3, dilation=2, padding=2)
        self.conv3 = nn.Conv2d(in_ch, in_ch, 3, dilation=4, padding=4)
        self.s1f1 = nn.Conv2d(in_ch*3, in_ch, kernel_size=1)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out = torch.concatenate((out1, out2, out3), dim=1)
        out = self.s1f1(out)
        return out


class EncUnit(nn.Module):
    def __init__(self, in_ch, out_ch, res_num, C_Q):
        """
        res_num:residual number
        C_Q: quantization input channel number
        """
        super().__init__()
        self.s2f5 = conv5x5(in_ch,out_ch)
        self.residual_blocks = nn.Sequential()
        for i in range(res_num):
            self.residual_blocks.append(residual_block(out_ch))
        self.conv1 = conv3x3(out_ch, out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.s1f1 = conv1x1(out_ch, C_Q)
    
    def forward(self, x):
        out1 = self.s2f5(x)
        out = self.residual_blocks(out1)
        out =self.conv1(out)
        out = out + out1
        E = self.conv2(out)
        out_q = self.s1f1(out)
        return E, out_q

class EncUnitLast(nn.Module):
    def __init__(self, in_ch, out_ch, res_num, C_Q):
        """
        last encoder unit, only output z(s)
        """
        super().__init__()
        self.s2f5 = conv5x5(in_ch,out_ch)
        self.residual_blocks = nn.Sequential()
        for i in range(res_num):
            self.residual_blocks.append(residual_block(out_ch))
        self.conv1 = conv3x3(out_ch, out_ch)
        self.s1f1 = conv1x1(out_ch, C_Q)   

    def forward(self, x):
        out1 = self.s2f5(x)
        out = self.residual_blocks(out1)
        out = self.conv1(out)
        out = out + out1
        out = self.s1f1(out)
        return out 

    
#Decode Unit for s>0
class DecUnit(nn.Module):
    def __init__(self, in_ch, C_Q, res_num, param_num):
        """
        in_ch: input_channels
        C_Q: channel number of Quantization output
        res_num: residual number
        K: logistic mixture number
        """
        super().__init__()
        self.s1f1 = conv1x1(C_Q, in_ch)
        self.residual_blocks = nn.Sequential()
        for i in range(res_num):
            self.residual_blocks.append(residual_block(in_ch))
        self.conv1 = conv3x3(in_ch, in_ch)
        self.conv2 = conv3x3(in_ch, in_ch*4)
        self.upsample = nn.PixelShuffle(2)
        self.atrous = AtrousConv(in_ch)
        self.s1f1_param  = conv1x1(in_ch, param_num)

    def forward(self, latent, f):
        """
        latent: decoded latent
        f: feature from previous layer
        """
        out_latent = self.s1f1(latent)
        out = out_latent + f
        out1 = self.residual_blocks(out)
        out1 = self.conv1(out1)
        out = out + out1
        out = self.conv2(out)
        out = self.upsample(out)
        out = self.atrous(out)
        params = self.s1f1_param(out)
        return out, params


class DecUnitFirst(nn.Module):
    def __init__(self, in_ch, C_Q, res_num, param_num):
        """
        in_ch: input_channels
        C_Q: channel number of Quantization output
        res_num: residual number
        K: logistic mixture number
        """
        super().__init__()
        self.s1f1 = conv1x1(C_Q, in_ch)
        self.residual_blocks = nn.Sequential()
        for i in range(res_num):
            self.residual_blocks.append(residual_block(in_ch))
        self.conv1 = conv3x3(in_ch, in_ch)
        self.conv2 = conv3x3(in_ch, in_ch*4)
        self.upsample = nn.PixelShuffle(2)
        self.atrous = AtrousConv(in_ch)
        self.s1f1_param  = conv1x1(in_ch, param_num)
    
    def forward(self, x):
        out1 = self.s1f1(x)
        out = self.residual_blocks(out1)
        out = self.conv1(out)
        out = out1 + out
        out = self.conv2(out)
        out = self.upsample(out)
        out = self.atrous(out)
        params = self.s1f1_param(out)
        return out, params


class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False