import torch
import torch.nn as nn


class Convolution2D(nn.Module):

    def __init__(self, inplanes, outplanes, batch_norm, use_bias):
        nn.Module.__init__(self)

        # print(inplanes)
        # print(outplanes)
        # print(batch_norm)
        # print(use_bias)

        self.conv = nn.Conv2d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [3, 3],
            stride       = [1, 1],
            padding      = [1, 1],
            bias         = use_bias)

        self._batch_norm = batch_norm


        if self._batch_norm:
            self.bn   = nn.BatchNorm2d(outplanes)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self._batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        return out

class Convolution3D(nn.Module):

    def __init__(self, inplanes, outplanes, batch_norm, use_bias):
        nn.Module.__init__(self)


        self.conv = nn.Conv3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [3, 3, 3],
            stride       = [1, 1, 1],
            padding      = [1, 1, 1],
            bias         = use_bias)

        self._batch_norm = batch_norm


        if self._batch_norm:
            self.bn   = nn.BatchNorm3d(outplanes)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self._batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        return out


class ResidualBlock2D(nn.Module):

    def __init__(self, inplanes, outplanes, batch_norm, use_bias):
        nn.Module.__init__(self)


        self.conv1 = nn.Conv2d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [3, 3],
            stride       = [1, 1],
            padding      = [1, 1],
            bias         = use_bias)

        self._batch_norm = batch_norm


        if self._batch_norm:
            self.bn1 = nn.BatchNorm2d(outplanes)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(
            in_channels  = outplanes, 
            out_channels = outplanes, 
            kernel_size  = [3, 3],
            stride       = [1, 1],
            padding      = [1, 1],
            bias         = use_bias)

        self._batch_norm = batch_norm


        if self._batch_norm:
            self.bn2 = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self._batch_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self._batch_norm:
            out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class ResidualBlock3D(nn.Module):

    def __init__(self, inplanes, outplanes, batch_norm, use_bias):
        nn.Module.__init__(self)


        self.conv1 = nn.Conv3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [3, 3],
            stride       = [1, 1],
            padding      = [1, 1],
            bias         = use_bias)

        self._batch_norm = batch_norm


        if self._batch_norm:
            self.bn1 = nn.BatchNorm3d(outplanes)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(
            in_channels  = outplanes, 
            out_channels = outplanes, 
            kernel_size  = [3, 3],
            stride       = [1, 1],
            padding      = [1, 1],
            bias         = use_bias)

        self._batch_norm = batch_norm


        if self._batch_norm:
            self.bn2 = nn.BatchNorm3d(outplanes)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self._batch_norm:
            out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)

        if self._batch_norm:
            out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class ConvolutionDownsample2D(nn.Module):

    def __init__(self, inplanes, outplanes, batch_norm, use_bias):
        nn.Module.__init__(self)

        self.conv = nn.Conv2d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [2, 2],
            stride       = [2, 2],
            padding      = [0, 0],
            bias         = use_bias)

        self._batch_norm = batch_norm

        if self._batch_norm:
            self.bn   = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self._batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        return out



class ConvolutionDownsample3D(nn.Module):

    def __init__(self, inplanes, outplanes, batch_norm, use_bias):
        nn.Module.__init__(self)

        self.conv = nn.Conv3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [2 ,2, 2],
            stride       = [2, 2, 2],
            padding      = [0, 0, 0],
            bias         = use_bias)

        self._batch_norm = batch_norm

        if self._batch_norm:
            self.bn   = nn.BatchNorm3d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        if self._batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        return out


class ConvolutionUpsample2D(nn.Module):

    def __init__(self, inplanes, outplanes, batch_norm, use_bias):
        nn.Module.__init__(self)

        self.conv = nn.ConvTranspose2d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [2, 2],
            stride       = [2, 2],
            padding      = [0, 0],
            bias         = use_bias)

        self._batch_norm = batch_norm

        if self._batch_norm:
            self.bn   = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv(x)
        if self._batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        return out


class ConvolutionUpsample3D(nn.Module):

    def __init__(self, inplanes, outplanes, batch_norm, use_bias):
        nn.Module.__init__(self)

        self.conv = nn.ConvTranspose3d(
            in_channels  = inplanes, 
            out_channels = outplanes, 
            kernel_size  = [2, 2, 2],
            stride       = [2, 2, 2],
            padding      = [0, 0, 0],
            bias         = use_bias)

        self._batch_norm = batch_norm

        if self._batch_norm:
            self.bn   = nn.BatchNorm3d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.conv(x)
        if self._batch_norm:
            out = self.bn(out)
        out = self.relu(out)
        return out


class BlockSeries2D(torch.nn.Module):


    def __init__(self, inplanes, n_blocks, batch_norm, residual, use_bias):
        torch.nn.Module.__init__(self)

        if not residual:
            self.blocks = [ Convolution2D(inplanes, inplanes, batch_norm, use_bias) for i in range(n_blocks) ]
        else:
            self.blocks = [ ResidualBlock2D(inplanes, inplanes, batch_norm, use_bias) for i in range(n_blocks)]

        for i, block in enumerate(self.blocks):
            self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

        return x


class BlockSeries3D(torch.nn.Module):


    def __init__(self, inplanes, n_blocks, batch_norm, residual, use_bias):
        torch.nn.Module.__init__(self)

        if not residual:
            self.blocks = [ 
                Convolution3D(
                    inplanes    = inplanes,
                    outplanes   = inplanes, 
                    batch_norm  = batch_norm, 
                    use_bias    = use_bias
                ) for i in range(n_blocks) ]
        else:
            self.blocks = [ ResidualBlock3D(                    
                    inplanes    = inplanes,
                    outplanes   = inplanes, 
                    batch_norm  = batch_norm, 
                    use_bias    = use_bias
                ) for i in range(n_blocks) ]
            
        for i, block in enumerate(self.blocks):
            self.add_module('block_{}'.format(i), block)


    def forward(self, x):
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

        return x
