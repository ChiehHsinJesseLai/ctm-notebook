"""
Import required libraries
"""
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from pg_modules.networks_stylegan2 import Conv2dLayer, Conv2dLayerDepthwise


def conv2d(*args, **kwargs):
    """
    Applies spectral normalization to a 2D convolution layer.
    """
    return spectral_norm(nn.Conv2d(*args, **kwargs))

def convTranspose2d(*args, **kwargs):
    """
    Applies spectral normalization to a transposed 2D convolution layer.
    """
    return spectral_norm(nn.ConvTranspose2d(*args, **kwargs))

def embedding(*args, **kwargs):
    """
    Applies spectral normalization to an embedding layer.
    """
    return spectral_norm(nn.Embedding(*args, **kwargs))

def linear(*args, **kwargs):
    """
    Applies spectral normalization to a linear layer.
    """
    return spectral_norm(nn.Linear(*args, **kwargs))

def NormLayer(c, mode='batch'):
    """
    Returns a normalization layer based on the specified mode.
    """
    if mode == 'group':
        return nn.GroupNorm(c//2, c)
    elif mode == 'batch':
        return nn.BatchNorm2d(c)

### Activations

class GLU(nn.Module):
    """
    Gated Linear Unit activation function.
    """
    def forward(self, x):
        """
        Forward pass for the GLU activation function.
        """
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])

class Swish(nn.Module):
    """
    Swish activation function.
    """
    def forward(self, feat):
        """
        Forward pass for the Swish activation function.
        """
        return feat * torch.sigmoid(feat)

### Upblocks

class InitLayer(nn.Module):
    """
    Initial layer for the network.
    """
    def __init__(self, nz, channel, sz=4):
        """
        Initializes the InitLayer with the specified parameters.
        """
        super().__init__()

        self.init = nn.Sequential(
            convTranspose2d(nz, channel*2, sz, 1, 0, bias=False),
            NormLayer(channel*2),
            GLU(),
        )

    def forward(self, noise):
        """
        Forward pass for the InitLayer.
        """
        noise = noise.view(noise.shape[0], -1, 1, 1)
        return self.init(noise)

def UpBlockSmall(in_planes, out_planes):
    """
    Creates a small upsampling block.
    """
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        NormLayer(out_planes*2), GLU())
    return block

class UpBlockSmallCond(nn.Module):
    """
    Small upsampling block with conditional normalization.
    """
    def __init__(self, in_planes, out_planes, z_dim):
        """
        Initializes the UpBlockSmallCond with the specified parameters.
        """
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)

        which_bn = functools.partial(CCBN, which_linear=linear, input_size=z_dim)
        self.bn = which_bn(2*out_planes)
        self.act = GLU()

    def forward(self, x, c):
        """
        Forward pass for the UpBlockSmallCond.
        """
        x = self.up(x)
        x = self.conv(x)
        x = self.bn(x, c)
        x = self.act(x)
        return x

def UpBlockBig(in_planes, out_planes):
    """
    Creates a big upsampling block.
    """
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        NormLayer(out_planes*2), GLU(),
        conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False),
        NoiseInjection(),
        NormLayer(out_planes*2), GLU()
        )
    return block

class UpBlockBigCond(nn.Module):
    """
    Big upsampling block with conditional normalization.
    """
    def __init__(self, in_planes, out_planes, z_dim):
        """
        Initializes the UpBlockBigCond with the specified parameters
        """
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv1 = conv2d(in_planes, out_planes*2, 3, 1, 1, bias=False)
        self.conv2 = conv2d(out_planes, out_planes*2, 3, 1, 1, bias=False)

        which_bn = functools.partial(CCBN, which_linear=linear, input_size=z_dim)
        self.bn1 = which_bn(2*out_planes)
        self.bn2 = which_bn(2*out_planes)
        self.act = GLU()
        self.noise = NoiseInjection()

    def forward(self, x, c):
        """
        Forward pass for the UpBlockBigCond.
        """
        # block 1
        x = self.up(x)
        x = self.conv1(x)
        x = self.noise(x)
        x = self.bn1(x, c)
        x = self.act(x)

        # block 2
        x = self.conv2(x)
        x = self.noise(x)
        x = self.bn2(x, c)
        x = self.act(x)

        return x

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block.
    """
    def __init__(self, ch_in, ch_out):
        """
        Initializes the SEBlock with the specified parameters.
        """
        super().__init__()
        self.main = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            conv2d(ch_in, ch_out, 4, 1, 0, bias=False),
            Swish(),
            conv2d(ch_out, ch_out, 1, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, feat_small, feat_big):
        """
        Forward pass for the SEBlock.
        """
        return feat_big * self.main(feat_small)

### Downblocks

class DownBlock(nn.Module):
    """
    Downsampling block.
    """
    def __init__(self, in_planes, out_planes, width=1):
        """
        Initializes the DownBlock with the specified parameters.
        """
        super().__init__()
        self.main = nn.Sequential(
            conv2d(in_planes, out_planes*width, 4, 2, 1, bias=True),
            NormLayer(out_planes*width),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        """
        Forward pass for the DownBlock.
        """
        return self.main(feat)

class DownBlockSGBlocks(nn.Module):
    """
    Downsampling block with SG blocks.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initializes the DownBlockSGBlocks with the specified parameters.
        """
        super().__init__()
        conv_depthwise = Conv2dLayerDepthwise(
            in_channels, in_channels,
            kernel_size=3,
            activation='linear')
        conv_pointwise = Conv2dLayer(
            in_channels,
            out_channels,
            kernel_size=1,
            activation='lrelu', down=2)
        self.main = nn.Sequential(conv_depthwise, conv_pointwise)

    def forward(self, feat):
        """
        Forward pass for the DownBlockSGBlocks.
        """
        return self.main(feat)

class SeparableConv2d(nn.Module):
    """
    Separable 2D convolution layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        """
        Initializes the SeparableConv2d layer with the specified parameters
        """
        super(SeparableConv2d, self).__init__()
        self.depthwise = conv2d(in_channels, in_channels, kernel_size=kernel_size,
            groups=in_channels, bias=bias, padding=1)
        self.pointwise = conv2d(in_channels, out_channels,
            kernel_size=1, bias=bias)

    def forward(self, x):
        """
        Forward pass for the SeparableConv2d layer.
        """
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class DownBlockSep(nn.Module):
    """
    Downsampling block with separable convolutions.
    """
    def __init__(self, in_planes, out_planes):
        """
        Initializes the DownBlockSep with the specified parameters.
        """
        super().__init__()
        self.main = nn.Sequential(
            SeparableConv2d(in_planes, out_planes, 3),
            NormLayer(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(2, 2),
        )

    def forward(self, feat):
        """
        Forward pass for the DownBlockSep.
        """
        return self.main(feat)

class DownBlockPatch(nn.Module):
    """
    Downsampling block with patch convolutions.
    """
    def __init__(self, in_planes, out_planes):
        """
        Initializes the DownBlockPatch with the specified parameters.
        """
        super().__init__()
        self.main = nn.Sequential(
            DownBlock(in_planes, out_planes),
            conv2d(out_planes, out_planes, 1, 1, 0, bias=False),
            NormLayer(out_planes),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, feat):
        """
        Forward pass for the DownBlockPatch.
        """
        return self.main(feat)

### CSM

class ResidualConvUnit(nn.Module):
    """
    Residual convolution unit.
    """
    def __init__(self, cin, activation, bn):
        """
        Initializes the ResidualConvUnit with the specified parameters.
        """
        super().__init__()
        self.conv = nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=True)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """
        Forward pass for the ResidualConvUnit.
        """
        return self.skip_add.add(self.conv(x), x)

class FeatureFusionBlock(nn.Module):
    """
    Feature fusion block.
    """
    def __init__(self, features,
                 activation, deconv=False,
                 bn=False, expand=False,
                 align_corners=True,
                 lowest=False):
        """
        Initializes the FeatureFusionBlock with the specified parameters.
        """
        super().__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2

        self.out_conv = nn.Conv2d(features,
                                  out_features,
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bias=True, groups=1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """
        Forward pass for the FeatureFusionBlock.
        """
        output = xs[0]

        if len(xs) == 2:
            output = self.skip_add.add(output, xs[1])

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output

### Misc

class NoiseInjection(nn.Module):
    """
    Noise injection layer.
    """
    def __init__(self):
        """
        Initializes the NoiseInjection layer.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, feat, noise=None):
        """
        Forward pass for the NoiseInjection layer.
        """
        if noise is None:
            batch, _, height, width = feat.shape
            noise = torch.randn(batch, 1, height, width).to(feat.device)

        return feat + self.weight * noise

class CCBN(nn.Module):
    ''' conditional batchnorm '''
    def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1):
        """
        Initializes the CCBN layer with the specified parameters.
        """
        super().__init__()
        self.output_size, self.input_size = output_size, input_size

        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)

        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum

        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y):
        """
         Forward pass for the CCBN layer.
        """
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1, 1, 1)
        bias = self.bias(y).view(y.size(0), -1, 1, 1)
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                           self.training, 0.1, self.eps)
        return out * gain + bias

class CCBN1D(nn.Module):
    '''
    Conditional batch normalization layer for 1D inputs. 
    '''
    def __init__(self, output_size, input_size, which_linear, eps=1e-5, momentum=0.1):
        """
        Initializes the CCBN1D layer with the specified parameters.
        """
        super().__init__()
        self.output_size, self.input_size = output_size, input_size

        # Prepare gain and bias layers
        self.gain = which_linear(input_size, output_size)
        self.bias = which_linear(input_size, output_size)

        # epsilon to avoid dividing by 0
        self.eps = eps
        # Momentum
        self.momentum = momentum

        self.register_buffer('stored_mean', torch.zeros(output_size))
        self.register_buffer('stored_var', torch.ones(output_size))

    def forward(self, x, y):
        """
        Forward pass for the CCBN1D layer.
        """
        # Calculate class-conditional gains and biases
        gain = (1 + self.gain(y)).view(y.size(0), -1)
        bias = self.bias(y).view(y.size(0), -1)
        out = F.batch_norm(x, self.stored_mean, self.stored_var, None, None,
                           self.training, 0.1, self.eps)
        return out * gain + bias

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, size, mode='bilinear', align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            size=self.size,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x
