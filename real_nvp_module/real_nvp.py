import torch
import torch.nn as nn

from real_nvp_module.coupling import Coupling
from real_nvp_module.splitting import Splitting
from real_nvp_module.squeezing import Squeezing


class RealNVP(nn.Module):
    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=6):
        super(RealNVP, self).__init__()
        self.alpha = 1e-5
        self.nc = in_channels
        self.z_channels = 4 ** (num_scales - 1) * in_channels  # 4**1 * 3 = 12

        layers = self._build_layers(in_channels, mid_channels, num_blocks, num_scales)
        self.model = nn.ModuleList(layers)

    def _build_layers(self, in_channels, mid_channels, num_blocks, num_scales):
        old_in_channels = in_channels
        layers = []
        for scale in range(num_scales):

            layers += [Coupling(in_channels, mid_channels, num_blocks, mask_type='checkerboard', reverse_mask=False),
                       Coupling(in_channels, mid_channels, num_blocks, mask_type='checkerboard', reverse_mask=True),
                       Coupling(in_channels, mid_channels, num_blocks, mask_type='checkerboard', reverse_mask=False)
                       ]

            if scale < num_scales - 1:
                in_channels *= 4
                mid_channels *= 2
                layers += [Squeezing(),
                           Coupling(in_channels, mid_channels, num_blocks, mask_type='channel_wise', reverse_mask=False),
                           Coupling(in_channels, mid_channels, num_blocks, mask_type='channel_wise', reverse_mask=True),
                           Coupling(in_channels, mid_channels, num_blocks, mask_type='channel_wise', reverse_mask=False)]
            else:
                layers += [Coupling(in_channels, mid_channels, num_blocks, mask_type='checkerboard', reverse_mask=True)]

            layers += [Splitting(scale)]
            in_channels = old_in_channels * 2
        return layers

    def forward(self, x):
        y = (x * (256 - 1.0) + torch.rand_like(x)) / (1.0 * 256)

        y = self.alpha * 0.5 + (1 - self.alpha) * y
        sldj = -(y.log() + (1 - y).log())
        sldj = sldj.view(sldj.size(0), -1).sum(-1)
        y = y.log() - (1 - y).log()

        z = None
        for i, layer in enumerate(self.model):
            y, sldj, z = layer.forward(y, sldj, z)

        if z is not None:
            z = torch.cat((z, y), dim=1)
        else:
            z = y

        return z, sldj

    def backward(self, z):
        y = None
        for i, layer in enumerate(reversed(self.model)):
            y, z = layer.backward(y, z)
        x = torch.sigmoid(y)
        return x
