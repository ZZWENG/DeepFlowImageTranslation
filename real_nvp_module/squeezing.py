import real_nvp_module.array_util as util
import torch.nn as nn


class Squeezing(nn.Module):
    def __init__(self):
        super(Squeezing, self).__init__()

    def forward(self, x, sldj, z=None):
        y = util.space_to_depth(x, 2)
        if z is not None:
            z = util.space_to_depth(z, 2)

        return y, sldj, z

    def backward(self, y, z):
        x = util.depth_to_space(y, 2)
        if z is not None:
            z = util.depth_to_space(z, 2)

        return x, z
