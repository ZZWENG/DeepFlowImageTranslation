import torch
import torch.nn as nn

from real_nvp_module.resnet import ResNet


def checkerboard_mask(height, width, reverse=False, device=None):
    checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
    mask = torch.tensor(checkerboard, dtype=torch.float32, device=device, requires_grad=False)
    if reverse:
        mask = 1 - mask
    mask = mask.view(1, 1, height, width)
    return mask


def channel_wise_mask(num_channels, reverse=False, device=None):
    half_channels = num_channels // 2
    channel_wise = [int(i < half_channels) for i in range(num_channels)]
    mask = torch.tensor(channel_wise, dtype=torch.float32, device=device, requires_grad=False)
    if reverse:
        mask = 1 - mask
    mask = mask.view(1, num_channels, 1, 1)
    return mask


class Coupling(nn.Module):
    def __init__(self, in_channels, mid_channels, num_blocks, mask_type, reverse_mask):
        super(Coupling, self).__init__()
        self.mask_type = mask_type
        self.reverse_mask = reverse_mask

        self.st_net = ResNet(in_channels, mid_channels, 2 * in_channels,
                             num_blocks=num_blocks, kernel_size=3, padding=1)

        self.scale = nn.utils.weight_norm(Scalar())

    def forward(self, x, sldj, z):
        y, sldj = self._flow(x, sldj, forward=True)

        return y, sldj, z

    def backward(self, y, z):
        x, _ = self._flow(y, forward=False)

        return x, z

    def _flow(self, x, sldj=None, forward=True):
        b = self._get_mask(x)
        x_b = x * b
        st = self.st_net(x_b, b)
        s, t = st.chunk(2, dim=1)
        s = self.scale(torch.tanh(s))
        s = s * (1 - b)
        t = t * (1 - b)

        if forward:
            exp_s = s.exp()
            x = x_b + (1 - b) * (x * exp_s + t)

            sldj += s.view(s.size(0), -1).sum(-1)
        else:
            exp_neg_s = s.mul(-1).exp()
            x = x_b + exp_neg_s * ((1 - b) * x - t)

        return x, sldj

    def _get_mask(self, x):
        if self.mask_type == 'checkerboard':
            return checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
        else:
            return channel_wise_mask(x.size(1), self.reverse_mask, device=x.device)


class Scalar(nn.Module):
    def __init__(self):
        super(Scalar, self).__init__()
        self.weight = nn.Parameter(torch.randn(1))

    def forward(self, x):
        x = self.weight * x
        return x
