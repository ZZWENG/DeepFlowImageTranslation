import torch
import torch.nn as nn


class Splitting(nn.Module):
    def __init__(self, scale):
        super(Splitting, self).__init__()
        self.scale = scale

    def forward(self, x, sldj, z):
        # Split in half along channel dimension
        new_z, y = x.chunk(2, dim=1)
        # scale = 0 : x: 1,4,14,14
        # scale = 1 : x: 1,2,14,14

        if z is None:
            z = new_z
        else:
            z = torch.cat((z, new_z), dim=1)

        # scale = 0: y: 1,2,14,14, z: 1,2,14,14
        # scale = 1: y: 1,1,14,14, z: 1,3,14,14
        return y, sldj, z

    def backward(self, y, z):
        if y is None:
            num_take = z.size(1) // (2 ** self.scale)   # 4 -> 2
        else:
            num_take = y.size(1)

        if num_take == z.size(1):
            new_y = z
            z = None
        else:
            z, new_y = z.split((z.size(1) - num_take, num_take), dim=1)
            # scale = 0, split by 2,2
            # scale = 1, split by

        # Add split features back to x
        if y is None:
            x = new_y
        else:
            x = torch.cat((new_y, y), dim=1)

        return x, z
