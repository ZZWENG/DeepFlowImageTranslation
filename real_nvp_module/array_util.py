import torch


def depth_to_space(x, block_size):
    if block_size == 1:
        return x
    output = x.permute(0, 2, 3, 1)
    (batch_size, d_height, d_width, d_depth) = output.size()
    s_depth = int(d_depth / block_size ** 2)
    s_width = int(d_width * block_size)
    s_height = int(d_height * block_size)
    t_1 = output.reshape(batch_size, d_height, d_width, block_size ** 2, s_depth)
    spl = t_1.split(block_size, 3)
    stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
    output = torch.stack(stack, 0)
    output = output.transpose(0, 1)
    output = output.permute(0, 2, 1, 3, 4)
    output = output.reshape(batch_size, s_height, s_width, s_depth)
    output = output.permute(0, 3, 1, 2)
    return output


def space_to_depth(x, block_size):
    if block_size == 1:
        return x
    output = x.permute(0, 2, 3, 1)
    (batch_size, s_height, s_width, s_depth) = output.size()
    d_depth = s_depth * block_size ** 2
    d_height = int(s_height / block_size)
    t_1 = output.split(block_size, 2)
    stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
    output = torch.stack(stack, 1)
    output = output.permute(0, 2, 1, 3)
    output = output.permute(0, 3, 1, 2)
    return output
