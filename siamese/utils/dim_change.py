import torch
from einops import rearrange


def bchw2bhwc(x):
    # return x.permute(0, 2, 3, 1)  # 【B, C, H, W】 -> 【B, H, W, C】
    return rearrange(x, 'b c h w -> b h w c')


def bhwc2bchw(x):
    # return x.permute(0, 3, 1, 2)  # 【B, H, W, C】 -> 【B, C, H, W】
    return rearrange(x, 'b h w c -> b c h w')


def bchw2bnc(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def bnc2bchw(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


if __name__ == '__main__':
    x = torch.randn(32, 256, 31, 25)
    print(x.shape)
    # y = bchw2bhwc(x)
    # print(y.shape)
    # z = bhwc2bchw(y)
    # print(z.shape)
    y = bchw2bnc(x)
    print(y.shape)
    z = bnc2bchw(y, 31, 25)
    print(z.shape)
