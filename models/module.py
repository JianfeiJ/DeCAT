import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBnReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1, dilation=1):
        super(ConvBnReLU1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation,
                              bias=False)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


def differentiable_warping(src_fea, src_proj, ref_proj, depth_samples):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_samples: [B, Ndepth, H, W] 
    # out: [B, C, Ndepth, H, W]
    batch, channels, height, width = src_fea.shape
    num_depth = depth_samples.shape[1]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
        rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]

        rot_depth_xyz = rot_xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samples.view(batch, 1, num_depth,
                                                                                             height * width)  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # avoid negative depth
        negative_depth_mask = proj_xyz[:, 2:] <= 1e-3
        proj_xyz[:, 0:1][negative_depth_mask] = width
        proj_xyz[:, 1:2][negative_depth_mask] = height
        proj_xyz[:, 2:3][negative_depth_mask] = 1
        proj_xy = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = proj_xy[:, 0, :, :] / ((width - 1) / 2) - 1  # [B, Ndepth, H*W]
        proj_y_normalized = proj_xy[:, 1, :, :] / ((height - 1) / 2) - 1
        proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]
        grid = proj_xy

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=True)

    warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)
    # warped_src_fea = F.grid_sample(src_fea, grid.view(batch, num_depth * height, width, 2), mode='bilinear',
    #                            padding_mode='zeros', align_corners=True)

    # warped_src_fea = warped_src_fea.view(batch, channels, num_depth, height, width)

    return warped_src_fea


# p: probability volume [B, D, H, W]
# depth_values: discrete depth values [B, D]
# get expected value, soft argmin
# return: depth [B, 1, H, W]
def depth_regression(p, depth_values):
    depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)
    depth = depth.unsqueeze(1)
    return depth


def depth_regression_1(p, depth_values):
    depth = torch.sum(p * depth_values, 1)
    depth = depth.unsqueeze(1)
    return depth


class Positional(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim, img_size=56, patch_size=3, stride=1, with_pos=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = Positional(embed_dim)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        # _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x


class Residual(nn.Module):
    def __init__(self):
        super(Residual, self).__init__()

        # img: [B,3,H,W]
        self.conv0 = ConvBnReLU(3, 8)
        # depth map:[B,1,H/2,W/2]
        self.conv1 = ConvBnReLU(1, 8)
        self.conv2 = ConvBnReLU(8, 8)
        self.deconv = nn.ConvTranspose2d(8, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False)

        self.bn = nn.BatchNorm2d(8)
        self.conv3 = ConvBnReLU(16, 8)
        self.res = nn.Conv2d(8, 1, 3, padding=1, bias=False)

    def forward(self, img, depth_0, depth_min, depth_max):
        batch_size = depth_min.size()[0]
        # pre-scale the depth map into [0,1]
        depth = (depth_0 - depth_min.view(batch_size, 1, 1, 1)) / (
                depth_max.view(batch_size, 1, 1, 1) - depth_min.view(batch_size, 1, 1, 1))

        conv0 = self.conv0(img)
        deconv = F.relu(self.bn(self.deconv(self.conv2(self.conv1(depth)))), inplace=True)
        cat = torch.cat((deconv, conv0), dim=1)
        del deconv, conv0
        # depth residual
        res = self.res(self.conv3(cat))
        del cat

        depth = F.interpolate(depth, scale_factor=2, mode="nearest") + res
        # convert the normalized depth back
        depth = depth * (depth_max.view(batch_size, 1, 1, 1) - depth_min.view(batch_size, 1, 1, 1)) + depth_min.view(
            batch_size, 1, 1, 1)

        return depth


# Learning Upsampling
class GU(nn.Module):
    def __init__(self):
        super(GU, self).__init__()
        self.upsample3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4 * 9, 1, stride=1, padding=0, dilation=1, bias=False)
        )
        self.upsample2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4 * 9, 1, stride=1, padding=0, dilation=1, bias=False)
        )
        self.upsample1 = nn.Sequential(
            nn.Conv2d(16, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 4 * 9, 1, stride=1, padding=0, dilation=1, bias=False)
        )

    def forward(self, depth_0, ref_feature):
        batch, channels, height, width = ref_feature.size()
        if channels == 64:
            upsample_weight = self.upsample3(ref_feature)  # [B,2*2*9,H,W]
            upsample_weight = upsample_weight.view(batch, 1, 9, 2, 2, height, width)
            upsample_weight = torch.softmax(upsample_weight, dim=2)
        elif channels == 32:
            upsample_weight = self.upsample2(ref_feature)  # [B,2*2*9,H,W]
            upsample_weight = upsample_weight.view(batch, 1, 9, 2, 2, height, width)
            upsample_weight = torch.softmax(upsample_weight, dim=2)
        else:
            upsample_weight = self.upsample1(ref_feature)  # [B,2*2*9,H,W]
            upsample_weight = upsample_weight.view(batch, 1, 9, 2, 2, height, width)
            upsample_weight = torch.softmax(upsample_weight, dim=2)

        upsample_weight_final = upsample_weight

        depth = upsample(depth_0, upsample_weight_final)

        return depth


def upsample(x, upsample_weight, scale=2):
    '''
    upsample the x with convex combination
    '''
    batch, _, height, width = x.size()

    replicate_pad = nn.ReplicationPad2d(1)
    x = replicate_pad(x)
    x = F.unfold(x, [3, 3], padding=0)
    x = x.view(batch, -1, 9, 1, 1, height, width)
    x_upsampled = torch.sum(x * upsample_weight, dim=2)
    x_upsampled = x_upsampled.permute(0, 1, 4, 2, 5, 3).contiguous().view(batch, -1, scale * height, scale * width)

    return x_upsampled


def depth_wta(p, depth_values):
    '''Winner take all.'''
    wta_index_map = torch.argmax(p, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_values, 1, wta_index_map).squeeze(1)
    return wta_depth_map