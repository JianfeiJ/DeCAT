import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .patchmatch import *
from .FET import GFF
import torch
import torch.nn as nn
import torch.nn.functional as F
from .dcn import DCN, DCNv2
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        # [B,8,H,W]
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)
        # [B,16,H/2,W/2]
        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)
        # [B,32,H/4,W/4]
        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv7 = ConvBnReLU(32, 32, 3, 1, 1)
        # [B,64,H/8,W/8]
        self.conv8 = ConvBnReLU(32, 64, 5, 2, 2)
        self.conv9 = ConvBnReLU(64, 64, 3, 1, 1)
        self.conv10 = ConvBnReLU(64, 64, 3, 1, 1)

        self.output1 = nn.Conv2d(64, 64, 1, bias=False)
        self.inner1 = nn.Conv2d(32, 64, 1, bias=True)
        self.inner2 = nn.Conv2d(16, 64, 1, bias=True)
        self.output2 = nn.Conv2d(64, 32, 1, bias=False)
        self.output3 = nn.Conv2d(64, 16, 1, bias=False)

    def forward(self, x):
        output_feature = {}

        conv1 = self.conv1(self.conv0(x))
        conv4 = self.conv4(self.conv3(self.conv2(conv1)))

        conv7 = self.conv7(self.conv6(self.conv5(conv4)))
        conv10 = self.conv10(self.conv9(self.conv8(conv7)))

        output_feature['stage_3'] = self.output1(conv10)

        intra_feat = F.interpolate(conv10, scale_factor=2, mode="bilinear") + self.inner1(conv7)
        del conv7, conv10
        output_feature['stage_2'] = self.output2(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear") + self.inner2(conv4)
        del conv4
        output_feature['stage_1'] = self.output3(intra_feat)

        del intra_feat

        return output_feature


class FeatureNet_DCN(nn.Module):
    def __init__(self):
        super(FeatureNet_DCN, self).__init__()

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        # [B,8,H,W]
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)
        # [B,16,H/2,W/2]
        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)
        # [B,32,H/4,W/4]
        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.conv7 = ConvBnReLU(32, 32, 3, 1, 1)
        # [B,64,H/8,W/8]
        self.conv8 = ConvBnReLU(32, 64, 5, 2, 2)
        self.conv9 = ConvBnReLU(64, 64, 3, 1, 1)
        self.conv10 = ConvBnReLU(64, 64, 3, 1, 1)

        # self.output1 = nn.Conv2d(64, 64, 1, bias=False)
        self.output1 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            DCN(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DCN(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DCN(64, 64, 3, 1, 1),
        )
        self.inner1 = nn.Conv2d(32, 64, 1, bias=True)
        self.inner2 = nn.Conv2d(16, 64, 1, bias=True)
        self.output2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            DCN(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DCN(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DCN(64, 32, 3, 1, 1),
        )
        self.output3 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            DCN(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DCN(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            DCN(64, 16, 3, 1, 1),
        )
        # self.output2 = nn.Conv2d(64, 32, 1, bias=False)
        # self.output3 = nn.Conv2d(64, 16, 1, bias=False)

    def forward(self, x):
        output_feature = {}

        conv1 = self.conv1(self.conv0(x))
        conv4 = self.conv4(self.conv3(self.conv2(conv1)))

        conv7 = self.conv7(self.conv6(self.conv5(conv4)))
        conv10 = self.conv10(self.conv9(self.conv8(conv7)))

        output_feature['stage_3'] = self.output1(conv10)

        intra_feat = F.interpolate(conv10, scale_factor=2, mode="bilinear", align_corners=True) + self.inner1(conv7)
        del conv7, conv10
        output_feature['stage_2'] = self.output2(intra_feat)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="bilinear", align_corners=True) + self.inner2(conv4)
        del conv4
        output_feature['stage_1'] = self.output3(intra_feat)

        del intra_feat

        return output_feature


class DWTMVSNet(nn.Module):
    def __init__(self, patchmatch_interval_scale=[0.005, 0.0125, 0.025], propagation_range=[6, 4, 2],
                 patchmatch_iteration=[1, 2, 2], patchmatch_num_sample=[8, 8, 16], propagate_neighbors=[0, 8, 16],
                 evaluate_neighbors=[9, 9, 9], dcn=True, upsampling='guided', double_weight=True, base_channels=16,
                 block_type='quadtree', topks=[2, 1, 1], nhead=8, d_model=64, cross_attention=['self', 'cross'] * 4):
        super(DWTMVSNet, self).__init__()
        self.stages = 4
        self.dcn = dcn
        self.upsampling = upsampling
        if self.dcn:
            self.feature = FeatureNet_DCN()
        else:
            self.feature = FeatureNet()

        self.double_weight = double_weight

        self.patchmatch_num_sample = patchmatch_num_sample

        num_features = [8, 16, 32, 64]

        self.propagate_neighbors = propagate_neighbors
        self.evaluate_neighbors = evaluate_neighbors
        # number of groups for group-wise correlation
        self.G = [4, 8, 8]

        self.GFF = GFF(base_channels, block_type, topks, nhead, d_model, cross_attention)
        self.GU = GU()
        self.Residual = Residual()

        for l in range(self.stages - 1):

            if l == 2:
                patchmatch = PatchMatch(True, propagation_range[l], patchmatch_iteration[l],
                                        patchmatch_num_sample[l], patchmatch_interval_scale[l],
                                        num_features[l + 1], self.G[l], self.propagate_neighbors[l], l + 1,
                                        evaluate_neighbors[l], double_weight)
            else:
                patchmatch = PatchMatch(False, propagation_range[l], patchmatch_iteration[l],
                                        patchmatch_num_sample[l], patchmatch_interval_scale[l],
                                        num_features[l + 1], self.G[l], self.propagate_neighbors[l], l + 1,
                                        evaluate_neighbors[l], double_weight)
            setattr(self, f'patchmatch_{l + 1}', patchmatch)

    def forward(self, imgs, proj_matrices, depth_min, depth_max):

        imgs_0 = torch.unbind(imgs['stage_0'], 1)
        imgs_1 = torch.unbind(imgs['stage_1'], 1)
        imgs_2 = torch.unbind(imgs['stage_2'], 1)
        imgs_3 = torch.unbind(imgs['stage_3'], 1)
        del imgs

        self.imgs_0_ref = imgs_0[0]
        self.imgs_1_ref = imgs_1[0]
        self.imgs_2_ref = imgs_2[0]
        self.imgs_3_ref = imgs_3[0]
        del imgs_1, imgs_2, imgs_3

        self.proj_matrices_0 = torch.unbind(proj_matrices['stage_0'].float(), 1)
        self.proj_matrices_1 = torch.unbind(proj_matrices['stage_1'].float(), 1)
        self.proj_matrices_2 = torch.unbind(proj_matrices['stage_2'].float(), 1)
        self.proj_matrices_3 = torch.unbind(proj_matrices['stage_3'].float(), 1)
        del proj_matrices

        assert len(imgs_0) == len(self.proj_matrices_0), "Different number of images and projection matrices"

        # step 1. Multi-scale feature extraction
        features = []
        # 输入图片尺寸512X640
        for img in imgs_0:
            output_feature = self.feature(img)
            features.append(output_feature)
        del imgs_0

        # features = self.GFF(features)
        ref_feature, src_features = features[0], features[1:]

        depth_min = depth_min.float()
        depth_max = depth_max.float()

        # step 2. Learning-based patchmatch
        depth = None
        view_weights = None
        depth_patchmatch = {}
        prob_volumes = {}
        depth_values = {}
        refined_depth = {}

        for l in reversed(range(1, self.stages)):
            src_features_l = [src_fea[f'stage_{l}'] for src_fea in src_features]
            projs_l = getattr(self, f'proj_matrices_{l}')
            ref_proj, src_projs = projs_l[0], projs_l[1:]
            # upsample_weight = self.upsample()

            if l > 1:
                depth, prob_volume, view_weights, depth_value = getattr(self, f'patchmatch_{l}')(ref_feature[f'stage_{l}'],
                                                                                    src_features_l,
                                                                                    ref_proj, src_projs,
                                                                                    depth_min, depth_max, depth=depth,
                                                                                    img=getattr(self, f'imgs_{l}_ref'),
                                                                                    view_weights=view_weights)
            else:
                depth, prob_volume, _, depth_value = getattr(self, f'patchmatch_{l}')(ref_feature[f'stage_{l}'], src_features_l,
                                                                         ref_proj, src_projs,
                                                                         depth_min, depth_max, depth=depth,
                                                                         img=getattr(self, f'imgs_{l}_ref'),
                                                                         view_weights=view_weights)

            del src_features_l, ref_proj, src_projs, projs_l

            depth_patchmatch[f'stage_{l}'] = depth
            prob_volumes[f'stage_{l}'] = prob_volume
            depth_values[f'stage_{l}']= depth_value

            depth = depth[-1].detach()
            if l > 1:
                # upsampling the depth map and pixel-wise view weight for next stage
                depth = F.interpolate(depth,
                                      scale_factor=2, mode='bilinear', align_corners=True)
                # depth = self.GU(depth, ref_feature[f'stage_{l}'])
                view_weights = F.interpolate(view_weights,
                                             scale_factor=2, mode='bilinear', align_corners=True)

        # step 3. Refinement
        if self.upsampling == 'guided':
            depth = self.GU(depth, ref_feature["stage_1"])
        elif self.upsampling == 'residual':
            depth = self.Residual(self.imgs_0_ref, depth, depth_min, depth_max)
        elif self.upsampling == 'bilinear':
            depth = F.interpolate(depth, scale_factor=2, mode='bilinear', align_corners=True)

        refined_depth['stage_0'] = depth

        del depth, ref_feature, src_features

        if self.training:
            return {"refined_depth": refined_depth,
                    "depth_patchmatch": depth_patchmatch,
                    "prob_volume": prob_volumes,
                    "depth_values": depth_values,
                    }

        else:
            photometric_confidence = torch.max(prob_volume, dim=1)[0]

            # num_depth = self.patchmatch_num_sample[0]
            # score_sum4 = 4 * F.avg_pool3d(F.pad(score.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1,
            #                               padding=0).squeeze(1)
            # # [B, 1, H, W]
            # depth_index = depth_regression(score, depth_values=torch.arange(num_depth, device=score.device,
            #                                                                 dtype=torch.float)).long()
            # depth_index = torch.clamp(depth_index, 0, num_depth - 1)
            # photometric_confidence = torch.gather(score_sum4, 1, depth_index)
            photometric_confidence = F.interpolate(photometric_confidence,
                                                   scale_factor=2, mode='bilinear', align_corners=True).squeeze(1)

            return {"refined_depth": refined_depth,
                    "depth_patchmatch": depth_patchmatch,
                    "photometric_confidence": photometric_confidence,
                    "prob_volume": prob_volumes,
                    "depth_values": depth_values,
                    }


def l1_loss(depth_patchmatch, refined_depth, depth_gt, mask, depth_values):
    stage = 4

    loss = 0
    for l in range(1, stage):
        depth_gt_l = depth_gt[f'stage_{l}']
        mask_l = mask[f'stage_{l}'] > 0.5
        depth2 = depth_gt_l[mask_l]

        depth_patchmatch_l = depth_patchmatch[f'stage_{l}']
        for i in range(len(depth_patchmatch_l)):
            depth1 = depth_patchmatch_l[i][mask_l]
            loss = loss + F.smooth_l1_loss(depth1, depth2, reduction='mean')

    l = 0
    depth_refined_l = refined_depth[f'stage_{l}']
    depth_gt_l = depth_gt[f'stage_{l}']
    mask_l = mask[f'stage_{l}'] > 0.5

    depth1 = depth_refined_l[mask_l]
    depth2 = depth_gt_l[mask_l]
    loss = loss + F.smooth_l1_loss(depth1, depth2, reduction='mean')

    return loss


def entropy_loss(depth_patchmatch, refined_depth, depth_gt, mask, depth_values, prob_volume):
    stage = 4
    total_entropy = 0
    total_depth = 0
    total_loss = 0
    for l in range(1, stage):
        depth_gt_l = depth_gt[f'stage_{l}']
        prob_volume_l = prob_volume[f'stage_{l}']
        depth_values_l = depth_values[f'stage_{l}']
        mask_l = mask[f'stage_{l}'] > 0.5
        depth_mask = depth_gt_l[mask_l]

        depth_patchmatch_l = depth_patchmatch[f'stage_{l}']
        entropy_weight = 2.0

        for i in range(len(depth_patchmatch_l)):
            entro_loss, depth_entropy = entropy(prob_volume_l[i], depth_gt_l, mask_l, depth_values_l[i])
            depth_entropy = depth_entropy.unsqueeze(1)
            entro_loss = entro_loss * entropy_weight
            # depth1 = depth_patchmatch_l[i][mask_l]
            depth_loss = F.smooth_l1_loss(depth_entropy[mask_l], depth_mask, reduction='mean')
            total_entropy += entro_loss
            total_depth += depth_loss
            total_loss += entro_loss




    l = 0
    depth_loss_final = 0

    depth_refined_l = refined_depth[f'stage_{l}']
    depth_gt_l = depth_gt[f'stage_{l}']
    mask_l = mask[f'stage_{l}'] > 0.5

    depth1 = depth_refined_l[mask_l]
    depth2 = depth_gt_l[mask_l]
    depth_loss_final += F.smooth_l1_loss(depth1, depth2, reduction='mean')
    total_loss += depth_loss_final
    total_depth += depth_loss_final

    return total_loss, total_depth, total_entropy


def entropy(prob_volume, depth_gt, mask, depth_value, return_prob_map=False):
    # from AA
    mask_true = mask.squeeze(1)
    depth_gt = depth_gt.squeeze(1)
    valid_pixel_num = torch.sum(mask_true, dim=[1, 2]) + 1e-6

    shape = depth_gt.shape  # B,H,W

    depth_num = depth_value.shape[1]
    if len(depth_value.shape) < 3:
        depth_value_mat = depth_value.repeat(shape[1], shape[2], 1, 1).permute(2, 3, 0, 1)  # B,N,H,W
    else:
        depth_value_mat = depth_value

    gt_index_image = torch.argmin(torch.abs(depth_value_mat - depth_gt.unsqueeze(1)), dim=1)

    gt_index_image = torch.mul(mask_true, gt_index_image.type(torch.float))
    gt_index_image = torch.round(gt_index_image).type(torch.long).unsqueeze(1)  # B, 1, H, W

    # gt index map -> gt one hot volume (B x 1 x H x W )
    gt_index_volume = torch.zeros(shape[0], depth_num, shape[1], shape[2]).type(mask_true.type()).scatter_(1,
                                                                                                           gt_index_image,
                                                                                                           1)

    # cross entropy image (B x D X H x W)
    cross_entropy_image = -torch.sum(gt_index_volume * torch.log(prob_volume + 1e-6), dim=1).squeeze(1)  # B, 1, H, W

    # masked cross entropy loss
    masked_cross_entropy_image = torch.mul(mask_true, cross_entropy_image)  # valid pixel
    masked_cross_entropy = torch.sum(masked_cross_entropy_image, dim=[1, 2])

    masked_cross_entropy = torch.mean(masked_cross_entropy / valid_pixel_num)  # Origin use sum : aggregate with batch
    # winner-take-all depth map
    wta_index_map = torch.argmax(prob_volume, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_value_mat, 1, wta_index_map).squeeze(1)

    if return_prob_map:
        photometric_confidence = torch.max(prob_volume, dim=1)[0]  # output shape dimension B * H * W
        return masked_cross_entropy, wta_depth_map, photometric_confidence
    return masked_cross_entropy, wta_depth_map
