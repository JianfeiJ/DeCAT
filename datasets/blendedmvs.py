from torch.utils.data import Dataset
from datasets.data_io import *
import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms as T
import random


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, img_wh=(768, 576), robust_train=True):

        super(MVSDataset, self).__init__()
        self.stages = 4
        self.datapath = datapath
        self.mode = mode
        self.listfile = listfile
        self.robust_train = robust_train
        assert self.mode in ['train', 'val', 'all'], \
            'split must be either "train", "val" or "all"!'

        self.img_wh = img_wh
        if img_wh is not None:
            assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, \
                'img_wh must both be multiples of 32!'
        self.nviews = nviews
        self.build_metas()

        # self.color_augment = T.ColorJitter(brightness=0.5, contrast=0.5)

    def build_metas(self):
        self.metas = []
        with open(self.listfile) as f:
            self.scans = [line.rstrip() for line in f.readlines()]
        for scan in self.scans:
            with open(os.path.join(self.datapath, scan, "cams/pair.txt")) as f:
                num_viewpoint = int(f.readline())
                for _ in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) >= self.nviews - 1:
                        self.metas += [(scan, ref_view, src_views)]

    def read_cam_file(self, scan, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        depth_min = float(lines[11].split()[0])
        depth_max = float(lines[11].split()[-1])

        return intrinsics, extrinsics, depth_min, depth_max

    def read_depth_mask(self, filename, depth_min, depth_max):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)

        depth = np.squeeze(depth, 2)

        mask = (depth >= depth_min) & (depth <= depth_max)
        mask = mask.astype(np.float32)
        if self.img_wh is not None:
            depth = cv2.resize(depth, self.img_wh,
                               interpolation=cv2.INTER_NEAREST)
        h, w = depth.shape
        depth_ms = {}
        mask_ms = {}

        for i in range(4):
            depth_cur = cv2.resize(depth, (w // (2 ** i), h // (2 ** i)), interpolation=cv2.INTER_NEAREST)
            mask_cur = cv2.resize(mask, (w // (2 ** i), h // (2 ** i)), interpolation=cv2.INTER_NEAREST)

            depth_ms[f"stage_{i}"] = depth_cur
            mask_ms[f"stage_{i}"] = mask_cur

        return depth_ms, mask_ms

    def read_img(self, filename):
        img = Image.open(filename)
        # if self.mode == 'train':
        #     img = self.color_augment(img)
        np_img = np.array(img, dtype=np.float32) / 255.
        if self.img_wh is not None:
            np_img = cv2.resize(np_img, self.img_wh,
                                interpolation=cv2.INTER_LINEAR)
        h, w, _ = np_img.shape
        np_img_ms = {
            "stage_3": cv2.resize(np_img, (w // 8, h // 8), interpolation=cv2.INTER_LINEAR),
            "stage_2": cv2.resize(np_img, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR),
            "stage_1": cv2.resize(np_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR),
            "stage_0": np_img
        }
        return np_img_ms

    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta

        if self.robust_train:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.nviews - 1)
            view_ids = [ref_view] + [src_views[i] for i in index]

        else:
            view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs_0 = []
        imgs_1 = []
        imgs_2 = []
        imgs_3 = []
        mask = None
        depth = None
        depth_min = None
        depth_max = None
        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/blended_images/{:0>8}.jpg'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            imgs = self.read_img(img_filename)
            imgs_0.append(imgs['stage_0'])
            imgs_1.append(imgs['stage_1'])
            imgs_2.append(imgs['stage_2'])
            imgs_3.append(imgs['stage_3'])

            # here, the intrinsics from file is already adjusted to the downsampled size of feature 1/4H0 * 1/4W0
            intrinsics, extrinsics, depth_min, depth_max = self.read_cam_file(scan, proj_mat_filename)
            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 0.125
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_3.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_2.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_1.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_0.append(proj_mat)

            if i == 0:  # reference view
                depth, mask = self.read_depth_mask(depth_filename, depth_min, depth_max)
                for l in range(self.stages):
                    mask[f'stage_{l}'] = np.expand_dims(mask[f'stage_{l}'], 2)
                    mask[f'stage_{l}'] = mask[f'stage_{l}'].transpose([2, 0, 1])
                    depth[f'stage_{l}'] = np.expand_dims(depth[f'stage_{l}'], 2)
                    depth[f'stage_{l}'] = depth[f'stage_{l}'].transpose([2, 0, 1])

        # imgs: N*3*H0*W0, N is number of images
        imgs_0 = np.stack(imgs_0).transpose([0, 3, 1, 2])
        imgs_1 = np.stack(imgs_1).transpose([0, 3, 1, 2])
        imgs_2 = np.stack(imgs_2).transpose([0, 3, 1, 2])
        imgs_3 = np.stack(imgs_3).transpose([0, 3, 1, 2])
        imgs = {}
        imgs['stage_0'] = imgs_0
        imgs['stage_1'] = imgs_1
        imgs['stage_2'] = imgs_2
        imgs['stage_3'] = imgs_3

        # proj_matrices: N*4*4
        proj_matrices_0 = np.stack(proj_matrices_0)
        proj_matrices_1 = np.stack(proj_matrices_1)
        proj_matrices_2 = np.stack(proj_matrices_2)
        proj_matrices_3 = np.stack(proj_matrices_3)
        proj = {}
        proj['stage_3'] = proj_matrices_3
        proj['stage_2'] = proj_matrices_2
        proj['stage_1'] = proj_matrices_1
        proj['stage_0'] = proj_matrices_0

        # data is numpy array
        return {"imgs": imgs,  # [N, 3, H, W]
                "proj_matrices": proj,  # [N,4,4]
                "depth": depth,  # [1, H, W]
                "depth_min": depth_min,  # scalar
                "depth_max": depth_max,  # scalar
                "mask": mask}  # [1, H, W]
