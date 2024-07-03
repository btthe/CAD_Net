import numpy as np
import torch
from torch.utils.data import Dataset
import os
import time
import collections
import random
# from layers import iou
from scipy.ndimage import zoom
import warnings
from scipy.ndimage.interpolation import rotate
from layers import nms
import math

class DataBowl3Classifier(Dataset):
    def __init__(self, data_dir, bboxpath, split_path, config, phase='train', split_comber=None):
        assert (phase == 'train' or phase == 'val' or phase == 'test')
        self.phase = phase
        # self.max_stride = config['max_stride']
        # self.stride = config['stride']
        sizelim = config['sizelim'] / config['reso']
        sizelim2 = config['sizelim2'] / config['reso']
        sizelim3 = config['sizelim3'] / config['reso']
        self.blacklist = config['blacklist']
        self.isScale = config['aug_scale']
        # self.r_rand = config['r_rand_crop']
        self.augtype = config['augtype']
        self.pad_value = config['pad_value']

        # self.split_comber = split_comber #test用到
        idcs_byte = np.load(split_path)
        idcs = np.array([s.decode('UTF-8') for s in idcs_byte])  # 文件名

        if phase != 'test':
            idcs = [f for f in idcs if (f not in self.blacklist)]

        self.filenames = [os.path.join(data_dir, '%s_clean.npy' % idx) for idx in idcs]
        self.lunanames = [f for f in self.filenames if len(f.split('/')[-1].split('_')[0]) < 20]

        # bboxpath = './bbox/'
################################################
        label_coords = []
        for idx in idcs:
            pbb = np.load(os.path.join(bboxpath, idx + '_pbb.npy'))
            pbb = pbb[pbb[:, 0] > -1]
            pbb = nms(pbb, 0.1)
            newpbb = []
            bbs = np.load(os.path.join(bboxpath, '%s_lbb.npy' % idx))
            for p in pbb:
                newpbb.append(np.concatenate([[int(idx)], p], 0))
###############################做label
            # if self.phase != 'test':
            for p in newpbb:
                if len(bbs) > 0:
                    for bb in bbs:
                        ##########################用直径距离标准生成label
                        if bb[3] != 0:
                            radiusSquared = pow((bb[3] / 2.0), 2.0)
                            x2 = float(p[3])
                            y2 = float(p[4])
                            z2 = float(p[2])
                            dist = math.pow(bb[1] - x2, 2.) + math.pow(bb[2] - y2, 2.) + math.pow(bb[0] - z2, 2.)  # 质心距离计算
                            if dist < radiusSquared:
                                p = np.hstack((p, [3]))
                            elif dist > radiusSquared and dist < 25:
                                p = np.hstack((p, [2]))
                else:
                    p = np.hstack((p, [0]))
                psize = p.shape
                if psize[0] != 7:
                    p = np.hstack((p, [0]))
                label_coords.append(p)
            #####################lbb加入
            if phase == 'train':
                # lbb = np.load(os.path.join(bboxpath, idx + '_lbb.npy'))
                for l in bbs:
                    if len(l) > 0:
                        if l[3] < 30:
                            label_coords.append(np.concatenate([[int(idx)], [100], l, [1]], 0))
                        if l[3] < 10:
                            label_coords.append(np.concatenate([[int(idx)], [100], l, [1]], 0))
                            label_coords.append(np.concatenate([[int(idx)], [100], l, [1]], 0))
                            label_coords.append(np.concatenate([[int(idx)], [100], l, [1]], 0))
                        if l[3] < 5:
                            label_coords.append(np.concatenate([[int(idx)], [100], l, [1]], 0))
                            label_coords.append(np.concatenate([[int(idx)], [100], l, [1]], 0))
######################################################################
        if phase == 'train':
            bboxes_make = []
            for idx in idcs:
                mm = []
                for m in label_coords:
                    if m[0] == int(idx) and m[6] != 3 and m[6] != 2:
                        mm.append(m)
                bboxes_make.append(mm)
            self.sample_bboxes = bboxes_make
        else:
            bboxes_make = []
            for idx in idcs:
                mm = []
                for m in label_coords:
                    if m[0] == int(idx) and m[6] == 2:
                        m[6] = 0
                    if m[0] == int(idx) and m[6] == 3:
                        m[6] = 1
                    if m[0] == int(idx):
                        mm.append(m)
                bboxes_make.append(mm)
            self.sample_bboxes = bboxes_make



        self.bboxes = []
        for i, l in enumerate(bboxes_make):
            if len(l) > 0:
                for t in l:
                    self.bboxes.append([np.concatenate([[i], t])])
                    if t.size != 7:
                        print(1)

        self.bboxes = np.concatenate(self.bboxes, axis=0)
            # np.random.shuffle(self.bboxes)
        self.crop = Pad_crop(config)

    def __getitem__(self, idx, split=None):
        # print ('get item', idx)
        t = time.time()
        np.random.seed(int(str(t % 1)[2:7]))

        bbox = self.bboxes[idx]
        # print(bbox)
        filename = self.filenames[int(bbox[0])]
        bboxes = self.sample_bboxes[int(bbox[0])]
        # print(bboxes)
        imgs = np.load(filename)
        isScale = self.augtype['scale'] and (self.phase == 'train')

        if self.phase == 'train':
            sample, target, bboxes = self.crop(imgs, bbox[3:7], bboxes, isScale)
            sample, target, bboxes = augment(sample, target, bboxes,
                                                        ifflip=self.augtype['flip'],
                                                        ifrotate=self.augtype['rotate'],
                                                        ifswap=self.augtype['swap'])
        else:
            sample, target, bboxes = self.crop(imgs, bbox[3:7], bboxes, isScale=False)

        if self.phase != 'test':
            label = np.array([bbox[7]]).astype(np.float)
        else:
            test_label = bbox[1:8]
        sample = (sample.astype(np.float32) - 128) / 128
        if self.phase != 'test':
            ss = torch.from_numpy(sample)
            return torch.from_numpy(sample), torch.from_numpy(label)
        else:
            ss = torch.from_numpy(sample)
            return torch.from_numpy(sample), test_label

    def __len__(self):
            return len(self.bboxes)


class Pad_crop(object):
    def __init__(self, config):
        self.crop_size = config['crop_size'] #[48, 48, 48]
        self.bound_size = config['bound_size'] #12 距斑块边界有12个像素以上的边缘
        # self.stride = config['stride']  #4
        self.pad_value = config['pad_value']  #0?

    def __call__(self, imgs, target, bboxes, isScale=False):
        if isScale:
            radiusLim = [3., 35.]  #半径限制
            scaleLim = [0.75, 1.25]
            scaleRange = [np.min([np.max([(radiusLim[0] / target[3]), scaleLim[0]]), 1])
                , np.max([np.min([(radiusLim[1] / target[3]), scaleLim[1]]), 1])]
            scale = np.random.rand() * (scaleRange[1] - scaleRange[0]) + scaleRange[0]
            crop_size = (np.array(self.crop_size).astype('float') / scale).astype('int')
        else:
            crop_size = self.crop_size
        bound_size = self.bound_size
        target = np.copy(target)
        # print(target)
        bboxes = np.copy(bboxes)

        start = []
        for i in range(3):
            r = target[3] / 2
            s = np.floor(target[i] - r) + 1 - bound_size
            e = np.ceil(target[i] + r) + 1 + bound_size - crop_size[i]
            if s > e:
                start.append(np.random.randint(e, s))  # !
            else:
                start.append(int(target[i]) - crop_size[i] / 2 + np.random.randint(-bound_size / 2, bound_size / 2))


        pad = []
        pad.append([0, 0])
        start = [int(x) for x in start]
        for i in range(3):
            leftpad = max(0, -start[i])
            rightpad = max(0, start[i] + crop_size[i] - imgs.shape[i + 1])
            pad.append([leftpad, rightpad])
        crop = imgs[:,
               max(start[0], 0):min(start[0] + crop_size[0], imgs.shape[1]),
               max(start[1], 0):min(start[1] + crop_size[1], imgs.shape[2]),
               max(start[2], 0):min(start[2] + crop_size[2], imgs.shape[3])]
        crop = np.pad(crop, pad, 'constant', constant_values=self.pad_value)

        for i in range(3):
            target[i] = target[i] - start[i]
        for i in range(len(bboxes)):
            for j in range(3):
                bboxes[i][j+2] = bboxes[i][j+2] - start[j]

        if isScale:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                crop = zoom(crop, [1, scale, scale, scale], order=1)
            newpad = self.crop_size[0] - crop.shape[1:][0]
            if newpad < 0:
                crop = crop[:, :-newpad, :-newpad, :-newpad]
            elif newpad > 0:
                pad2 = [[0, 0], [0, newpad], [0, newpad], [0, newpad]]
                crop = np.pad(crop, pad2, 'constant', constant_values=self.pad_value)
            for i in range(4):
                target[i] = target[i] * scale
            for i in range(len(bboxes)):
                for j in range(4):
                    bboxes[i][j+2] = bboxes[i][j+2] * scale
        return crop, target, bboxes
def augment(sample, target, bboxes, ifflip=True, ifrotate=True, ifswap=True):
    #                     angle1 = np.random.rand()*180
    if ifrotate:
        validrot = False
        counter = 0
        while not validrot:
            newtarget = np.copy(target)
            angle1 = np.random.rand() * 180   #随机角度
            size = np.array(sample.shape[2:4]).astype('float')
            rotmat = np.array([[np.cos(angle1 / 180 * np.pi), -np.sin(angle1 / 180 * np.pi)],
                               [np.sin(angle1 / 180 * np.pi), np.cos(angle1 / 180 * np.pi)]])
            newtarget[1:3] = np.dot(rotmat, target[1:3] - size / 2) + size / 2
            if np.all(newtarget[:3] > target[3]) and np.all(newtarget[:3] < np.array(sample.shape[1:4]) - newtarget[3]):
                validrot = True
                target = newtarget
                sample = rotate(sample, angle1, axes=(2, 3), reshape=False)
                # coord = rotate(coord, angle1, axes=(2, 3), reshape=False)
                for box in bboxes:
                    box[3:5] = np.dot(rotmat, box[3:5] - size / 2) + size / 2
            else:
                counter += 1
                if counter == 3:
                    break
    if ifswap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))
            # coord = np.transpose(coord, np.concatenate([[0], axisorder + 1]))
            target[:3] = target[:3][axisorder]
            bboxes[:, 2:5] = bboxes[:, 2:5][:, axisorder]

    if ifflip:
        #         flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])   #返回一个地址连续的数组
        # coord = np.ascontiguousarray(coord[:, ::flipid[0], ::flipid[1], ::flipid[2]])
        for ax in range(3):
            if flipid[ax] == -1:
                target[ax] = np.array(sample.shape[ax + 1]) - target[ax]
                bboxes[:, ax+2] = np.array(sample.shape[ax + 1]) - bboxes[:, ax+2]
    return sample, target, bboxes

def collate(batch):
    if torch.is_tensor(batch[0]):
        return [b.unsqueeze(0) for b in batch]
    elif isinstance(batch[0], np.ndarray):
        return batch
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], collections.Iterable):
        transposed = zip(*batch)
        return [collate(samples) for samples in transposed]