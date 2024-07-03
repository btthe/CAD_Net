import numpy as np
import os
from config_training import config as config_training
from layers import nms, iou
import math
import  pandas

split_path = 'value_luna8.npy'
blacklist  = [ '417','077','188','876','057','087','130','468']
image_dir =  config_training['preprocess_result_path']
bboxpath = './bboxsenet84/'

idcs_byte = np.load(split_path)
idcs = np.array([s.decode('UTF-8') for s in idcs_byte])  # 文件名

# if phase != 'test':
idcs = [f for f in idcs if (f not in blacklist)]
filenames = [os.path.join(image_dir, '%s_clean.npy' % idx) for idx in idcs]
print(filenames)
lunanames = [f for f in filenames if len(f.split('/')[-1].split('_')[0]) < 20]

idcs = [f.split('-')[0] for f in idcs]

label_coords1 = []
label_coords=[]
label_pos = []
label_neg = []
for idx in idcs:
    pbb = np.load(os.path.join(bboxpath, idx + '_pbb.npy'))
    pbb = pbb[pbb[:, 0] >-1]
    pbb = nms(pbb,0.1)
    newpbb = []

    for p in pbb:
        newpbb.append(np.concatenate([[int(idx)], p], 0))
    lbb = np.load(os.path.join(bboxpath, idx + '_lbb.npy'))
    for l in lbb:
        if len(l) > 0:
            if l[3]<35:
                newpbb.append(np.concatenate([[int(idx)], [100], l], 0))
            if l[3]<20:
                newpbb.append(np.concatenate([[int(idx)], [100], l], 0))
                newpbb.append(np.concatenate([[int(idx)], [100], l], 0))
            if l[3]<10:
                newpbb.append(np.concatenate([[int(idx)], [100], l], 0))
                newpbb.append(np.concatenate([[int(idx)], [100], l], 0))
                newpbb.append(np.concatenate([[int(idx)], [100], l], 0))
                newpbb.append(np.concatenate([[int(idx)], [100], l], 0))

    bbs = np.load(os.path.join(bboxpath, '%s_lbb.npy' % idx))

    for p in newpbb:
        if len(bbs) > 0:
            for bb in bbs:
                # score = iou(p[2:6], bb)
                # if score > 0.05:
                #     p = np.hstack((p, [1]))
##########################用直径距离标准生成label
                if bb[3] != 0:
                    radiusSquared = pow((bb[3] / 2.0), 2.0)
                    x2 = float(p[3])
                    y2 = float(p[4])
                    z2 = float(p[2])
                    dist = math.pow(bb[1] - x2, 2.) + math.pow(bb[ 2] - y2, 2.) + math.pow(bb[0] - z2,2.)  # 质心距离计算
                    if dist < radiusSquared:
                        p = np.hstack((p, [1]))
        else:
            p = np.hstack((p, [0]))
        psize = p.shape
        if psize[0] != 7:
            p = np.hstack((p, [2]))
        label_coords.append(p)



candidate_np= np.array(label_coords)
df = pandas.DataFrame(candidate_np)
df.columns = {'seriesuid',' confidence','coordZ','coordX','coordY','d','class' }#'probability',
df.to_csv('test.csv', index=False)

# for l in label_coords:
#     if l[6]==1:
#         label_pos.append(l)
#     else:
#         label_neg.append(l)
#         print(l[2:6])

# label_pos=[]
# label_neg=[]
# for l in label_coords:
#     if l[6] == 1:
#         label_pos.append(l)
#     else:
#         label_neg.append(l)
# sample_posbboxes = label_pos
# sample_negbboxes = label_neg
#
# posbboxes = []
# for i, l in enumerate(label_pos):
#     if len(l) > 0:
#         if l[5] > 3:
#             posbboxes.append([np.concatenate([[i], l])])
#         if l[5] > 10:
#             posbboxes += [[np.concatenate([[i], l])]] * 2   # 数据增强，2倍
#         if l[5] > 20:
#             posbboxes += [[np.concatenate([[i], l])]] * 4 * 4
# posbboxes = np.concatenate(posbboxes, axis=0)
#
# bbox = posbboxes[0]
#
# filename = filenames[int(bbox[0])]
# imgs = np.load(filename)
# bboxes = sample_posbboxes[int(bbox[0])]
# print(1)
#
# negbboxes = []
# for i, l in enumerate(label_neg):
#     if len(l) > 0:
#         negbboxes.append([np.concatenate([[i], l])])
# negbboxes = np.concatenate(negbboxes, axis=0)
# print(1)


# bboxes = []
# for i, l in enumerate(label_coords):
#     if len(l) > 0:
#         bboxes.append([np.concatenate([[i], l])])
# bboxes = np.concatenate(bboxes, axis=0)
# print(1)

# mlabels = []
# for idx in idcs:
#     mm = []
#     for m in label_coords:
#         if m[0]==int(idx):
#             mm.append(m)
#     mlabels.append(mm)
# # print(1)
# sample_bboxes = mlabels

# labels=[]
# for idx in idcs:
#     l = np.load(os.path.join(bboxpath, '%s_lbb.npy' % idx))
#     if np.all(l == 0):
#         l = np.array([])
#     labels.append(l)
# sample_bboxes = labels
# bboxes = []
# for i, l in enumerate(mlabels):
#     if len(l) > 0:
#         for t in l: #t[5]结节直径
#             bboxes.append([np.concatenate([[i], t])])
#
# bboxes = np.concatenate(bboxes, axis=0)
#
# #########################索引有问题
# for i in range(3):
#     bbox = bboxes[i]
#     print(bbox[3:7])
#     filename = filenames[int(bbox[0])]
#     bbboxes = sample_bboxes[int(bbox[0])]
#     imgs = np.load(filename)
#
# print(1)























