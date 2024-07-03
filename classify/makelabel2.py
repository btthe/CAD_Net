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




filenames = [os.path.join(image_dir, '%s_clean.npy' % idx) for idx in idcs]
lunanames = [f for f in filenames if len(f.split('/')[-1].split('_')[0]) < 20]



a3=0
b2=0
c0=0
d0=0
label_coords = []
for idx in idcs:
    pbb = np.load(os.path.join(bboxpath, idx + '_pbb.npy'))
    pbb = pbb[pbb[:, 0] > 0]
pbb = nms(pbb, 0.1)
newpbb = []
bbs = np.load(os.path.join(bboxpath, '%s_lbb.npy' % idx))
for p in pbb:
    newpbb.append(np.concatenate([[int(idx)], p], 0))
###############################做label
# if self.phase != 'test':
for idx in idcs:
    pbb = np.load(os.path.join(bboxpath, idx + '_pbb.npy'))
    pbb = pbb[pbb[:, 0] >-1]
    pbb = nms(pbb,0.1)
    newpbb = []

    for p in pbb:
        newpbb.append(np.concatenate([[int(idx)], p], 0))
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
                        a3 = a3 + 1
                    elif dist > radiusSquared and dist < 125:
                        p = np.hstack((p, [2]))
                        b2 = b2 + 1
        else:
            p = np.hstack((p, [0]))
            c0 = c0 + 1
        psize = p.shape
        if psize[0] <= 7:
            p = np.hstack((p, [0]))
            d0 = d0 + 1
        if psize[0] >= 7:
            print(p)
            p = p[0:7]
            print('after',p)
        label_coords.append(p)




print (a3,b2,c0,d0)
candidate_np= np.array(label_coords)
df = pandas.DataFrame(candidate_np)
df.columns = {'seriesuid',' confidence','coordZ','coordX','coordY','d','class' }#'probability',
df.to_csv('test.csv', index=False)

















