import matplotlib
# %matplotlib inline
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
sys.path.append('../training/')
sys.path.append('../')
from config_training import config
sys.path.append('../preprocessing/')
from step1 import *
# from full_prep import lumTrans
from layers import nms,iou

# import math
# def sigmoid(x):
#     y=1/(1+math.exp(-x))
#     return y
img = np.load('D:/1TJU/硕士/IGRTS/AI_System/prep_result/LIDC-IDRI-00532_clean.npy')
pbb = np.load('D:/1TJU/AAcode/practice2/CADNet/training\classifier/bbox0612/035_pbb.npy')
# all = pbb[:,0]
# pproblist = []
# for i in range(len(all)):
#     pprob = sigmoid(all[i])
#     pproblist.append(pprob)
# pprobnp = np.array(pproblist)
# pbb = pbb[pprobnp>0.5]

pbb = pbb[pbb[:,0]>0]
print(pbb.shape)

pbb = nms(pbb, 0.05)
print(pbb.shape)
box = pbb[0].astype('int')[1:]
print(box)


ax = plt.subplot(1,1,1)
plt.imshow(img[0,box[0]],'gray')
plt.axis('off')
rect = patches.Rectangle((box[2]-box[3],box[1]-box[3]),box[3]*2,box[3]*2,linewidth=2,edgecolor='red',facecolor='none')
ax.add_patch(rect)
plt.show()


