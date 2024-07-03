import numpy as np
from layers import nms, iou, acc
import time
import multiprocessing as mp

save_dir = 'E:/PycharmProjects/grt/training/detector/results/res18-20180928-190822/bbox/'#results/ma_offset40_res_n6_100-1/
pbb = np.load(save_dir + 'allpbb.npy')
lbb = np.load(save_dir + 'alllbb.npy')

conf_th = [-1, 0, 1] #置信度
nms_th = [0.3, 0.5, 0.7]#负样本非极大值抑制
detect_th = [0.2, 0.3] #与lable重合度

# conf_th = [-1] #置信度
# nms_th = [0.05]#负样本非极大值抑制
# detect_th = [0.05] #与lable重合度

def mp_get_pr(conf_th, nms_th, detect_th, num_procs=64):
    start_time = time.time()

    num_samples = len(pbb)
    split_size = int(np.ceil(float(num_samples) / num_procs))
    num_procs = int(np.ceil(float(num_samples) / split_size))

    manager = mp.Manager()
    tp = manager.list(range(num_procs))
    fp = manager.list(range(num_procs))
    fn = manager.list(range(num_procs))
    procs = []
    for pid in range(num_procs):
        proc = mp.Process(
            target=get_pr,
            args=(
                pbb[pid * split_size:min((pid + 1) * split_size, num_samples)],
                lbb[pid * split_size:min((pid + 1) * split_size, num_samples)],
                conf_th, nms_th, detect_th, pid, tp, fp, fn))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()

    tp = np.sum(tp)
    fp = np.sum(fp)
    fn = np.sum(fn)

    end_time = time.time()
    print('conf_th %1.1f, nms_th %3.2f, detect_th %3.2f, tp %d, fp %d, fn %d, recall %f, time %3.2f' % (
        conf_th, nms_th, detect_th, len(tp), len(fp), len(fn), float(len(tp)) / (len(fn)+len(tp)), end_time - start_time))#召回率已改


def get_pr(pbb, lbb, conf_th, nms_th, detect_th, pid, tp_list, fp_list, p_list):
    # tp, fp, p = 0, 0, 0
    tp = []
    fp = []
    fn = []
    # print(pbb.shape)
    for i in range(len(pbb)):
        tpi, fpi, fni,lbb_n = acc(pbb[i], lbb[i], conf_th, nms_th, detect_th)
        tp += tpi
        fp += fpi
        fn += fni
    tp_list[pid] = tp
    fp_list[pid] = fp
    p_list[pid] = fn




if __name__ == '__main__':
    for ct in conf_th:
        for nt in nms_th:
            for dt in detect_th:
                mp_get_pr(ct, nt, dt)