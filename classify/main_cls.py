import argparse
import os
import time
import numpy as np
import data_cls_r
from importlib import import_module
import shutil
from utils import *
import sys
from torch import nn
import pandas as pd
sys.path.append('../')
# from split_combine import SplitComb
import pandas

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from config_training import config as config_training


from tensorboardX import SummaryWriter
from torch.nn.functional import binary_cross_entropy
import math

parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Classifier')
parser.add_argument('--model', '-m', metavar='MODEL', default='seres_classifier',
                    help='model')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')####CPU线程数
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')  #改过  # 训练整批数据多少次
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=2, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--save-freq', default='1', type=int, metavar='S',
                    help='save frequency')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
#E:/PycharmProjects/grt - 2/training/classifier/results/res-pre-20181125-144714/023.ckpt
parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')
##############################################################################
parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
###############################################################################
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
parser.add_argument('--gpu', default='0', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=1, type=int, metavar='N',
                    help='number of gpu for test')#gpu数量要改!

def main():
    global args
    args = parser.parse_args()

    torch.manual_seed(0)  # 设置生成随机数的种子
    torch.cuda.set_device(0)

    model = import_module(args.model)
    config, net = model.get_model()
    start_epoch = args.start_epoch
    save_dir = args.save_dir

    if args.resume: #继续训练读取ckpt
        checkpoint = torch.load(args.resume)
        #####多gpu转到单gpu
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        #####################
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        if not save_dir:
            save_dir = checkpoint['save_dir']
        else:
            save_dir = os.path.join('results', save_dir)
        net.load_state_dict(new_state_dict)
    else:
        if start_epoch == 0:
            start_epoch = 1
        if not save_dir:
            exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime())
            save_dir = os.path.join('results', args.model + '-' + exp_id) #模型保存路径
        else:
            save_dir = os.path.join('results', save_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = os.path.join(save_dir, 'log')
    if args.test != 1:
        sys.stdout = Logger(logfile)
        pyfiles = [f for f in os.listdir('./') if f.endswith('.py')]
        for f in pyfiles:
            shutil.copy(f, os.path.join(save_dir, f))
    n_gpu = setgpu(args.gpu)
    args.n_gpu = n_gpu
    net = net.cuda()
    # loss = loss.cuda()
    cudnn.benchmark = True #增加程序的运行效率
    datadir = config_training['preprocess_result_path']  # 预处理文件路径
    bboxpath_trian = config_training['bbox_path_train']
    bboxpath_value = config_training['bbox_path_value']
    bboxpath_test = config_training['bbox_path_test']


############test
    if args.test == 1:  # 测试

        dataset = data_cls_r.DataBowl3Classifier(
            datadir,bboxpath_test,
            'value_luna4.npy',
            config,
            phase='test')
        test_loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False)

        test(test_loader, net, save_dir,config)
        return
###############
    dataset = data_cls_r.DataBowl3Classifier(
            datadir, bboxpath_trian,
            'value_luna4.npy',
            config,
            phase='train')
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    dataset = data_cls_r.DataBowl3Classifier(
        datadir, bboxpath_value,
        'value_luna4.npy',
        config,
        phase='val')
    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    optimizer = torch.optim.SGD(
        net.parameters(),
        args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    def get_lr(epoch):  #学习率根据epoch增加而减小
        if epoch <= args.epochs * 0.5:
            lr = args.lr
        elif epoch <= args.epochs * 0.8:
            lr = 0.1 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr

    writer = SummaryWriter('runs4')  # 绘制loss曲线

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train(train_loader, net,  epoch, optimizer, get_lr, args.save_freq, save_dir)
        validate_loss, validate_acc = validate(val_loader, net)

        writer.add_scalars('loss', {'train_losss': train_loss,
                                    'validate_loss': validate_loss, }, epoch)
        writer.add_scalar('train_losss ', train_loss, epoch)
        writer.add_scalar('validate_loss', validate_loss, epoch)
        writer.add_scalars('accuraccy', {'train_accuracy': train_acc, 'validate_accuracy': validate_acc}, epoch)
        writer.add_scalar('train_accuracy', train_acc, epoch)
        writer.add_scalar('validate_accuracy', validate_acc, epoch)
    writer.close()


def train(data_loader, net,  epoch, optimizer, get_lr, save_freq, save_dir):
    start_time = time.time()

    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    accHist = []
    tpn = 0
    fpn = 0
    fnn = 0
    tnn = 0
    tpall = 0
    tnall = 0
    correct = 0
    total = 0
    for i, (data, target) in enumerate(data_loader):
        data = Variable(data.cuda(non_blocking=True))
        targetdata = target.numpy()[:, 0]
        target = target.float()
        target = Variable(target.cuda(non_blocking=True))

        output = net(data)
        loss_output = binary_cross_entropy(output, target)

        outdata = output.data.cpu().numpy()
        del output
        optimizer.zero_grad()
        loss_output.backward()
        optimizer.step()

        loss_output_data = loss_output.item()
        pred = outdata > 0.5
        tpn += np.sum(1 == pred[targetdata == 1])
        fpn += np.sum(1 == pred[targetdata == 0])
        fnn += np.sum(0 == pred[targetdata == 1])
        tnn += np.sum(0 == pred[targetdata == 0])
        tpall += np.sum(targetdata == 1)
        tnall += np.sum(targetdata == 0)

        metrics.append(loss_output_data)


    if epoch % args.save_freq == 0:
        state_dict = net.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        torch.save({
            'epoch': epoch,
            'save_dir': save_dir,
            'state_dict': state_dict,
            'args': args},
            os.path.join(save_dir, '%03d.ckpt' % epoch))

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    acc = (tpn + tnn) / (tpall + tnall)
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * tpn / tpall,
        100.0 * tnn / tnall,
        tpall,
        tnall,
        end_time - start_time))
    print('loss %2.4f,acc %.4f, tpn %d, fpn %d, fnn %d,  tnn %d' % (
        np.mean(metrics), acc, tpn, fpn, fnn, tnn))

    return np.mean(metrics), acc

def validate(data_loader, net):
    start_time = time.time()

    net.eval()

    metrics = []
    tpn = 0
    fpn = 0
    fnn = 0
    tnn = 0
    tpall = 0
    tnall = 0
    for i, (data, target) in enumerate(data_loader):
        data = Variable(data.cuda(non_blocking=True))
        targetdata = target.numpy()[:, 0]
        target = target.float()
        target = Variable(target.cuda(non_blocking=True))

        output = net(data)
        # loss_output = binary_cross_entropy(output, target)
        loss_output = binary_cross_entropy(output, target)

        outdata = output.data.cpu().numpy()
        del output
        loss_output_data = loss_output.item()

        pred = outdata > 0.5
        tpn += np.sum(1 == pred[targetdata == 1])
        fpn += np.sum(1 == pred[targetdata == 0])
        fnn += np.sum(0 == pred[targetdata == 1])
        tnn += np.sum(0 == pred[targetdata == 0])
        tpall += np.sum(targetdata == 1)
        tnall += np.sum(targetdata == 0)

        metrics.append(loss_output_data)
    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    acc = (tpn+tnn)/(tpall+tnall)
    print('validate:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * tpn / tpall,
        100.0 * tnn / tnall,
        tpall,
        tnall,
        end_time - start_time))
    print('loss %2.6f,  acc %.4f,  tpn %d,  fpn %d,  fnn %d,  tnn %d ' % (
        np.mean(metrics), acc, tpn, fpn, fnn, tnn))

    return np.mean(metrics), acc

def test(data_loader, net,save_dir,conifg):
    start_time = time.time()
    if 'output_feature' in conifg:
        print(1)
    save_dir = os.path.join(save_dir, 'newcls13_csv')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    net.eval()

    outputlist = []
    targetlist = []
    for i, (data, target) in enumerate(data_loader):
        data = Variable(data).cuda()
        target = [np.asarray(t, np.float32) for t in target]

        output = net(data)
        outputlist.append(output.data.cpu().numpy())
        targetlist.append(target)
        del output
    output = np.concatenate(outputlist, 0)
    target = np.concatenate(targetlist, 0)
    anstable = np.hstack((target, output))
    # sigmoid = nn.Sigmoid()  # 激活函数
    new_anstable = []
    for ans in anstable:
        sigpp = sigmoid(ans[1])
        final_pob1 = 0.7 * sigpp + 0.3 * ans[7]
        final_pob2 = 0.6 * sigpp + 0.4 * ans[7]
        final_pob3 = 0.5 * sigpp + 0.5 * ans[7]
        final_pob4 = 0.4 * sigpp + 0.6 * ans[7]
        final_ans = np.hstack((ans, final_pob1, final_pob2, final_pob3, final_pob4))
        new_anstable.append(final_ans)

    df = pandas.DataFrame(new_anstable)
    df.columns = {'seriesuid', 'confidence', 'coordZ', 'coordX', 'coordY', 'd', 'calss', 'classprob',
                  'fianlprpb1', 'fianlprpb2', 'fianlprpb3', 'fianlprpb4'}  # 'probability',
    df.to_csv(os.path.join(save_dir, 'rescls13_test.csv'), index=False)
    pos_pred = []
    for ans in new_anstable:
        if ans[8] >= 0.5:
            pos_pred.append(ans)
    pos_pred = np.asarray(pos_pred)
    df = pandas.DataFrame(pos_pred)
    df.columns = {'seriesuid', 'confidence', 'coordZ', 'coordX', 'coordY', 'd', 'calss', 'classprob',
                  'fianlprpb1','fianlprpb2','fianlprpb3','fianlprpb4'}  # 'probability',
    df.to_csv(os.path.join(save_dir, 'newres13_pos0.5.csv'), index=False)


def sigmoid(x):
    return 1 / (1 + math.exp(-x)) # 激活函数



if __name__ == '__main__':
    main()