import argparse
import time
import data
from data_detector import DataBowl3Detector
from importlib import import_module
import shutil
from utils import *
import sys

sys.path.append('../')
from split_combine import SplitComb

import torch
from torch.nn import DataParallel
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from config_training import config as config_training
from layers_senet import acc
from tensorboardX import SummaryWriter



parser = argparse.ArgumentParser(description='PyTorch DataBowl3 Detector')
parser.add_argument('--model', '-m', metavar='MODEL', default='res18_senet',
                    help='model') # 网络模型
######################################
parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')####CPU线程数
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=8, type=int,
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

parser.add_argument('--save-dir', default='', type=str, metavar='SAVE',
                    help='directory to save checkpoint (default: none)')

parser.add_argument('--test', default=0, type=int, metavar='TEST',
                    help='1 do test evaluation, 0 not')
parser.add_argument('--split', default=8, type=int, metavar='SPLIT',
                    help='In the test phase, split the image to 8 parts')
##############################
parser.add_argument('--gpu', default='0,1,2,3,4,5', type=str, metavar='N',
                    help='use gpu')
parser.add_argument('--n_test', default=6, type=int, metavar='N',
                    help='number of gpu for test')


def main():
    global args
    args = parser.parse_args()

    torch.manual_seed(0) # 随机数种子
    ##################################
    torch.cuda.set_device(0)

    model = import_module(args.model)
    config, net, loss, get_pbb = model.get_model()
    start_epoch = args.start_epoch
    save_dir = args.save_dir

    if args.resume:
        checkpoint = torch.load(args.resume)
        if start_epoch == 0:
            start_epoch = checkpoint['epoch'] + 1
        if not save_dir:
            save_dir = checkpoint['save_dir']
        else:
            save_dir = os.path.join('results', save_dir)
        net.load_state_dict(checkpoint['state_dict'])
    else:
        if start_epoch == 0:
            start_epoch = 1
        if not save_dir:
            exp_id = time.strftime('%Y%m%d-%H%M%S', time.localtime()) # 根据当前时间生成文件夹
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
    loss = loss.cuda()
    cudnn.benchmark = True
    ##########################################
    net = DataParallel(net)
    ######################################
    datadir = config_training['preprocess_result_path']  #预处理文件路径

    if args.test == 1: #测试模式
        margin = 32
        sidelen = 96

        split_comber = SplitComb(sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])
        dataset = DataBowl3Detector(
            datadir,
            'value_luna9.npy',
            config,
            phase='test',
            split_comber=split_comber)
        test_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=data.collate,
            pin_memory=False)

        test(test_loader, net, get_pbb, save_dir, config)
        return


    dataset = DataBowl3Detector(
        datadir,
        'train_luna9.npy',
        config,
        phase='train')
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    dataset = DataBowl3Detector(
        datadir,
        'value_luna9.npy',
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
        weight_decay=args.weight_decay)  #优化器用的是Momentum

    def get_lr(epoch):  #学习率根据epoch增加而减小
        if epoch <= args.epochs * 0.5:
            lr = args.lr
        elif epoch <= args.epochs * 0.8:
            lr = 0.1 * args.lr
        else:
            lr = 0.01 * args.lr
        return lr

    writer = SummaryWriter('runs9-res') #绘制loss曲线

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc= train(train_loader, net, loss, epoch, optimizer, get_lr, args.save_freq, save_dir)
        validate_loss, validate_acc = validate(val_loader, net, loss)

        writer.add_scalars('loss', {'train_losss': train_loss,
                                     'validate_loss':  validate_loss,}, epoch)
        writer.add_scalar('train_losss ',train_loss,epoch)
        writer.add_scalar('validate_loss',validate_loss,epoch)
        writer.add_scalars('accuracy',{'train_accuracy': train_acc,'validate_accuracy': validate_acc},epoch)
        writer.add_scalar('train_accuracy', train_acc, epoch)
        writer.add_scalar('validate_accuracy', validate_acc, epoch)
    writer.close() # 记录损失和精确度


def train(data_loader, net, loss, epoch, optimizer, get_lr, save_freq, save_dir):
    start_time = time.time()

    net.train()
    lr = get_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader): # enumerate() 函数用于将一个可遍历的数据对象组合为一个索引序列
        data = Variable(data.cuda(non_blocking=True))
        target = Variable(target.cuda(non_blocking=True))
        coord = Variable(coord.cuda(non_blocking=True))

        output = net(data, coord) # 前向传播
        loss_output = loss(output, target) # 计算损失
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step() # 反向传播、更新参数

        loss_output[0] = loss_output[0].item()
        metrics.append(loss_output)
        del output # 将损失值添加到 metrics列表中，并删除输出变量以释放内存。

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

    metrics = np.asarray(metrics, np.float32) # 转为numpy数组
    metrics_accuracy = (np.sum(metrics[:, 6]) + np.sum(metrics[:, 8])) / (
                np.sum(metrics[:, 7]) + np.sum(metrics[:, 9]))
    print('Epoch %03d (lr %.5f)' % (epoch, lr))
    print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f, accuracy %3.8f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5]),
        metrics_accuracy))
    #真阳性率 真阴性率 总正样本 负样本 time 损失 准确率
    return  np.mean(metrics[:, 0]), metrics_accuracy

def validate(data_loader, net, loss):
    start_time = time.time()

    net.eval()

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data.cuda(non_blocking=True))
        target = Variable(target.cuda(non_blocking=True))
        coord = Variable(coord.cuda(non_blocking=True))

        output = net(data, coord)
        loss_output = loss(output, target, train=False)
        del output
        loss_output[0] = loss_output[0].item()
        metrics.append(loss_output)

    end_time = time.time()

    metrics = np.asarray(metrics, np.float32)
    metrics_accuracy = (np.sum(metrics[:, 6]) + np.sum(metrics[:, 8])) / (np.sum(metrics[:, 7]) + np.sum(metrics[:, 9]))
    print('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
        100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
        100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
        np.sum(metrics[:, 7]),
        np.sum(metrics[:, 9]),
        end_time - start_time))
    print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f, accuracy %3.8f' % (
        np.mean(metrics[:, 0]),
        np.mean(metrics[:, 1]),
        np.mean(metrics[:, 2]),
        np.mean(metrics[:, 3]),
        np.mean(metrics[:, 4]),
        np.mean(metrics[:, 5]),
        metrics_accuracy))

    return np.mean(metrics[:, 0]), metrics_accuracy


def test(data_loader, net, get_pbb, save_dir, config, sidelen=None):
    start_time = time.time()
    save_dir = os.path.join(save_dir, 'bboxsenet9-res')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(save_dir)
    net.eval()
    namelist = []
    split_comber = data_loader.dataset.split_comber
    TP=0
    FP=0
    FN=0
    allpbb = []
    alllbb = []
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = data_loader.dataset.filenames[i_name].split('-')[0].split('/')[-1].split('_clean')[0]
        data = data[0][0]
        coord = coord[0][0]
        isfeat = False
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        n_per_run = args.n_test
        print(data.size())
        splitlist = range(0, len(data) + 1, n_per_run)
        if splitlist[-1] != len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist) - 1):
            input = Variable(data[splitlist[i]:splitlist[i + 1]]).cuda()
            inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]]).cuda()
            if isfeat:
                output, feature = net(input, inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output = net(input, inputcoord)
            outputlist.append(output.data.cpu().numpy())
            del output
        output = np.concatenate(outputlist, 0)
        output = split_comber.combine(output, nzhw=nzhw)
        if isfeat:
            feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])[:, :, :, :, :, np.newaxis]
            feature = split_comber.combine(feature, sidelen)[..., 0] #前面sidelen = 144

        thresh = -3
        pbb, mask = get_pbb(output, thresh, ismask=True)

        if isfeat:
            feature_selected = feature[mask[0], mask[1], mask[2]]
            np.save(os.path.join(save_dir, name + '_feature.npy'), feature_selected)
        tp,fp,fn,lbb_n= acc(pbb,lbb,0,0.1,0.1)
        TP+=len(tp)
        FP+=len(fp)
        FN+=len(fn)
        print([len(tp),len(fp),len(fn)])
        print([i_name, name])

        allpbb.append(pbb)
        alllbb.append(lbb)
        np.save(os.path.join(save_dir, name + '_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, name + '_lbb.npy'), lbb)
        namelist.append(list)

    np.save(os.path.join(save_dir, 'allpbb.npy'), allpbb)
    np.save(os.path.join(save_dir, 'alllbb.npy'), alllbb)
    np.save(os.path.join(save_dir, 'namelist.npy'), namelist)
    end_time = time.time()

    print('elapsed time is %3.2f seconds' % (end_time - start_time))
    print(TP,FP,FN)


def singletest(data, net, config, splitfun, combinefun, n_per_run, margin=64, isfeat=False):
    z, h, w = data.size(2), data.size(3), data.size(4)
    print(data.size())
    data = splitfun(data, config['max_stride'], margin)
    data = Variable(data.cuda(non_blocking=True), volatile=True, requires_grad=False)
    splitlist = range(0, args.split + 1, n_per_run)
    outputlist = []
    featurelist = []
    for i in range(len(splitlist) - 1):
        if isfeat:
            output, feature = net(data[splitlist[i]:splitlist[i + 1]])
            featurelist.append(feature)
        else:
            output = net(data[splitlist[i]:splitlist[i + 1]])
        output = output.data.cpu().numpy()
        outputlist.append(output)

    output = np.concatenate(outputlist, 0)
    output = combinefun(output, z / config['stride'], h / config['stride'], w / config['stride'])
    if isfeat:
        feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])
        feature = combinefun(feature, z / config['stride'], h / config['stride'], w / config['stride'])
        return output, feature
    else:
        return output


if __name__ == '__main__':
    main()