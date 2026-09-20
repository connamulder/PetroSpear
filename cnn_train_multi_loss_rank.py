"""
    @Project: PetroSpear
    @File   : 06_cnn_train_multi_loss.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2026-07-01
    @Info   : 重构代码，实现RockModel类、岩石图像数据集英文名称列表和对应的中文名称字典获取函数模块化。
    @Date   : 2026-07-07
    @Info   : loss修改为自定义CustomCrossEntropyLoss
"""

import os
import argparse
import time
import matplotlib.pyplot as plt
import sys

import torch
import torchvision.transforms as transforms

from torchsummary import summary
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from models.models_rock import RockModel
from models.celoss_cus import CustomCrossEntropyLoss

from dataset_rocks_list import ROCKS
from dataset_rocks_labels_name import get_rock_labels_name

from similarity_matrix_clip import build_similarity_matrix_by_clip
from similarity_matrix_neo4j import build_similarity_matrix_by_neo4j
from similarity_matrix_glove import build_similarity_matrix_by_glove

from utils import utils_result, utils_draw_confusion_matrix


# 是否为编码阶段的测试，编码阶段加载少量数据
is_coding_test = False

parser = argparse.ArgumentParser()
# Training parameters
parser.add_argument("--model_type", type=str, default='resnet50', help="pre-trained model: vgg16 | resnet50 | vit_base")
parser.add_argument("--is_pretrained", action='store_true', help="is model pretrained")
parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
parser.add_argument("--n_stage", type=int, default=1, help="number of stages of training")
parser.add_argument("--batch_size", type=int, default=16, help="8 | 128")
parser.add_argument("--n_stop", type=int, default=10, help="number of epochs of training")
parser.add_argument("--do_train", action='store_true', help="is do training")
parser.add_argument("--is_parall", action='store_true', help="is data paralled")
parser.add_argument("--celoss_type", type=str, default='celoss_glove',
                    help="cross entropy losses: celoss_node | celoss_clip | celoss_glove")

# Dataset
parser.add_argument("--dataset_name", type=str, default='plutonicrocks13_v3', help="dataset, mnist  cifar10 | rocks7 | rocks32")
parser.add_argument("--dataset_file", type=str, default='plutonicrocks13_V3_train.txt', help="the file name of the dataset")
parser.add_argument("--val_file", type=str, default='plutonicrocks13_V3_val.txt', help="the file name of the val")
parser.add_argument("--test_file", type=str, default='plutonicrocks13_V3_test.txt', help="the file name of the test")
parser.add_argument("--n_classes", type=int, default=13, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=224, help="size of each image dimension: 224 | 299")
# Fixed parameters
parser.add_argument("--lr", type=float, default=0.00005, help="adam: learning rate")
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument("--data_dist", type=str, default='norm', help="data distribution.norm: mean=std=[0.5, 0.5, 0.5]")
parser.add_argument("--loss_weight", action='store_true', help="is data paralled")
parser.set_defaults(loss_weight=False)
parser.add_argument('--soft_weight', default=0.1, type=float, metavar='S', help='soft loss weight')
parser.add_argument("--do_aug", action='store_true', help="is do augument")
parser.set_defaults(do_aug=True)
parser.add_argument("--imbalance_loader", action='store_true', help="use imbalance loader")
parser.set_defaults(imbalance_loader=True)
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
parser.set_defaults(resume=True)
opt = parser.parse_args()
print(opt)

if is_coding_test:
    opt.is_pretrained = True
    opt.do_train = False
    opt.is_parall = False
    opt.celoss_type = "celoss_glove"
    opt.n_epochs = 10000
    opt.n_stage = 1
    opt.n_stop = 20
    ratio_dataset_test = 0.1

"""
opt.model_type = 'resnet50'
opt.is_pretrained = True
opt.do_train = False
opt.is_parall = True
opt.celoss_type = "celoss_clip"
opt.n_epochs = 10000
opt.n_stop = 50
opt.soft_weight = 0.2
opt.n_stage = 2
"""


def evaluate_model(model, data_loader, cuda=True, ratio_dataset=1.0):
    top1_acc = utils_result.AverageMeter()

    for iter, (x_, y_) in enumerate(data_loader):
        if is_coding_test:
            n_train_patch = int(len(data_loader) * ratio_dataset)
            if iter > n_train_patch:
                break

        if cuda:
            x_, y_ = x_.cuda(), y_.cuda()

        with torch.no_grad():
            y_pred = model(x_)
            prec1_test, prec2_test, prec3_test = utils_draw_confusion_matrix.accuracy(y_pred.data, y_,
                                                                                      topk=(1, 2, 3))
            top1_acc.update(prec1_test.item())

        print(
            "[Batch %d/%d]" % (iter, len(data_loader))
        )

    top1_acc_all = top1_acc.avg

    return top1_acc_all


def main():
    # 装载岩石图像数据
    class_loss_weights = None
    data_loader = None
    val_loader = None
    test_loader = None
    if 'rocks' in opt.dataset_name:
        channel_mean = []
        channel_std = []
        if opt.data_dist == 'norm':
            channel_mean = [0.5, 0.5, 0.5]
            channel_std = [0.5, 0.5, 0.5]

        # tensor([0.5617, 0.5398, 0.5221]) tensor([0.1609, 0.1573, 0.1584])
        #                                        transforms.RandomHorizontalFlip(p=0.5),
        #                                        transforms.RandomVerticalFlip(p=0.5),
        #                                        transforms.RandomRotation(degrees=30),
        if opt.do_aug:
            train_tf = transforms.Compose([transforms.Resize(opt.img_size),
                                           transforms.CenterCrop(opt.img_size),
                                           transforms.RandomHorizontalFlip(p=0.5),
                                           transforms.RandomVerticalFlip(p=0.5),
                                           transforms.RandomRotation(degrees=30),
                                           transforms.ToTensor(),
                                           transforms.Normalize(channel_mean, channel_std)])
            print("执行transforms数据增强！")
        else:
            train_tf = transforms.Compose([transforms.Resize(opt.img_size),
                                           transforms.CenterCrop(opt.img_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(channel_mean, channel_std)])
            print("不执行transforms数据增强！")

        val_tf = transforms.Compose([transforms.Resize(opt.img_size),
                                     transforms.CenterCrop(opt.img_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(channel_mean, channel_std)])
        # 预读取图像并存储到pkl文件中，可启动GPU训练
        dataset_file = ''
        val_file = ''
        test_file = ''
        if sys.platform == 'linux':
            dataset_file = os.path.join('/home/u2021800205/rockclass', opt.dataset_file)
            val_file = os.path.join('/home/u2021800205/rockclass', opt.val_file)
            test_file = os.path.join('/home/u2021800205/rockclass', opt.test_file)
        elif sys.platform == 'win32':
            dataset_file = os.path.join(r"F:\11_CV_Datasets\rockclass", opt.dataset_file)
            val_file = os.path.join(r"F:\11_CV_Datasets\rockclass", opt.val_file)
            test_file = os.path.join(r"F:\11_CV_Datasets\rockclass", opt.test_file)

        train_data = ROCKS(dataset_file, opt.img_size, transform=train_tf)
        val_data = ROCKS(val_file, opt.img_size, transform=val_tf)
        test_data = ROCKS(test_file, opt.img_size, transform=val_tf)

        labels = train_data.get_labels()
        labels = [int(x) for x in labels]
        labels = torch.tensor(labels)
        # 计算每个类别的样本数量
        label_counts = torch.bincount(labels)
        class_loss_weights = torch.max(label_counts) / label_counts.float()

        if opt.imbalance_loader:
            # weight = [ ] 里面每一项代表该样本种类占总样本的倒数。
            class_weights = 1.0 / label_counts.float()
            sample_weights = class_weights[labels]

            # 创建采样器
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            data_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, sampler=sampler)
        else:
            data_loader = torch.utils.data.DataLoader(train_data, batch_size=opt.batch_size, pin_memory=True,
                                                      shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=opt.batch_size, pin_memory=True, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=opt.batch_size, pin_memory=True, shuffle=True)

    data = data_loader.__iter__().__next__()[0]
    img_shape = (data.shape[1], opt.img_size, opt.img_size)
    print("image shape: {}".format(img_shape))

    # 组织输出目录
    model_type = opt.model_type
    res_dir = '%s/%s' % (opt.dataset_name, model_type)
    res_dir = "./output/{}/res_{}_{}".format(res_dir, opt.dataset_name, opt.img_size)

    if opt.imbalance_loader:
        res_dir = "{}_{}".format(res_dir, "imbalance_loader")
    if opt.is_pretrained:
        res_dir = "{}_{}".format(res_dir, "pretrained")
    else:
        res_dir = "{}_{}".format(res_dir, "scratch")

    # 根据loss类型设置输出目录。同时，定义训练损失函数（criterion）
    criterion = None
    if opt.celoss_type == "celoss_clip":
        similarity_matrix = build_similarity_matrix_by_clip(opt.dataset_name)
        criterion = CustomCrossEntropyLoss(sim_matrix=similarity_matrix)
    if opt.celoss_type == "celoss_glove":
        similarity_matrix = build_similarity_matrix_by_glove(opt.dataset_name)
        criterion = CustomCrossEntropyLoss(sim_matrix=similarity_matrix)
    elif opt.celoss_type == "celoss_node":
        # 需要替换为 节点相似度矩阵
        similarity_matrix = build_similarity_matrix_by_neo4j(opt.dataset_name)
        criterion = CustomCrossEntropyLoss(sim_matrix=similarity_matrix)
    res_dir = "{}_{}".format(res_dir, opt.celoss_type)

    soft_weight = int(opt.soft_weight * 10)
    res_dir = "{}_epochs_{}_nstop_{}_weight_{}_stage_{}_lr_{:f}".format(res_dir, opt.n_epochs, opt.n_stop,
                                                                        soft_weight, opt.n_stage, opt.lr)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir, exist_ok=True)
    print("训练结果输出目录: {}".format(res_dir))

    # 定义岩石图像分类模型
    if opt.model_type == "vit_base":
        _, labels_name = get_rock_labels_name(opt.dataset_name)
        model = RockModel(class_num=opt.n_classes, model_type=model_type, pretrained=opt.is_pretrained,
                          img_size=opt.img_size,
                          labels_name=labels_name)
    else:
        model = RockModel(class_num=opt.n_classes, model_type=model_type, pretrained=opt.is_pretrained, img_size=opt.img_size)
    utils_result.save_model_to_txt(model, opt.batch_size, opt.img_size, res_dir)
    if (opt.model_type == "vit_base") or (opt.model_type == "desnet"):
        print("不输出summary")
    else:
        summary(model, input_size=(3, opt.img_size, opt.img_size), device='cpu')

    # 定义优化器（optimizer）
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), opt.lr,
                                momentum=opt.momentum,
                                weight_decay=opt.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    # 设定计算环境，GPU或CPU计算
    cuda = True if torch.cuda.is_available() else False
    print("cuda: {}".format(cuda))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: {}".format(device))
    if cuda:
        if opt.is_parall:
            model = torch.nn.DataParallel(model.cuda())
        else:
            model = model.cuda()
        if criterion is not None:
            criterion = criterion.cuda()

    # 执行训练
    if opt.do_train:
        train_stage = 1
        current_weight = 0.0

        if opt.resume:
            weight_file = '%s/%s_%s_model.pth' % (res_dir, opt.dataset_name, model_type)
            if os.path.exists(weight_file):
                print("正在从模型参数文件{}恢复.".format(weight_file))
                if opt.is_parall:
                    model.load_state_dict(torch.load(weight_file))
                else:
                    checkpoint = torch.load(weight_file)
                    model.load_state_dict(checkpoint)
                current_weight = opt.soft_weight
                train_stage += 1

        log_writer = SummaryWriter(log_dir=res_dir)

        losses = utils_result.AverageMeter()
        losses_standard = utils_result.AverageMeter()
        losses_soft = utils_result.AverageMeter()
        top1 = utils_result.AverageMeter()
        top2 = utils_result.AverageMeter()
        top3 = utils_result.AverageMeter()
        losses_val = utils_result.AverageMeter()
        top1_val = utils_result.AverageMeter()
        top2_val = utils_result.AverageMeter()
        top3_val = utils_result.AverageMeter()
        loss_history = []
        pred_history = []
        loss_val_history = []
        pred_val_history = []

        best_loss = 10.0
        train_step = 0
        start_time = time.time()

        for epoch in range(opt.n_epochs):

            # training
            model.train()

            for iter, (x_, y_) in enumerate(data_loader):
                if is_coding_test:
                    n_train_patch = int(len(data_loader) * ratio_dataset_test)
                    if iter > n_train_patch:
                        break

                if iter == data_loader.dataset.__len__() // opt.batch_size:
                    break

                if cuda:
                    x_, y_ = x_.cuda(), y_.cuda()
                # update D network
                optimizer.zero_grad()
                y_pred = model(x_)

                loss_standard, loss_soft = criterion(y_pred, y_)
                losses_standard.update(loss_standard.item())
                losses_soft.update(loss_soft.item())
                loss = (1-current_weight) * loss_standard + current_weight * loss_soft

                losses.update(loss.item())

                prec1, prec2, prec3 = utils_draw_confusion_matrix.accuracy(y_pred.data, y_, topk=(1, 2, 3))
                top1.update(prec1.item())
                top2.update(prec2.item())
                top3.update(prec2.item())

                loss.backward()
                optimizer.step()

                print(
                    "[Stage %d/%d] [Epoch %d/%d] [Batch %d/%d] [loss: %f] [pred: %f] [current_weight: %f] [loss_standard: %f] [loss_soft: %f]"
                    % (train_stage, opt.n_stage, epoch, opt.n_epochs, iter, len(data_loader), loss.item(), prec1.item(), current_weight, loss_standard.item(), loss_soft.item())
                )

            # 学习率调整
            # scheduler.step()
            # current_lr = optimizer.param_groups[0]['lr']

            loss_history.append(losses.avg)
            pred_history.append(top1.avg)
            if log_writer is not None:
                # log_writer.add_scalar('tl/lr', current_lr, epoch)
                log_writer.add_scalar('tl/train_loss', losses.avg, epoch)
                log_writer.add_scalar('tl/train_pred', top1.avg, epoch)
                log_writer.add_scalar('tl/train_pred2', top2.avg, epoch)
                log_writer.add_scalar('tl/train_pred3', top3.avg, epoch)

                log_writer.add_scalar('tl/train_loss_standard', losses_standard.avg, epoch)
                log_writer.add_scalar('tl/train_loss_soft', losses_soft.avg, epoch)

            losses.reset()
            top1.reset()

            # validation
            model.eval()
            for iter, (x_, y_) in enumerate(val_loader):
                if iter == val_loader.dataset.__len__() // opt.batch_size:
                    break

                if is_coding_test:
                    n_train_patch = int(len(val_loader) * ratio_dataset_test)
                    if iter > n_train_patch:
                        break

                if cuda:
                    x_, y_ = x_.cuda(), y_.cuda()
                with torch.no_grad():
                    y_pred = model(x_)

                    v_loss_standard, v_loss_soft = criterion(y_pred, y_)
                    v_loss = (1 - current_weight) * v_loss_standard + current_weight * v_loss_soft

                prec_val, prec2_val, prec3_val = utils_draw_confusion_matrix.accuracy(y_pred.data, y_, topk=(1, 2, 3))
                top1_val.update(prec_val.item())
                top2_val.update(prec2_val.item())
                top3_val.update(prec3_val.item())

                losses_val.update(v_loss.item())

                print(
                    "[Stage %d/%d] [Epoch %d/%d] [Batch %d/%d] [val loss: %f] [val pred: %f] [val pred2: %f] [val pred3: %f]"
                    % (train_stage, opt.n_stage, epoch, opt.n_epochs, iter, len(val_loader), v_loss.item(), prec_val.item(), prec2_val.item(), prec3_val.item())
                )

            loss_val_history.append(losses_val.avg)
            pred_val_history.append(top1_val.avg)
            if log_writer is not None:
                log_writer.add_scalar('tl/val_loss', losses_val.avg, epoch)
                log_writer.add_scalar('tl/val_pred', top1_val.avg, epoch)
                log_writer.flush()

            # 监测验证集loss实现早停机制
            if losses_val.avg < best_loss:
                best_loss = losses_val.avg
                if train_stage == 1:
                    weight_file = '%s/%s_%s_model.pth' % (res_dir, opt.dataset_name, model_type)
                else:
                    weight_file = '%s/%s_%s_model_stage_%d.pth' % (res_dir, opt.dataset_name, model_type, train_stage)
                torch.save(model.state_dict(), weight_file)
                train_step = 0
            else:
                train_step += 1
            if train_step > opt.n_stop:
                if train_stage >= 1 and train_stage < opt.n_stage:
                    train_step = 0
                    current_weight = opt.soft_weight
                    train_stage += 1
                    best_loss = 10.0
                else:
                    break

            losses_val.reset()
            top1_val.reset()

        end_time = time.time()
        # --------------------------- Saving train time --------------------------- #
        train_time = end_time - start_time
        runtime_file_path = '%s/runtime_train.txt' % res_dir
        with open(runtime_file_path, 'w') as runtime_file:
            runtime_file.write("start time: %s\n" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
            runtime_file.write("end time: %s\n" % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time)))

            hours, remainder = divmod(train_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            time_format = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

            runtime_file.write("train time: %s\n" % time_format)

        # --------------------------- Display performance --------------------------- #
        # plot loss
        plt.plot(loss_history, label='loss')
        plt.plot(pred_history, label='pred')
        plt.plot(loss_val_history, label='val_loss')
        plt.plot(pred_val_history, label='val_pred')
        plt.legend()
        plt.savefig('%s/%s_%s_train_plot.png' % (res_dir, opt.dataset_name, model_type))
        plt.savefig('%s/%s_%s_train_plot.svg' % (res_dir, opt.dataset_name, model_type))
        plt.clf()
    else:
        from mistake_cos_matrix import compute_cost_matrix_from_semantics
        import numpy as np

        method = 'linear'
        sim_matrix_type = 'glove'
        if opt.celoss_type == 'celoss_clip':
            sim_matrix_type = 'clip'
        elif opt.celoss_type == 'celoss_glove':
            sim_matrix_type = 'glove'
        elif opt.celoss_type == 'celoss_node':
            sim_matrix_type = 'neo4j'

        # 计算代价矩阵 (使用线性转换)
        cost_mat, _ = compute_cost_matrix_from_semantics(
            dataset_name=opt.dataset_name,
            method=method,
            sim_matrix_type=sim_matrix_type
        )

        train_stage = opt.n_stage
        for num_stage in range(train_stage):
            if num_stage == 0:
                weight_file = '%s/%s_%s_model.pth' % (res_dir, opt.dataset_name, model_type)
                pt_file = '%s/%s_%s_model.pt' % (res_dir, opt.dataset_name, model_type)
            else:
                weight_file = '%s/%s_%s_model_stage_%d.pth' % (res_dir, opt.dataset_name, model_type, train_stage)
                pt_file = '%s/%s_%s_model_stage_%d.pt' % (res_dir, opt.dataset_name, model_type, train_stage)

            if os.path.exists(weight_file):
                print("正在从模型参数文件{}恢复.".format(weight_file))
                if opt.is_parall:
                    model.load_state_dict(torch.load(weight_file))
                    model.eval()
                else:
                    checkpoint = torch.load(weight_file)
                    model.load_state_dict(checkpoint)
                    # model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})

                    model.eval()

                    # 转换为pt模型文件
                    example = torch.rand(1, 3, opt.img_size, opt.img_size)
                    example_label = torch.ones(1, 3, opt.img_size, opt.img_size)
                    if cuda:
                        example, example_label = example.cuda(), example_label.cuda()
                    traced_script_module = torch.jit.trace(model, example)
                    traced_script_module.save(pt_file)
                    output = traced_script_module(example_label)
                    print(output)

                # 随机选择一些样本
                # indices = np.random.choice(len(val_loader), size=16, replace=False)
                # images = [val_loader[i][0] for i in indices]
                # labels = [val_loader[i][1] for i in indices]
                y_preds = []
                y_preds_top2 = []
                y_preds_top3 = []
                y_trues = []

                top1_test = utils_result.AverageMeter()
                top2_test = utils_result.AverageMeter()
                top3_test = utils_result.AverageMeter()

                for iter, (x_, y_) in enumerate(val_loader):
                    if is_coding_test:
                        n_train_patch = int(len(val_loader) * ratio_dataset_test)
                        if iter > n_train_patch:
                            break

                    """
                    if iter == val_loader.dataset.__len__() // opt.batch_size:
                        break
                    """
                    if cuda:
                        x_, y_ = x_.cuda(), y_.cuda()

                    with torch.no_grad():
                        y_pred_logits = model(x_)
                        prec1_test, prec2_test, prec3_test = utils_draw_confusion_matrix.accuracy(y_pred_logits.data,
                                                                                                  y_,
                                                                                                  topk=(1, 2, 3))
                        top1_test.update(prec1_test.item())
                        top2_test.update(prec2_test.item())
                        top3_test.update(prec3_test.item())

                        _, top3_indices = torch.topk(y_pred_logits, k=3, dim=-1)
                        y_pred = top3_indices[:, 0]
                        top2_indices = top3_indices[:, 1]
                        top3_indices = top3_indices[:, 2]

                        # y_pred = torch.argmax(y_pred_logits, dim=1)
                    if iter == 0:
                        y_preds = y_pred
                        y_preds_top2 = top2_indices
                        y_preds_top3 = top3_indices
                        y_trues = y_
                    else:
                        y_preds = torch.cat((y_preds, y_pred), dim=0)
                        y_preds_top2 = torch.cat((y_preds_top2, top2_indices), dim=0)
                        y_preds_top3 = torch.cat((y_preds_top3, top3_indices), dim=0)
                        y_trues = torch.cat((y_trues, y_), dim=0)

                    print(
                        "[Batch %d/%d]" % (iter, len(val_loader))
                    )

                acc_test_file_path = '%s/acc_test_%d.txt' % (res_dir, num_stage+1)

                se_cost = cost_mat[y_trues.cpu().numpy(), y_preds.cpu().numpy()]
                mistake_severity = np.mean(se_cost[se_cost != 0]) if np.any(se_cost != 0) else 0.0

                se_cost_top2 = cost_mat[y_trues.cpu().numpy(), y_preds_top2.cpu().numpy()]
                mistake_severity_top2 = np.mean(se_cost_top2[se_cost_top2 != 0]) if np.any(se_cost_top2 != 0) else 0.0

                se_cost_top3 = cost_mat[y_trues.cpu().numpy(), y_preds_top3.cpu().numpy()]
                mistake_severity_top3 = np.mean(se_cost_top3[se_cost_top3 != 0]) if np.any(se_cost_top3 != 0) else 0.0

                mistake_severity_ave = (mistake_severity + mistake_severity_top2 + mistake_severity_top3) / 3

                with open(acc_test_file_path, 'w') as acc_test_file:
                    acc_test_file.write("测试集准确率:\n")
                    acc_test_file.write("Top1 Acc: %.6f\n" % top1_test.avg)
                    acc_test_file.write("Top2 Acc: %.6f\n" % top2_test.avg)
                    acc_test_file.write("Top3 Acc: %.6f\n" % top3_test.avg)
                    acc_test_file.write("Mistake severity: %.6f\n" % mistake_severity)
                    acc_test_file.write("Mistake severity (top2): %.6f\n" % mistake_severity_top2)
                    acc_test_file.write("Mistake severity (top3): %.6f\n" % mistake_severity_top3)
                    acc_test_file.write("Mistake severity (ave): %.6f\n" % mistake_severity_ave)

                utils_draw_confusion_matrix.draw_cm(y_preds, y_trues, opt.n_classes, res_dir, stage=num_stage+1)
            else:
                print("模型参数文件{}不存在.".format(weight_file))


if __name__ == '__main__':
    main()
