"""
    @Project: PetroSpear
    @File   : utils/utils_draw_confusion_matrix.py
    @Author : mulder
    @E-mail : c_mulder@163.com
    @Date   : 2026-09-08
    @Info   : Implementation of following functions:
                  accuracy
                  draw_cm、draw_confusion_matrix
                  draw_sim_matrix
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.metrics import confusion_matrix, classification_report


def accuracy(output, target, topk=(1,)):
    """
    Top-k accuracy calculation

    参数:
    output (np.ndarray): N x N 分类器Logits输出
    target (np.ndarray): N x 1 Label
    topk (tuple[int, ...]):

    返回:
    res (list[float]): top-k accuracy values
    """

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        # 使用 unsqueeze 和 reshape 提高鲁棒性
        # correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct = pred.eq(target.unsqueeze(0).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def draw_sim_matrix(confusion_matrix_data, class_labels, filename, res_dir):
    # 设置绘图风格
    plt.figure(figsize=(10, 8))  # 设置画布大小，确保 13x13 的格子不会太拥挤
    sns.set_theme(style="white")  # 设置背景风格

    # 绘制热力图 (混淆矩阵)
    ax = sns.heatmap(
        confusion_matrix_data,
        annot=True,  # 在单元格中显示数值
        fmt='.2f',  # 数值格式为整数 (decimal)
        cmap='Blues',  # 颜色映射方案，'Blues', 'YlGnBu', 'viridis' 等都很适合
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar_kws={'label': 'Similarity'}  # 颜色条的标签
    )

    # 美化图表
    plt.title(filename, fontsize=9, pad=20)
    # plt.xlabel('Predicted Label', fontsize=9)
    # plt.ylabel('True Label', fontsize=9)

    # 确保 x 和 y 轴的标签水平显示，避免重叠
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # 调整布局以防止标签被裁剪
    plt.tight_layout()

    # 显示图形
    # plt.show()
    filename_png = "{}.{}".format(filename, "png")
    filename_svg = "{}.{}".format(filename, "svg")
    # 保存图形
    # 如果需要保存为图片，可以取消下面这行的注释
    filepath = '%s/%s' % (res_dir, filename_png)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

    filepath = '%s/%s' % (res_dir, filename_svg)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')


def draw_confusion_matrix(cm, label_class_num, res_dir, stage=1):
    normalize = True
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # 绘制混淆矩阵
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.yticks(range(label_class_num))
    plt.xticks(range(label_class_num))
    # plt.yticks(range(opt.n_classes), label_name)
    # plt.xticks(range(opt.n_classes), label_name, rotation=45)

    # plt.tight_layout()
    plt.colorbar()

    for i in range(label_class_num):
        for j in range(label_class_num):
            color = (1, 1, 1) if i == j else (0, 0, 0)  # 对角线字体白色，其他黑色
            value = float(format('%.2f' % cm[j, i]))
            # value = float(format('%.1f' % (cm[j, i] / label_counts[j])))
            plt.text(i, j, value, verticalalignment='center', horizontalalignment='center', color=color)

    if matplotlib.__version__ < '3.8':
        figmanager = plt.get_current_fig_manager()
        figmanager.window.state('zoomed')  # 窗口最大化
    else:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()  # 窗口最大化

    figure = plt.gcf()
    inche_w = 10
    inche_h = 8
    if label_class_num > 50:
        inche_w = 30
        inche_h = 24
    elif label_class_num > 30:
        inche_w = 20
        inche_h = 16
    figure.set_size_inches(inche_w, inche_h)

    plt.savefig('%s/confusion_matrix_%d.png' % (res_dir, stage), dpi=300)
    plt.savefig('%s/confusion_matrix_%d.svg' % (res_dir, stage), dpi=300)
    # plt.show()


def draw_cm(y_preds, y_trues, n_classes, res_dir, stage=1):
    # 计算每个类别的样本数量
    label_counts = torch.bincount(y_trues)
    if len(label_counts) == n_classes:
        # 输出报告
        report = classification_report(y_trues.cpu(), y_preds.cpu(), output_dict=True)
        report_file = "%s/classification_report_%d.csv" % (res_dir, stage)
        df = pd.DataFrame(report).transpose()
        df.to_csv(report_file, index=True)

        # 计算混淆矩阵
        cm = confusion_matrix(y_trues.cpu().numpy(), y_preds.cpu().numpy())
        draw_confusion_matrix(cm, n_classes, res_dir, stage)
    else:
        print("label_counts < opt.n_classes")
