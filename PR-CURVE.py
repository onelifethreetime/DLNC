import sys
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import torch


def PrecisionRecall_PrecisionScope_Curve(image, text, label, ylim, color, dist_method='COS',
                                         k1=None, k_step=50, k2=None, cov=None, sigma1=None, sigma2=None, center=None,
                                         figure_num=1, marker='*', linestyle='-'):
    """
    image, text, label参照fx_calc_map_label函数
    k1, k_step, k2: precision-scope曲线的横坐标范围[k1, k2]，每隔k_step一个点
    figure_num: figure编号
    marker, linestyle: 曲线的样式
    """
    # 每行是第i个img特征与各个text特征的距离
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')     # 计算两个输入集合的距离，欧式距离
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')   # 计算两个输入集合的距离，余弦距离
    else:
        img = torch.tensor(image).to('cuda')
        text = torch.tensor(text).to('cuda')
        center = torch.tensor(center).to('cuda')
        cov = torch.tensor(cov).to('cuda')
        sigma1 = torch.tensor(sigma1).to('cuda')
        sigma2 = torch.tensor(sigma2).to('cuda')

        numcases = len(img)
        dist = (torch.ones((numcases, numcases)) * 666666).to('cuda')
        # for x in range(10):
        #     tmp = JointDistance(img, text, center[:, x].t().unsqueeze(0), cov[x], sigma1[x], sigma2[x])
        #     dist[torch.gt(dist, tmp)] = tmp[torch.gt(dist, tmp)]
        dist = dist.cpu().detach().numpy()
    # 按行返回数组值从小到大的索引
    ord = dist.argsort()

    # 统计每个label下的样本数
    lab_uni = np.arange(max(label) + 1)
    lab_sta = []
    for l in lab_uni:
        lab_sta.append(np.sum(label == l))

    numcases = dist.shape[0]
    precision, recall = [], []
    for i in range(numcases):
        order = ord[i]
        right = 0.0
        p, r = [], []
        for j in range(dist.shape[1]):
            if label[i] == label[order[j]]:
                right += 1
            p.append(right / (j + 1))
            r.append(right / lab_sta[label[i]])
        precision.append(np.array(p).reshape(1, -1))
        recall.append(np.array(r).reshape(1, -1))
    # precision和recall
    precision = np.mean(np.concatenate(precision), axis=0)
    recall = np.mean(np.concatenate(recall), axis=0)

    # 画图
    fig = plt.figure(num=figure_num)
    # precision-recall曲线
    # 数据点太多，影响描点效果，从中选择一部分
    # 将[0, 1]划成20个小区间（每段0.05），每个区间选择一个p-r点
    recall_list, precision_list = [], []
    start = 0.0
    for i in range(recall.shape[0]):
        if recall[i] == 1.0:
            recall_list.append(recall[i])
            precision_list.append(precision[i])
            break
        if recall[i] >= start:
            recall_list.append(recall[i])
            precision_list.append(precision[i])
            start += 0.05
    # 左子图
    ax1 = fig.add_subplot(1, 1, 1)
    # legend_label = ['DSCMR', 'SDM']
    # ax1.legend(legend_label, loc='center right')
    ax1.plot(recall_list, precision_list, color=color, marker=marker, linestyle=linestyle, linewidth=1.2, markersize=4)
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_xlim(0, 1.05)
    ax1.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    y_ticks = 0.1 * np.arange(int(min(precision_list) / 0.1), int(max(precision_list) / 0.1 + 2), 1)
    ax1.set_ylim(0, 0.82)
    if ylim == 0.9:
        ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    else:
        ax1.set_ylim(0, 0.92)
        ax1.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # precision-scope曲线
    # 横坐标范围[k1, k2]，每隔k_step一个点
    if k1 is None:
        k1 = 1
    if k2 is None:
        k2 = dist.shape[1]
    scope = list(range(k1, k2, k_step)) + [k2]
    precision_scope = [precision[i - 1] for i in scope]
    # # 右子图
    # ax2 = fig.add_subplot(1, 2, 2)
    # # legend_label = ['DSCMR', 'SDM']
    # # ax2.legend(legend_label, loc='center right')
    # ax2.plot(scope, precision_scope, marker=marker, linestyle=linestyle, linewidth=1, markersize=4)
    # ax2.set_xlabel('Scope')
    # ax2.set_ylabel('Precision')
    # ax2.set_xlim(0, dist.shape[1] + 10)
    # y_ticks = 0.1 * np.arange(int(min(precision_scope) / 0.1), int(max(precision_scope) / 0.1 + 2), 1)
    # ax2.set_ylim(y_ticks[0], y_ticks[-1])
    # ax2.set_yticks(y_ticks)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    #
    # # 调整两个子图的位置
    plt.subplots_adjust(left=0.10, bottom=0.15, right=0.95, top=0.95)


if __name__ == '__main__':
    # 创建figure
    img_query = plt.figure(num=1, figsize=[5.2, 4])
    plt.grid()
    txt_query = plt.figure(num=2, figsize=[5.2, 4])
    plt.grid()
    # 记录图例label
    legend_label = []




    # 模型一
    DSCMR_data = np.load('/opt/data/private/text/CLIP4CMR_master/PAN.npz')  # Img2Txt: 0.5026, Txt2Img: 0.4448
    img_feature = DSCMR_data['image']
    txt_feature = DSCMR_data['text']
    labels = DSCMR_data['label']
    legend_label.append('CCA')
    PrecisionRecall_PrecisionScope_Curve(img_feature, txt_feature, labels, ylim=0.9, color='lime', k1=50,
                                         k_step=20,
                                         figure_num=1, marker='<', linestyle='-', dist_method='L2')
    PrecisionRecall_PrecisionScope_Curve(txt_feature, img_feature, labels, color='lime', ylim=0.8, k1=50,
                                         k_step=20,
                                         figure_num=2, marker='<', linestyle='-', dist_method='L2')
    '''
    # 模型一
    DSCMR_data = np.load('baselines/KCCA.npz')  # Img2Txt: 0.5026, Txt2Img: 0.4448
    img_feature = DSCMR_data['image']
    txt_feature = DSCMR_data['text']
    labels = DSCMR_data['label']
    legend_label.append('KCCA')
    PrecisionRecall_PrecisionScope_Curve(img_feature, txt_feature, labels, ylim=0.9, color='teal', k1=50,
                                         k_step=20,
                                         figure_num=1, marker='<', linestyle='-', dist_method='L2')
    PrecisionRecall_PrecisionScope_Curve(txt_feature, img_feature, labels, color='teal', ylim=0.8, k1=50,
                                         k_step=20,
                                         figure_num=2, marker='<', linestyle='-', dist_method='L2')

    '''

    # 设置图例，保存图片
    # img_query.legend(legend_label, loc='upper right', bbox_to_anchor=(0.82, 1))
    img_query.legend(legend_label, loc='upper right', fontsize='medium')
    plt.rc('axes', titlesize=12)  # fontsize of the axes title
    plt.rc('axes', labelsize=12)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=12)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=12)  # fontsize of the tick labels
    plt.rc('legend', fontsize=12)  # legend fontsize
    img_query.savefig('/opt/data/private/text/CLIP4CMR_master/img-query.pdf')
    txt_query.legend(legend_label, loc='upper right', fontsize='medium')
    # txt_query.tight_layout()  # 裁剪掉空白
    txt_query.savefig('/opt/data/private/text/CLIP4CMR_master/txt-query.pdf')

    sys.exit()