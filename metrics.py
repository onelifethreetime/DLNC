import torch
import numpy as np
import torch.nn.functional as F
from torch import nn


class dual_softmax_loss(nn.Module):
    def __init__(self, ):
        super(dual_softmax_loss, self).__init__()

    def forward(self, sim_matrix, temp=1000):
        sim_matrix = sim_matrix * F.softmax(sim_matrix / temp, dim=0) * len(sim_matrix)  # With an appropriate temperature parameter, the model achieves higher performance
        logpt = F.log_softmax(sim_matrix, dim=-1)  # row softmax and column softmax
        logpt = torch.diag(logpt)
        loss = -logpt
        return loss


def log_sum_exp(x):
    '''Utility function for computing log-sun-exp while determining
    This will be used to determine unaveraged confidence loss across all examples in a batch.
    '''
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), -1, keepdim=True)) + x_max


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def calcdist(img, txt):
    '''
    Input img = (batch,dim), txt = (batch,dim)
    Output Euclid Distance Matrix = Tensor(batch,batch), and dist[i,j] = d(img_i,txt_j)
    '''
    dist = img.unsqueeze(1) - txt.unsqueeze(0)
    dist = torch.sum(torch.pow(dist, 2), dim=2)
    return torch.sqrt(dist)


def calcmatch(label):
    '''
    Input label = (batch,)
    Output Match Matrix =Tensor(batch,batch) and match[i,j] == 1 iff. label[i]==label[j]
    '''
    match = label.unsqueeze(1) - label.unsqueeze(0)
    match[match != 0] = 1
    return 1 - match


def calcneg(dist, label, anchor, positive):
    '''
    Input dist = (batch,batch), label = (batch,), anchor = index, positive = index
    Output chosen negative sample index
    '''

    standard = dist[anchor, positive]  # positive distance
    dist = dist[anchor] - standard  # distance of other samples
    if max(dist[label != label[anchor]]) >= 0:  # there exists valid negative
        dist[dist < 0] = max(dist) + 2  # delete negative samples below standard
        dist[label == label[anchor]] = max(dist) + 2  # delete positive samples
        return int(torch.argmin(dist).cpu())  # return the closest negative sample
    else:  # choose argmax
        dist[label == label[anchor]] = min(dist) - 2  # delete positive samples
        return int(torch.argmax(dist).cpu())


def calcneg_dot(img, txt, match, anchor, positive):
    '''
    Input img = (batch,dim), txt = (batch,dim), match = (batch,batch), anchor = index, positive = index
    Output chosen negative sample index
    '''
    distdot = torch.sum(torch.mul(img.unsqueeze(1), txt.unsqueeze(0)), 2)
    distdot[match == 1] = -66666
    return int(torch.argmax(distdot[anchor]).cpu())


def Triplet(img, txt, label):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,)
    Output dist = (batch,batch),match = (batch,batch), triplets = List with shape(pairs,3)
    '''
    batch = img.shape[0]
    dist = calcdist(img, txt)
    match = calcmatch(label)
    match_n = match.cpu().numpy()
    positive_list = np.argwhere(match_n == 1).tolist()   # the index list of all positive samples
    for positive in positive_list:
        negative = calcneg(dist, label, positive[0], positive[1])  # calculate negatives
        # negative = calcneg_dot(img, txt, match, anchor, positive)  # calculate negative with dot  效果很差
        triplet_list.append([positive[0], int(positive[1].cuda()), negative])

    return dist, match, triplet_list


def Positive(img, txt, label):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,)
    Output dist = (batch,batch),match = (batch,batch), positives = List with shape(pairs,2)
    Remark: return (anchor,positive) without finding triplets
    '''
    batch = img.shape[0]
    dist = calcdist(img, txt)
    match = calcmatch(label)
    sample_list = torch.tensor([x for x in range(batch)]).int().cuda()
    positive_list = [[i, int(j.cpu())] for i in range(batch) for j in sample_list[label == label[i]]]
    return dist, match, positive_list


def Modality_invariant_Loss(img, txt, label, margin=0.2):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    Calculate invariant loss between images and texts belonging to the same class
    '''
    batch = img.shape[0]
    dist = calcdist(img, txt)
    dist = torch.pow(dist, 2)
    match = calcmatch(label)  # similar is 1, dissimilar is 0
    pos = torch.mul(dist, match)
    loss = torch.sum(pos)

    return loss / batch


def Contrastive_Loss(img, txt, label, margin=0.2):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    Calculate triplet loss
    '''
    batch = img.shape[0]
    dist = calcdist(img, txt)
    dist = torch.pow(dist, 2)
    match = calcmatch(label)  # similar is 1, dissimilar is 0
    pos = torch.mul(dist, match)
    neg = margin - torch.mul(dist, 1-match)
    neg = torch.clamp(neg, 0)
    loss = torch.sum(pos) + torch.sum(neg)

    return loss / batch


def Triplet_Loss(img, txt, label, margin=0.2, semi_hard=True):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    Calculate triplet loss
    '''
    loss = 0
    dist = calcdist(img, txt)
    dist = torch.pow(dist, 2)
    match = calcmatch(label)
    match_n = match.cpu().numpy()
    positive = np.argwhere(match_n == 1).tolist()  # the index list of all positive samples
    for x in positive:
        # # Semi-Hard Negative Mining
        if semi_hard:
            neg_index = torch.where(match[x[0]] == 0)  # the index list of all negative samples (shared by image and text)
            neg_dis = dist[x[0]][neg_index]
            tmp = dist[x[0], x[1]] - neg_dis + margin
            tmp = torch.clamp(tmp, 0)
            loss = loss + torch.sum(tmp, dim=-1)
        else:
            # Hard Negative Mining
            negative = calcneg(dist, label, x[0], x[1])  # calculate hard negative
            tmp = dist[x[0], x[1]] - dist[x[0], negative] + margin
            if tmp > 0:
                loss = loss + tmp

    return loss / len(positive)


def Lifted_Loss(img, txt, label, margin=1):  # the margin is set to be 1 as the original paper
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    Calculate lifted structured embedding loss
    '''
    # dist, match, positive = Positive(img, txt, label)
    dist = calcdist(img, txt)
    match = calcmatch(label)
    match_n = match.cpu().numpy()
    positive = np.argwhere(match_n == 1).tolist()  # the index list of all positive samples
    loss = 0
    for x in positive:
        neg_index = torch.where(match[x[0]] == 0)   # the index list of all negative samples (shared by image and text)
        neg_dis_anchor = dist[x[0]][neg_index]
        neg_dis_postive = dist[x[1]][neg_index]
        tmp = dist[x[0], x[1]] + log_sum_exp(margin - neg_dis_postive) + log_sum_exp(margin - neg_dis_anchor)
        loss = loss + tmp

    return loss / (2 * len(positive))


def Npairs(img, txt, label, margin=0.2, alpha=0.1):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter, alpha = parameter
    Calculate N-pairs loss
    '''
    # dist, match, positive = Positive(img, txt, label)
    batch = img.shape[0]
    distdot_it = torch.exp(F.linear(img, txt))
    distdot_ti = torch.t(distdot_it)
    match = calcmatch(label)
    match_n = match.cpu().numpy()
    positive = np.argwhere(match_n == 1).tolist()   # the index list of all positive samples
    loss = 0
    for x in positive:
        neg_index = torch.where(match[x[0]] == 0)  # the index list of all negative samples
        tmp_i2t = distdot_it[x[0], x[1]] - log_sum_exp(distdot_it[x[0]][neg_index])
        tmp_t2i = distdot_ti[x[0], x[1]] - log_sum_exp(distdot_ti[x[0]][neg_index])
        loss = loss + (tmp_i2t + tmp_t2i)/2
    loss = -loss / len(positive)
    for x in range(batch):
        loss = loss + alpha * (torch.norm(img[x]) + torch.norm(txt[x])) / batch

    return loss


# def Supervised_CrossModal_Contrastive_Loss(img, txt, label,tau=0.2):
#     '''
#     Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
#     An unofficial implementation of supervised contrastive loss for multimodal learning
#     '''
#     loss = 0
#     batch = img.shape[0]
#     sim = img.mm(txt.t())
#     #sim = sim - sim.diag().diag()
#     sim=sim/tau
#     #dist = calcdist(img, txt)
#     #dist = torch.pow(dist, 2)
#     #dist = dist / (torch.sum(dist) / (batch * batch)) # scale the metric
#     match = calcmatch(label)  # 相似为1，不相似为0
#     match_n = match.cpu().numpy()
#     positive = np.argwhere(match_n == 1).tolist()  # the index list of all positive samples
#     loss = 0
#     for x in positive:
#         neg_index = torch.where(match[x[0]] == 0)  # the index list of all negative samples
#         pos_sim = sim[x[0], x[1]]
#         #neg_sims = sim[x[0]][neg_index]
#         tmp = pos_sim - log_sum_exp(sim[0])
#         #tmp = pos_sim - log_sum_exp(neg_sims)
#
#         loss = loss + tmp
#
#     loss = -loss / len(positive)
#
#     return loss

# def Supervised_IntraModal_Contrastive_Loss(img, label,tau=0.2):
#     '''
#     Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
#     An unofficial implementation of supervised contrastive loss for multimodal learning
#     '''
#     loss = 0
#     batch = img.shape[0]
#     sim = img.mm(img.t())
#     sim=sim - sim.diag().diag()
#     sim=sim/tau
#     #dist = calcdist(img, txt)
#     #dist = torch.pow(dist, 2)
#     #dist = dist / (torch.sum(dist) / (batch * batch)) # scale the metric
#     match = calcmatch(label)  # 相似为1，不相似为0
#     match_n = match.cpu().numpy()
#     positive = np.argwhere(match_n == 1).tolist()  # the index list of all positive samples
#     loss = 0
#     for x in positive:
#         neg_index = torch.where(match[x[0]] == 0)  # the index list of all negative samples
#         pos_sim = sim[x[0], x[1]]
#         #neg_sims = sim[x[0]][neg_index]
#         #tmp=pos_sim-torch.log(torch.sum(torch.exp(neg_sims))+pos_sim)
#         tmp=pos_sim-log_sum_exp(sim)
#         #tmp = pos_sim - log_sum_exp(neg_sims)
#
#         loss = loss + tmp
#
#     loss = -loss / len(positive)
#     #print('intra_modal_image-text_loss')
#     #print(loss)
#     return loss

# def Supervised_CrossModal_Contrastive_Loss(img, txt, label,tau=0.2):
#     '''
#     Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
#     An unofficial implementation of supervised contrastive loss for multimodal learning
#     '''
#     loss = 0
#     batch = img.shape[0]
#     sim = img.mm(txt.t())
#     #sim = sim - sim.diag().diag()
#     sim=sim/tau
#     #dist = calcdist(img, txt)
#     #dist = torch.pow(dist, 2)
#     #dist = dist / (torch.sum(dist) / (batch * batch)) # scale the metric
#     match = calcmatch(label)  # 相似为1，不相似为0
#     match_n = match.cpu().numpy()
#     positive = np.argwhere(match_n == 1).tolist()  # the index list of all positive samples
#     loss = 0
#     for x in positive:
#         neg_index = torch.where(match[x[0]] == 0)  # the index list of all negative samples
#         pos_sim = sim[x[0], x[1]]
#         neg_sims = sim[x[0]][neg_index]
#         sim1=sim[x[0]]
#         total_sim=log_sum_exp(sim1)
#         tmp = pos_sim - total_sim
#         loss = loss + tmp
#
#     loss = -loss / len(positive)
#
#     return loss

def Supervised_CrossModal_Contrastive_Loss(img, txt,tau=0.2):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    An unofficial implementation of supervised contrastive loss for multimodal learning
    '''
    batch = img.shape[0]
    sim = img.mm(txt.t())
    loss=0
    for i in range(batch):

        # sim_max=sim[i].data.max()
        # sim[i]=sim[i]-sim_max
        # sim[i]=(sim[i] / tau).exp()
        pos_sim=sim[i][i]
        total_sim=log_sum_exp(sim[i])
        tmp=pos_sim-total_sim
        loss=loss+tmp
    loss=-loss/batch
    #sim = sim - sim.diag().diag()
    #sim=(sim/tau).exp()

    #diag=sim.diag()
    #=-(diag/sim.sum(1)).log().mean()
    #dist = calcdist(img, txt)
    #dist = torch.pow(dist, 2)
    #dist = dist / (torch.sum(dist) / (batch * batch)) # scale the metric
    # match = calcmatch(label)  # 相似为1，不相似为0
    # match_n = match.cpu().numpy()
    # positive = np.argwhere(match_n == 1).tolist()  # the index list of all positive samples
    # loss = 0
    # for x in positive:
    #     neg_index = torch.where(match[x[0]] == 0)  # the index list of all negative samples
    #     pos_sim = sim[x[0], x[1]]
    #     neg_sims = sim[x[0]][neg_index]
    #     sim1=sim[x[0]]
    #     total_sim=log_sum_exp(sim1)
    #     tmp = pos_sim - total_sim
    #     loss = loss + tmp
    #
    # loss = -loss / len(positive)

    return loss

def Supervised_IntraModal_Contrastive_Loss(img, label,tau=0.2):
    '''
    Input img = (batch,dim), txt = (batch,dim), label = (batch,), margin = parameter
    An unofficial implementation of supervised contrastive loss for multimodal learning
    '''
    loss = 0
    batch = img.shape[0]
    sim = img.mm(img.t())
    sim=sim - sim.diag().diag()
    sim=sim/tau

    #dist = calcdist(img, txt)
    #dist = torch.pow(dist, 2)
    #dist = dist / (torch.sum(dist) / (batch * batch)) # scale the metric
    #match = calcmatch(label)  # 相似为1，不相似为0
    match=label
    match_n = match.cpu().numpy()
    positive = np.argwhere(match_n == 1).tolist()  # the index list of all positive samples
    loss = 0
    for x in positive:
        neg_index = torch.where(match[x[0]] == 0)  # the index list of all negative samples
        pos_sim = sim[x[0], x[1]]
        neg_sims = sim[x[0]][neg_index]
        #tmp=pos_sim-torch.log(torch.sum(torch.exp(neg_sims))+pos_sim)
        tmp=0.0
        if x[0]!=x[1]:
            sim1=sim[x[0]]
            total_sim=log_sum_exp(sim1)
            #if total_sim==nan:
            tmp=pos_sim-total_sim

        loss = loss + tmp

    loss = -loss / (len(positive)-batch)
    #print('intra_modal_image-text_loss')
    #print(loss)
    return loss

def regularization(features, centers, labels):
    # features = l2norm(features, dim=-1)
    distance = (features - centers[labels])
    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)
    distance = (torch.sum(distance, 0, keepdim=True)) / features.shape[0]

    return distance


def PAN(features, centers, labels, add_regularization=False):
    """The prototype contrastive loss and regularization loss in
    PAN(https://dl.acm.org/doi/abs/10.1145/3404835.3462867)"""
    batch = features.shape[0]
    features_square = torch.sum(torch.pow(features, 2), 1, keepdim=True)  # 在第一个维度上平方
    centers_square = torch.sum(torch.pow(torch.t(centers), 2), 0, keepdim=True)
    features_into_centers = 2 * torch.matmul(features, torch.t(centers))
    dist = -(features_square + centers_square - features_into_centers)
    output = F.log_softmax(dist, dim=1)
    dce_loss = F.nll_loss(output, labels)

    if add_regularization:
        reg_loss = regularization(features, centers, labels)
        loss = dce_loss + reg_loss

    loss = dce_loss

    return loss / batch


def Label_Regression_Loss(view1_predict, view2_predict, label_onehot):
    loss = ((view1_predict - label_onehot.float()) ** 2).sum(1).sqrt().mean() + (
                (view2_predict - label_onehot.float()) ** 2).sum(1).sqrt().mean()

    return loss

def cross_modal_contrastive_ctriterion(fea, tau=0.2):
    batch_size = fea[0].shape[0]
    all_fea = torch.cat(fea)
    sim = all_fea.mm(all_fea.t())

    sim = (sim / tau).exp()
    sim = sim - sim.diag().diag()
    sim_sum1 = sum([sim[:, v * batch_size: (v + 1) * batch_size] for v in range(2)])
    diag1 = torch.cat([sim_sum1[v * batch_size: (v + 1) * batch_size].diag() for v in range(2)])
    loss1 = -(diag1 / sim.sum(1)).log().mean()

    sim_sum2 = sum([sim[v * batch_size: (v + 1) * batch_size] for v in range(2)])
    diag2 = torch.cat([sim_sum2[:, v * batch_size: (v + 1) * batch_size].diag() for v in range(2)])
    loss2 = -(diag2 / sim.sum(1)).log().mean()
    return loss1 + loss2


import torch
import torch.nn as nn


class CrossModalCenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self,device, num_classes, feat_dim=512, use_gpu=True,init_weight=True):
        super(CrossModalCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.device=device
        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(self.device))
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
        if init_weight:
            self.__init_weight()

    def __init_weight(self):
        nn.init.kaiming_normal_(self.centers, mode='fan_out')
        #nn.init.kaiming_normal_(self.predictLayer.weight.data, mode='fan_out')

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """

        self.centers.data = l2norm(self.centers.data, dim=1)
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

def calc_label_sim(label_1, label_2):
    Sim = label_1.float().mm(label_2.float().t())
    return Sim

def calc_loss(view1_feature, view2_feature, view1_predict, view2_predict, labels_1_onthot, labels_2_one_hot, alpha=1e-3, beta=1e-1):
    term1 = ((view1_predict-labels_1_onthot.float())**2).sum(1).sqrt().mean() + ((view2_predict-labels_2_one_hot.float())**2).sum(1).sqrt().mean()

    cos = lambda x, y: x.mm(y.t()) / ((x ** 2).sum(1, keepdim=True).sqrt().mm((y ** 2).sum(1, keepdim=True).sqrt().t())).clamp(min=1e-6) / 2.
    theta11 = cos(view1_feature, view1_feature)
    theta12 = cos(view1_feature, view2_feature)
    theta22 = cos(view2_feature, view2_feature)
    Sim11 = calc_label_sim(labels_1_onthot, labels_1_onthot).float()
    Sim12 = calc_label_sim(labels_1_onthot, labels_2_one_hot).float()
    Sim22 = calc_label_sim(labels_2_one_hot, labels_2_one_hot).float()
    term21 = ((1+torch.exp(theta11)).log() - Sim11 * theta11).mean()
    term22 = ((1+torch.exp(theta12)).log() - Sim12 * theta12).mean()
    term23 = ((1 + torch.exp(theta22)).log() - Sim22 * theta22).mean()
    term2 = term21 + term22 + term23

    term3 = ((view1_feature - view2_feature)**2).sum(1).sqrt().mean()

    im_loss = term1 + alpha * term2 + beta * term3
    return im_loss
'''
def Label_Regression_Loss(view1_predict, view2_predict, label_onehot):
    loss = ((view1_predict - label_onehot.float()) ** 2).sum(1).sqrt().mean() + (
                (view2_predict - label_onehot.float()) ** 2).sum(1).sqrt().mean()

    return loss
'''
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class ViCC(nn.Module):
    '''
    Basically, it's a MoCo for video input: https://arxiv.org/abs/1911.05722
    With SwaV clustering and without momentum encoder https://arxiv.org/abs/2006.09882
    '''

    def __init__(self,
                 dim=1024,
                 K=1920,
                 m=0.999,
                 T=0.1,
                 ################
                 nmb_prototypes=20,
                 nmb_views=[2],
                 world_size=1,
                 epsilon=0.05,
                 sinkhorn_iterations=3,
                 views_for_assign=0,
                 improve_numerical_stability=False,
                 #################
                 ):
        '''
        dim (D): feature dimension (default: 128)
        K: queue size; number of negative keys (default: 2048 CoCLR / 3840 SwaV / 1920 ViCC)
        m: moco momentum of updating key encoder (default: 0.999), only used for MoCo/CoCLR
        T: softmax temperature (default: 0.07 CoCLR / 0.1 Swav,ViCC)
        nmb_prototypes (C) (default: 300)
        nmb_views: amount of views used in a list, e.g. [2] or [2,2]
        epsilon: regularization parameter for Sinkhorn-Knopp algorithm
        sinkhorn_iterations: number of iterations in Sinkhorn-Knopp algorithm
        views_for_assign: list of views id used for computing assignments
        improve_numerical_stability: improves numerical stability in Sinkhorn-Knopp algorithm
        '''
        super(ViCC, self).__init__()

        self.dim = dim
        self.K = K
        self.m = m
        self.T = T

        #######################
        self.nmb_prototypes = nmb_prototypes
        self.nmb_views = nmb_views
        self.world_size = world_size
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.views_for_assign = views_for_assign
        self.improve_numerical_stability = improve_numerical_stability
        #print("=> viewsfa:", self.views_for_assign, "nmb views:", self.nmb_views)

        # create the encoder (including non-linear projection head: 2 FC layers)
        #backbone, self.param = select_backbone(network)
        #feature_size = self.param['feature_size']
        '''
        self.encoder_q = nn.Sequential(
            backbone,
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(feature_size, feature_size, kernel_size=1, bias=True),
            nn.ReLU(),
            nn.Conv3d(feature_size, dim, kernel_size=1, bias=True))
        '''
        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(dim, nmb_prototypes, bias=False)  # Should be dim (D) x nmb_prototypes (C)

        self.softmax = nn.Softmax(dim=1).cuda()
        # self.cos = nn.CosineSimilarity(dim=1)
        self.use_the_queue = False
        self.start_queue = False
        self.queue = None
        self.batch_shuffle = None

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        '''
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]

        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        '''
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        '''
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward_head(self, x):
        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, block):
        #block 2xBxdim dim 1024
        '''Output: logits, targets'''
        (N, B, *_) = block.shape  # [B,N,C,T,H,W] e.g. [16, 2, 3, 32, 128, 128]
        assert N == 2
        x1 = block[0, :, :].contiguous()
        x2 = block[1, :, :].contiguous()

        # compute features of x1
        #s = self.encoder_q(x1)  # queries: B,C,1,1,1
        #s = nn.functional.normalize(s, dim=1)
        s = x1.view(B, self.dim)  # To B, C e.g. 16, 128

        # compute features of x2
        if self.batch_shuffle:
            with torch.no_grad():
                # shuffle for making use of BN
                x2, idx_unshuffle = self._batch_shuffle_ddp(x2)

                t = self.encoder_q(x2)  # keys: B,C,1,1,1
                t = nn.functional.normalize(t, dim=1)

                # undo shuffle
                t = self._batch_unshuffle_ddp(t, idx_unshuffle)
        else:
            pass
            #t = self.encoder_q(x2)  # keys: B,C,1,1,1
            #t = nn.functional.normalize(t, dim=1)

        t = x2.view(B, self.dim)  # To B, C e.g. 16, 128

        ########### SWAV ######################
        # ============ multi-res forward passes ... ============
        # Embedding: Projhead(x): 2B x D. e.g. 32 x 128. Output: Prototype scores: Prototypes(Projhead(x)): 2B x C, e.g. 32 x 300
        embedding, output = self.forward_head(torch.cat((s, t)))  # B, K # Positive examples
        embedding = embedding.detach()  # Detach embedding: we dont need gradients, this is only used to fill the queue.
        bs = B

        # ============ swav loss ... ============
        self.views_for_assign=[0,1]
        loss = 0
        for i, crop_id in enumerate(self.views_for_assign):  # views for assign: [0, 1]
            with torch.no_grad():
                out = output[bs * crop_id: bs * (crop_id + 1)]  # B x K, e.g. 16 x 300

                # time to use the queue
                if self.queue is not None:
                    if self.start_queue and (self.use_the_queue or not torch.all(self.queue[i, -1, :] == 0)):
                        use_the_queue = True
                        # Queue size must be divisible by batch size, queue is Queue_Size x dim, e.g. 3480, 128.
                        # SWAV queue is a tensor of [N, L, Feat_dim], Coclr queue is a tensor of [Feat dim, L]
                        # prototypes are dim x K, e.g. 128 x 300
                        out = torch.cat((torch.mm(
                            self.queue[i],
                            self.prototypes.weight.t()),out))  # out is 16 x 300 (for current feature), cat this with 2048 x 300, prototype scores of queue
                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]

                # get assignments
                q_swav = out / self.epsilon
                if self.improve_numerical_stability:
                    M = torch.max(q_swav)
                    import torch.distributed as dist
                    dist.all_reduce(M, op=dist.ReduceOp.MAX)
                    q_swav -= M
                q_swav = torch.exp(q_swav).t()
                q_swav = self.distributed_sinkhorn(q_swav, self.sinkhorn_iterations)[-bs:]
                # q_swav are now soft assignments B x C e.g. 16 x 300

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(self.nmb_views)), crop_id):  # Use crop ids except current crop id
                p_swav = self.softmax(output[bs * v: bs * (v + 1)] / self.T)  # B x 300

                # swap prediction problem
                subloss -= torch.mean(torch.sum(q_swav * torch.log(p_swav), dim=1))
            loss += subloss / (np.sum(self.nmb_views) - 1)
        loss /= len(self.views_for_assign)

        return embedding, output, loss

    def distributed_sinkhorn(self, Q, nmb_iters):
        with torch.no_grad():
            Q = self.shoot_infs(Q)
            sum_Q = torch.sum(Q)
            #import torch.distributed as dist
            #dist.all_reduce(sum_Q)
            Q /= sum_Q
            r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
            c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (self.world_size * Q.shape[1])
            for it in range(nmb_iters):
                u = torch.sum(Q, dim=1)
                #dist.all_reduce(u)
                u = r / u
                u = self.shoot_infs(u)
                Q *= u.unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def shoot_infs(self, inp_tensor):
        """Replaces inf by maximum of tensor"""
        mask_inf = torch.isinf(inp_tensor)
        ind_inf = torch.nonzero(mask_inf)
        if len(ind_inf) > 0:
            for ind in ind_inf:
                if len(ind) == 2:
                    inp_tensor[ind[0], ind[1]] = 0
                elif len(ind) == 1:
                    inp_tensor[ind[0]] = 0
            m = torch.max(inp_tensor)
            for ind in ind_inf:
                if len(ind) == 2:
                    inp_tensor[ind[0], ind[1]] = m
                elif len(ind) == 1:
                    inp_tensor[ind[0]] = m
        return inp_tensor


class Loss(nn.Module):
    def __init__(self, batch_size,  temperature_f,  device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature_f = temperature_f
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_feature(self, h_i, h_j):
        if h_i.shape[0]==self.batch_size:
            N = 2 * self.batch_size
            h = torch.cat((h_i, h_j), dim=0)

            sim = torch.matmul(h, h.T) / self.temperature_f
            sim_i_j = torch.diag(sim, self.batch_size)
            sim_j_i = torch.diag(sim, -self.batch_size)

            positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
            mask = self.mask_correlated_samples(N)
            negative_samples = sim[mask].reshape(N, -1)

            labels = torch.zeros(N).to(positive_samples.device).long()
            logits = torch.cat((positive_samples, negative_samples), dim=1)
            loss = self.criterion(logits, labels)
            loss /= N
        else:
            N = 2 * 109
            h = torch.cat((h_i, h_j), dim=0)

            sim = torch.matmul(h, h.T) / self.temperature_f
            sim_i_j = torch.diag(sim, 109)
            sim_j_i = torch.diag(sim, -109)

            positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
            mask = self.mask_correlated_samples(N)
            negative_samples = sim[mask].reshape(N, -1)

            labels = torch.zeros(N).to(positive_samples.device).long()
            logits = torch.cat((positive_samples, negative_samples), dim=1)
            loss = self.criterion(logits, labels)
            loss /= N

        return loss