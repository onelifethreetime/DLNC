from __future__ import print_function, absolute_import
from audioop import cross
import time
from utils.meters import AverageMeter
import torch.nn as nn
import torch
from torch.nn import functional as F
import math




def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx 
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W
def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory_text = memory
        self.memory_image = memory
        # self.tri = TripletLoss_ADP(alpha = 1, gamma = 1, square = 1)
    def train(self, epoch, data_loader_image,data_loader_text, optimizer, print_freq=10, train_iters=400, i2t=None, t2i=None):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        criterion_tri = OriTripletLoss(256, 0.3) # (batchsize, margin)


        end = time.time()
        for i in range(train_iters):
            # load data
            inputs_image = data_loader_image.next()
            inputs_text = data_loader_text.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs_image, labels_image, indexes_image = self._parse_data_image(inputs_image)
            inputs_text,labels_text, indexes_text = self._parse_data_text(inputs_text)
            # KL any?

            # forward
            f_out_image,f_out_text = self._forward(inputs_image,inputs_text)


            # intra-modality nce loss
            loss_image = self.memory_image(f_out_image, labels_image)
            loss_text= self.memory_text(f_out_text, labels_text)
            # row_ind text_index col_ind_index image_index i2r text-to-image r2i image-to-text
            # cross contrastive learning
            if t2i:
                image2text_labels = torch.tensor([i2t[key.item()] for key in labels_image]).cuda()
                text2image_labels = torch.tensor([t2i[key.item()] for key in labels_text]).cuda()
                alternate = False
                if alternate:
                    # accl
                    if epoch % 2 == 1:
                        cross_loss = 1 * self.memory_image(f_out_text,text2image_labels.long())
                    else:
                        cross_loss = 1 * self.memory_text(f_out_image, image2text_labels .long())
                else:
                    cross_loss = self.memory_image(f_out_text, text2image_labels.long()) + self.memory_text(f_out_image, image2text_labels .long())
                    # Unidirectional
                    # cross_loss = self.memory_rgb(f_out_ir, ir2rgb_labels.long())
                    # cross_loss = self.memory_ir(f_out_rgb, rgb2ir_labels.long()) 
            else:
                cross_loss = torch.tensor(0.0)

            new_loss_rgb = loss_text
            new_cross_loss = cross_loss
            
            loss =loss_image+new_loss_rgb+0.25*new_cross_loss # total loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss ir {:.3f}\t'
                      'Loss rgb {:.3f}\t'
                      'Loss cross {:.3f}\t'
                    #   'Loss tri rgb {:.3f}\t'
                    #   'Loss tri ir {:.3f}\t'
                      .format(epoch, i + 1, len(data_loader_image),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,loss_image,new_loss_rgb,new_cross_loss
                            #   , loss_tri_rgb
                            # , loss_tri_ir
                              ))

    def _parse_data_image(self, inputs):
        imgs, pseudo_labels, indexes = inputs
        return imgs.cuda(), pseudo_labels.cuda(), indexes.cuda()

    def _parse_data_text(self, inputs):
        texts, pseudo_labels, indexes = inputs
        return texts.cuda(), pseudo_labels.cuda(), indexes.cuda()

    def _forward(self, x1, x2):
        return self.encoder(x1, x2)


class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct