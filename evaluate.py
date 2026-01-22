"""
evaluation indicators, including mAP, recall@K.
mAP is the most common metric in supervised cross-modal retrieval, which measures the performance of the retrieval model on each category.
"""
import numpy as np
import scipy.spatial
import torch
from collections import Counter

def fx_calc_map_label(image, text, label, k=0, dist_method='COS'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')
    ord = dist.argsort() # [batch, batch]
    numcases = dist.shape[0]
    if k == 0:
      k = numcases
    res = []
    for i in range(numcases):
        order = ord[i]
        p = 0.0
        r = 0.0
        for j in range(k):
            if label[i] == label[order[j]]:
                r += 1
                p += (r / (j + 1))
        if r > 0:
            res += [p / r]
        else:
            res += [0]

    return np.mean(res)

# def fx_calc_map_label(image, text, label, k=0, dist_method='COS'):
#     if dist_method == 'L2':
#         dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
#     elif dist_method == 'COS':
#         dist = scipy.spatial.distance.cdist(image, text, 'cosine')
#     ord = dist.argsort()
#     numcases = dist.shape[0]
#     sim = (np.dot(label, label.T) > 0).astype(float)
#     tindex = np.arange(numcases, dtype=float) + 1
#     if k == 0:
#         k = numcases
#     res = []
#     for i in range(numcases):
#         order = ord[i]
#         sim[i] = sim[i][order]
#         num = sim[i].sum()
#         a = np.where(sim[i]==1)[0]
#         sim[i][a] = np.arange(a.shape[0], dtype=float) + 1
#         res += [(sim[i] / tindex).sum() / num]
#
#     return np.mean(res)

def fx_calc_recall(image, text, label, k=0, dist_method='L2'):
    if dist_method == 'L2':
        dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
    elif dist_method == 'COS':
        dist = scipy.spatial.distance.cdist(image, text, 'cosine')

    ord = dist.argsort() # [batch, batch]
    ranks = np.zeros(image.shape[0])

    # R@K
    for i in range(image.shape[0]):
        q_label = label[i]
        r_labels = label[ord[i]]
        ranks[i] = np.where(r_labels == q_label)[0][0]
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Prec@K
    for K in [1, 2, 4, 8, 16]:
        prec_at_k = calc_precision_at_K(ord, label, K)
        print("P@{} : {:.3f}".format(k, 100 * prec_at_k))

    return r1, r5, r10


def calc_mean(label, res, num_class):
    num_list = [0 for i in range(10)]
    value_list = [0 for i in range(10)]
    for i in range(len(res)):
        num_list[label[i]] += 1
        value_list[label[i]] += res[i]
    for i in range(num_class):
        if num_list[i] != 0:
            value_list[i] = value_list[i]/num_list[i]
            value_list[i] = round(value_list[i], 4)
        else:
            value_list[i] = 0
    return value_list

def fx_calc_map_multilabel_k(retrieval, retrieval_labels, query, query_label, k=0, metric='cosine'):
    dist = scipy.spatial.distance.cdist(query, retrieval, metric)
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    for i in range(numcases):
        order = ord[i].reshape(-1)

        tmp_label = (np.dot(retrieval_labels[order], query_label[i]) > 0)
        if tmp_label.sum() > 0:
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
            total_pos = float(tmp_label.sum())
            if total_pos > 0:
                res += [np.dot(tmp_label, prec) / total_pos]
    return np.mean(res)

def fx_calc_map_multilabel_k1(image, text, label, k=0, dist_method='L2'):
  if dist_method == 'L2':
    dist = scipy.spatial.distance.cdist(image, text, 'euclidean')
  elif dist_method == 'COS':
    dist = scipy.spatial.distance.cdist(image, text, 'cosine')
  ord = dist.argsort()
  numcases = dist.shape[0]
  if k == 0:
    k = numcases
  res = []
  for i in range(numcases):
    order = ord[i].reshape(-1)

    tmp_label = (np.dot(label[order], label[i]) > 0)
    if tmp_label.sum() > 0:
      prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])
      total_pos = float(tmp_label.sum())
      if total_pos > 0:
        res += [np.dot(tmp_label, prec) / total_pos]

  return np.mean(res)