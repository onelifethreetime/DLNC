import numpy as np
from numpy.random import randint
import random
import math


# def get_mask(view_num, data_size, missing_ratio):
#     """
#     :param view_num: number of views
#     :param data_size: size of data
#     :param missing_ratio: missing ratio
#     :return: mask matrix
#     """
#     assert view_num >= 2
#     miss_sample_num = math.floor(data_size*missing_ratio)
#     data_ind = [i for i in range(data_size)]
#     random.shuffle(data_ind)
#     miss_ind = data_ind[:miss_sample_num]
#     mask = np.ones([data_size, view_num])
#     for j in range(miss_sample_num):
#         while True:
#             rand_v = np.random.rand(view_num)
#             v_threshold = np.random.rand(1)
#             observed_ind = (rand_v >= v_threshold)
#             ind_ = ~observed_ind
#             rand_v[observed_ind] = 1
#             rand_v[ind_] = 0
#             if np.sum(rand_v) > 0 and np.sum(rand_v) < view_num:
#                 break
#         mask[miss_ind[j]] = rand_v
#
#     return mask

def get_mask(view_num, data_size, missing_ratio):

    #d_index = np.arange(data_size)
    state = np.random.seed(2001)
    d_index = np.arange(data_size)
    np.random.shuffle(d_index)
    paired_mark = int(math.ceil(data_size *missing_ratio / 10))
    mask = np.zeros(shape=(data_size, view_num))
    mask[d_index[:paired_mark]] = 1
    o_index = d_index[paired_mark:]
    l = math.ceil(o_index.shape[0] / 2)
    mask[o_index[:l], 0] = 1
    mask[o_index[l:], 1] = 1
    print(np.shape(np.where(mask[:, 0] == 0)))
    print(np.shape(np.where(mask[:, 1] == 0)))

    return mask