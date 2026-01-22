from sklearn.preprocessing import MinMaxScaler
import numpy as np

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Sampler
import scipy.io
import torch
import pickle

# class Pascal(Dataset):
#     def __init__(self,load_train):
#         path = './datasets/pascal/'
#         train_image = scipy.io.loadmat(path + "images_pascal-sentences_vgg19_4096d.mat")
#         labels = scipy.io.loadmat(path + "labels_pascal-sentences.mat")
#         train_text = scipy.io.loadmat(path + "texts_pascal-sentences_doc2vec_300.mat")
#         img_train=train_image['images'][:800].astype(np.float32)
#         img_test = train_image['images'][800:].astype(np.float32)
#         text_train=train_text['texts'][:800].astype(np.float32)
#         text_test = train_text['texts'][800:].astype(np.float32)
#         label_train=labels['labels'][:,:800]
#         label_test=labels['labels'][:,800:]
#         self.x1=img_train
#         self.x2 = text_train
#         self.y = label_train
#         self.x11=img_test
#         self.x22=text_test
#         self.yy=label_test
#         self.load_train=load_train
#         #labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
#
#     def __len__(self):
#         if self.load_train == True:
#             return self.x1.shape[0]
#         else:
#             return self.x11.shape[0]
#     def __getitem__(self, idx):
#         if self.load_train==True:
#             return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], torch.from_numpy(np.array(self.y.squeeze()[idx])), torch.from_numpy(np.array(idx)).long()
#         else:
#             return [torch.from_numpy(self.x11[idx]), torch.from_numpy(self.x22[idx])], torch.from_numpy(np.array(self.yy.squeeze()[idx])), torch.from_numpy(np.array(idx)).long()

class Pascal(Dataset):
    def __init__(self,load_train):
        path_train = 'data3/pascal/clip_train.pkl'
        path_test = 'data3/pascal/clip_test.pkl'

        with open(path_train, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            label_train = data['label']-1
            text_train = data['text']
            img_train = data['image']

        with open(path_test, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            label_test = data['label']-1
            text_test = data['text']
            img_test = data['image']

        self.x1 = img_train
        self.x2 = text_train
        self.y = label_train
        self.x11 = img_test
        self.x22 = text_test
        self.yy = label_test
        self.load_train = load_train

    def __len__(self):
        if self.load_train == True:
            return self.x1.shape[0]
        else:
            return self.x11.shape[0]

    def __getitem__(self, idx):

        # if self.load_train == True:
        #     return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], torch.from_numpy(
        #         np.array(self.y.squeeze()[idx])), torch.from_numpy(np.array(idx)).long()
        # else:
        #     return [torch.from_numpy(self.x11[idx]), torch.from_numpy(self.x22[idx])], torch.from_numpy(
        #         np.array(self.yy.squeeze()[idx])), torch.from_numpy(np.array(idx)).long()
        x1 = self.x1[idx]
        x2 = self.x2[idx]
        if self.load_train == True:
            return [torch.from_numpy(x1), torch.from_numpy(x2)], torch.from_numpy(
                np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()
        else:
            return [torch.from_numpy(self.x11[idx]), torch.from_numpy(self.x22[idx])], torch.from_numpy(
                np.array(self.yy[idx])), torch.from_numpy(np.array(idx)).long()


# class Nus_wide(Dataset):
#     def __init__(self,load_train):
#         path = './datasets/nus_wide/'
#         MAP = -1
#         img_train =scipy.io.loadmat(path + "train_img.mat")['train_img']
#         img_test = scipy.io.loadmat(path + "test_img.mat")['test_img']
#         text_train = scipy.io.loadmat(path + "train_txt.mat")['train_txt']
#         text_test = scipy.io.loadmat(path + "test_txt.mat")['test_txt']
#         label_train =scipy.io.loadmat(path + "train_img_lab.mat")['train_img_lab'].reshape(8000)
#         label_test =scipy.io.loadmat(path + "test_img_lab.mat")['test_img_lab'].reshape(2000,)
#         self.x1=img_train
#         self.x2 = text_train
#         self.y = label_train
#         self.x11=img_test
#         self.x22=text_test
#         self.yy=label_test
#         self.load_train=load_train
#
#     def __len__(self):
#         if self.load_train == True:
#             return self.x1.shape[0]
#         else:
#             return self.x11.shape[0]
#     def __getitem__(self, idx):
#
#         # if self.load_train == True:
#         #     return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], torch.from_numpy(
#         #         np.array(self.y.squeeze()[idx])), torch.from_numpy(np.array(idx)).long()
#         # else:
#         #     return [torch.from_numpy(self.x11[idx]), torch.from_numpy(self.x22[idx])], torch.from_numpy(
#         #         np.array(self.yy.squeeze()[idx])), torch.from_numpy(np.array(idx)).long()
#         x1=self.x1[idx]
#         x2= self.x2[idx]
#         if self.load_train == True:
#             return [torch.from_numpy(x1), torch.from_numpy(x2)], torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()
#         else:
#             return [torch.from_numpy(self.x11[idx]), torch.from_numpy(self.x22[idx])], torch.from_numpy(np.array(self.yy[idx])), torch.from_numpy(np.array(idx)).long()

class Nus_wide(Dataset):
    def __init__(self,load_train):
        path_train = 'data3/nus-wide/clip_train.pkl'
        path_test = 'data3/nus-wide/clip_test.pkl'

        with open(path_train, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            label_train  = data['label']
            text_train = data['text']
            img_train= data['image']

        with open(path_test, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            label_test = data['label']
            text_test= data['text']
            img_test = data['image']

        self.x1=img_train
        self.x2 = text_train
        self.y = label_train
        self.x11=img_test
        self.x22=text_test
        self.yy=label_test
        self.load_train=load_train

    def __len__(self):
        if self.load_train == True:
            return self.x1.shape[0]
        else:
            return self.x11.shape[0]
    def __getitem__(self, idx):

        # if self.load_train == True:
        #     return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], torch.from_numpy(
        #         np.array(self.y.squeeze()[idx])), torch.from_numpy(np.array(idx)).long()
        # else:
        #     return [torch.from_numpy(self.x11[idx]), torch.from_numpy(self.x22[idx])], torch.from_numpy(
        #         np.array(self.yy.squeeze()[idx])), torch.from_numpy(np.array(idx)).long()
        x1=self.x1[idx]
        x2= self.x2[idx]
        if self.load_train == True:
            return [torch.from_numpy(x1), torch.from_numpy(x2)], torch.from_numpy(np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()
        else:
            return [torch.from_numpy(self.x11[idx]), torch.from_numpy(self.x22[idx])], torch.from_numpy(np.array(self.yy[idx])), torch.from_numpy(np.array(idx)).long()
class Wikipedia(Dataset):
    def __init__(self,load_train):
        path_train = 'data3/wikipedia/clip_train.pkl'
        path_test = 'data3/wikipedia/clip_test.pkl'
        MAP = -1

        with open(path_train, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            label_train  = data['label']
            text_train = data['text']
            img_train= data['image']

        with open(path_test, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            label_test = data['label']
            text_test= data['text']
            img_test = data['image']

        self.x1=img_train
        self.x2 = text_train
        self.y = label_train
        self.x11=img_test
        self.x22=text_test
        self.yy=label_test
        self.load_train=load_train

    def __len__(self):
        if self.load_train == True:
            return self.x1.shape[0]
        else:
            return self.x11.shape[0]

    def __getitem__(self, idx):

        x1 = self.x1[idx]
        x2 = self.x2[idx]
        if self.load_train == True:
            return [torch.from_numpy(x1), torch.from_numpy(x2)], torch.from_numpy(
                np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()
        else:
            return [torch.from_numpy(self.x11[idx]), torch.from_numpy(self.x22[idx])], torch.from_numpy(np.array(self.yy[idx])), torch.from_numpy(np.array(idx)).long()

# class Wikipedia(Dataset):
#     def __init__(self,load_train):
#         path = './datasets/wikipedia/'
#         MAP = -1
#         img_train = scipy.io.loadmat(path + "train_img.mat")['train_img'].astype(np.float32)
#         img_test = scipy.io.loadmat(path + "test_img.mat")['test_img'].astype(np.float32)
#         text_train = scipy.io.loadmat(path + "train_txt.mat")['train_txt'].astype(np.float32)
#         text_test = scipy.io.loadmat(path + "test_txt.mat")['test_txt'].astype(np.float32)
#         label_train = scipy.io.loadmat(path + "train_img_lab.mat")['train_img_lab']
#         label_test = scipy.io.loadmat(path + "test_img_lab.mat")['test_img_lab']
#         self.x1=img_train
#         self.x2 = text_train
#         self.y = label_train-1
#         self.x11=img_test
#         self.x22=text_test
#         self.yy=label_test-1
#         self.load_train=load_train
#
#     def __len__(self):
#         if self.load_train == True:
#             return self.x1.shape[0]
#         else:
#             return self.x11.shape[0]
#
#     def __getitem__(self, idx):
#
#         if self.load_train == True:
#             return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], torch.from_numpy(
#                 np.array(self.y.squeeze()[idx])), torch.from_numpy(np.array(idx)).long()
#         else:
#             return [torch.from_numpy(self.x11[idx]), torch.from_numpy(self.x22[idx])], torch.from_numpy(
#                 np.array(self.yy.squeeze()[idx])), torch.from_numpy(np.array(idx)).long()
class FullDataset(Dataset):
    def __init__(self, num_views, data_list, labels):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.num_views):
            data.append(torch.tensor(self.data_list[i][idx].astype('float32')))
        return data, torch.tensor(self.labels[idx]), torch.tensor(np.array(idx)).long()

class MultiviewDataset(Dataset):
    def __init__(self, num_views, data_list, labels):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, idx):
        data = []
        for i in range(self.num_views):
            data.append(torch.tensor(self.data_list[i][idx].astype('float32')))
        return data, torch.tensor(self.labels[idx]), torch.tensor(np.array(idx)).long()


class SingleviewDataset0(Dataset):
    def __init__(self, num_views, data_list, labels):
        self.num_views = num_views
        self.data_list = data_list
        self.labels = labels

    def __len__(self):
        return self.data_list[0].shape[0]

    def __getitem__(self, idx):
        return  torch.tensor(self.data_list[0][idx].astype('float32')), torch.tensor(self.labels[idx]), torch.tensor(np.array(idx)).long()

class SingleviewDataset1(Dataset):
    def __init__(self, image_list, labels):
        self.data_list = image_list
        self.labels = labels

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        return  torch.tensor(self.data_list[idx].astype('float32')), torch.tensor(self.labels[idx]), torch.tensor(np.array(idx)).long()



class RandomSampler(Sampler):
    """ sampling without replacement """
    def __init__(self, num_data, num_sample):
        iterations = num_sample // num_data + 1
        self.indices = torch.cat([torch.randperm(num_data) for _ in range(iterations)]).tolist()[:num_sample]

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

class Mscoco(Dataset):
    def __init__(self):
        path='./datasets/MSCOCO/'
        img_train =  scipy.io.loadmat(path + "train_img.mat")['train_img'].astype(np.float32)
        img_test =  scipy.io.loadmat(path + "test_img.mat")['test_img'].astype(np.float32)
        text_train = scipy.io. loadmat(path + "train_txt.mat")['train_txt'].astype(np.float32)
        text_test =  scipy.io.loadmat(path + "test_txt.mat")['test_txt'].astype(np.float32)
        label_train =  scipy.io.loadmat(path + "train_lab.mat")['train_lab']
        label_test =  scipy.io.loadmat(path + "test_lab.mat")['test_lab'] #ulti-labeled label
        self.x1=img_train
        self.x2 = text_train
        self.y = label_train

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):

        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class Xmedianet(Dataset):
    def __init__(self,load_train):
        path_train = 'data3/xmedianet/clip_train.pkl'
        path_test = 'data3/xmedianet/clip_test.pkl'
        MAP = -1

        with open(path_train, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            label_train  = data['label']
            text_train = data['text']
            img_train= data['image']

        with open(path_test, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            label_test = data['label']
            text_test= data['text']
            img_test = data['image']

        self.x1=img_train
        self.x2 = text_train
        self.y = label_train
        self.x11=img_test
        self.x22=text_test
        self.yy=label_test
        self.load_train=load_train

    def __len__(self):
        if self.load_train == True:
            return self.x1.shape[0]
        else:
            return self.x11.shape[0]

    def __getitem__(self, idx):

        x1 = self.x1[idx]
        x2 = self.x2[idx]
        if self.load_train == True:
            return [torch.from_numpy(x1), torch.from_numpy(x2)], torch.from_numpy(
                np.array(self.y[idx])), torch.from_numpy(np.array(idx)).long()
        else:
            return [torch.from_numpy(self.x11[idx]), torch.from_numpy(self.x22[idx])], torch.from_numpy(np.array(self.yy[idx])), torch.from_numpy(np.array(idx)).long()

class LabeledDataSet(Dataset):
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return img, text, label

    def __len__(self):

        count = len(self.images)
        assert len(
            self.images) == len(self.labels)
        return count

class UnLabeledDataSet(Dataset):
    def __init__(self, images, texts,labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return img, text,label

    def __len__(self):
        count = len(self.images)
        assert len(
            self.images) == len(self.labels)
        return count



def load_data(dataset,load_train=False):
    if dataset == "pascal":
        dataset = Pascal(load_train)
        dims = [1024, 1024]
        #dims = [4096, 300]
        view = 2
        if load_train==True:
            data_size =800 #train800 test 200
        else:
            data_size=200
        class_num = 20
    elif dataset == "nus_wide":
        dataset = Nus_wide(load_train)
        #dims = [4096, 1000]
        dims = [1024, 1024]
        view = 2
        class_num = 10
        if load_train==True:
            data_size =8000 #train8000 test 2000
        else:
            data_size=1000
    elif dataset == "wikipedia":
        dataset = Wikipedia(load_train)
        #dims = [4096, 5000]
        dims = [1024, 1024]
        view = 2
        #train 2173 test 693
        if load_train==True:
            data_size =2157 #train800 test 200
        else:
            data_size=462
        class_num =10
    elif dataset == "mscoco":
        dataset = Mscoco(load_train)
        dims = [4096,300]
        view = 2
        data_size = 120218 #test 200
        class_num =80
    elif dataset == "xmedianet":
        dataset = Xmedianet(load_train)
        dims = [1024,1024]
        view = 2
        if load_train==True:
            data_size =32000 #train800 test 200
        else:
            data_size=4000
        class_num =200
    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num

def load_deep_features(data_name, semi_set=False, ratio=0.5):

    if data_name == 'wikipedia':
        path_train = 'data3/wikipedia/clip_train.pkl'
        path_test = 'data3/wikipedia/clip_test.pkl'
        MAP = -1

        with open(path_train, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            label_train  = data['label']
            text_train = data['text']
            img_train= data['image']

        with open(path_test, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            label_test = data['label']
            text_test= data['text']
            img_test = data['image']

        train_labels = [label_train, label_train]

        if len(train_labels[0].shape) == 1 or train_labels[0].shape[1] == 1:
            classes = np.unique(train_labels[0])
            k = int(train_labels[0].shape[0] * ratio / classes.shape[0])

        n_view = 2
        for v in range(n_view):
            if len(train_labels[0].shape) == 1 or train_labels[0].shape[1] == 1:
                train_seen_inx = []
                train_unseen_inx = []
                for i in range(len(classes)):
                    c = classes[i]
                    inx = (c == train_labels[v].reshape([-1])).nonzero()[0]
                    state = np.random.seed(2000)
                    np.random.shuffle(inx)
                    train_seen_inx.append(inx[0: k])
                    train_unseen_inx.append(inx[k::])
                train_seen_inx = np.concatenate(train_seen_inx)
                train_unseen_inx = np.concatenate(train_unseen_inx)
            else:
                train_seen_inx = np.arange(k)
                train_unseen_inx = np.arange(k, train_labels[v].shape[0])

        se_text_train = text_train[train_seen_inx]
        se_img_train = img_train[train_seen_inx]
        se_label_train=label_train[train_seen_inx]

        un_text_train=text_train[train_unseen_inx]
        un_image_train=img_train[train_unseen_inx]
        un_label_train=label_train[train_unseen_inx]


    elif data_name == 'nus_wide':
        path_train = 'data3/nus-wide/clip_train.pkl'
        path_test = 'data3/nus-wide/clip_test.pkl'
        MAP = -1

        with open(path_train, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            label_train = data['label']
            text_train = data['text']
            img_train = data['image']

        with open(path_test, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            label_test = data['label']
            text_test = data['text']
            img_test = data['image']
        train_labels = [label_train, label_train]
        if len(train_labels[0].shape) == 1 or train_labels[0].shape[1] == 1:
            classes = np.unique(train_labels[0])
            k = int(train_labels[0].shape[0] * ratio / classes.shape[0])

        n_view = 2
        for v in range(n_view):
            if len(train_labels[0].shape) == 1 or train_labels[0].shape[1] == 1:
                train_seen_inx = []
                train_unseen_inx = []
                for i in range(len(classes)):
                    c = classes[i]
                    inx = (c == train_labels[v].reshape([-1])).nonzero()[0]
                    state = np.random.seed(2001)
                    np.random.shuffle(inx)
                    train_seen_inx.append(inx[0: k])
                    train_unseen_inx.append(inx[k::])
                train_seen_inx = np.concatenate(train_seen_inx)
                train_unseen_inx = np.concatenate(train_unseen_inx)
            else:
                train_seen_inx = np.arange(k)
                train_unseen_inx = np.arange(k, train_labels[v].shape[0])

        se_text_train = text_train[train_seen_inx]
        se_img_train = img_train[train_seen_inx]
        se_label_train = label_train[train_seen_inx]

        un_text_train = text_train[train_unseen_inx]
        un_image_train = img_train[train_unseen_inx]
        un_label_train = label_train[train_unseen_inx]

    elif data_name == 'xmedianet':
        path_train = 'data3/xmedianet/clip_train.pkl'
        path_test = 'data3/xmedianet/clip_test.pkl'
        MAP = -1

        with open(path_train, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            label_train = data['label']
            text_train = data['text']
            img_train = data['image']

        with open(path_test, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            label_test = data['label']
            text_test = data['text']
            img_test = data['image']
        train_labels = [label_train, label_train]
        if len(train_labels[0].shape) == 1 or train_labels[0].shape[1] == 1:
            classes = np.unique(train_labels[0])
            k = int(train_labels[0].shape[0] * ratio / classes.shape[0])

        n_view = 2
        for v in range(n_view):
            if len(train_labels[0].shape) == 1 or train_labels[0].shape[1] == 1:
                train_seen_inx = []
                train_unseen_inx = []
                for i in range(len(classes)):
                    c = classes[i]
                    inx = (c == train_labels[v].reshape([-1])).nonzero()[0]
                    state = np.random.seed(1999)
                    np.random.shuffle(inx)
                    train_seen_inx.append(inx[0: k])
                    train_unseen_inx.append(inx[k::])
                train_seen_inx = np.concatenate(train_seen_inx)
                train_unseen_inx = np.concatenate(train_unseen_inx)
            else:
                train_seen_inx = np.arange(k)
                train_unseen_inx = np.arange(k, train_labels[v].shape[0])

        se_text_train = text_train[train_seen_inx]
        se_img_train = img_train[train_seen_inx]
        se_label_train = label_train[train_seen_inx]

        un_text_train = text_train[train_unseen_inx]
        un_image_train = img_train[train_unseen_inx]
        un_label_train = label_train[train_unseen_inx]

    elif data_name == 'pascal':
        path_train = 'data3/pascal/clip_train.pkl'
        path_test = 'data3/pascal/clip_test.pkl'
        MAP = -1

        with open(path_train, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            label_train = data['label']
            text_train = data['text']
            img_train = data['image']
            label_train-=np.min(label_train)

        with open(path_test, 'rb') as f_pkl:
            data = pickle.load(f_pkl)
            label_test = data['label']
            text_test = data['text']
            img_test = data['image']
            label_test-=np.min(label_test)

        train_labels = [label_train, label_train]
        if len(train_labels[0].shape) == 1 or train_labels[0].shape[1] == 1:
            classes = np.unique(train_labels[0])
            k = int(train_labels[0].shape[0] * ratio / classes.shape[0])

        n_view = 2
        for v in range(n_view):
            if len(train_labels[0].shape) == 1 or train_labels[0].shape[1] == 1:
                train_seen_inx = []
                train_unseen_inx = []
                for i in range(len(classes)):
                    c = classes[i]
                    inx = (c == train_labels[v].reshape([-1])).nonzero()[0]
                    state = np.random.seed(1999)
                    np.random.shuffle(inx)
                    train_seen_inx.append(inx[0: k])
                    train_unseen_inx.append(inx[k::])
                train_seen_inx = np.concatenate(train_seen_inx)
                train_unseen_inx = np.concatenate(train_unseen_inx)
            else:
                train_seen_inx = np.arange(k)
                train_unseen_inx = np.arange(k, train_labels[v].shape[0])

        se_text_train = text_train[train_seen_inx]
        se_img_train = img_train[train_seen_inx]
        se_label_train = label_train[train_seen_inx]

        un_text_train = text_train[train_unseen_inx]
        un_image_train = img_train[train_unseen_inx]
        un_label_train = label_train[train_unseen_inx]

    return se_img_train,se_text_train,se_label_train,un_image_train,un_text_train,un_label_train,img_test,text_test,label_test,MAP


def get_data(dataname, batch_size, semi_set, ratio):
    se_img_train,se_text_train,se_label_train,un_image_train,un_text_train,un_label_train,img_test,text_test,label_test,MAP = load_deep_features(dataname, semi_set, ratio)
    labeled_dataset = LabeledDataSet(se_img_train, se_text_train,se_label_train)
    unlabeled_dataset = UnLabeledDataSet(un_image_train, un_text_train,un_label_train)

    labeled_dataloader = DataLoader(labeled_dataset, batch_size=batch_size, drop_last=True,shuffle=True, num_workers=4)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, drop_last=True,shuffle=True, num_workers=4)

    test_dataset=LabeledDataSet(img_test,text_test,label_test)
    test_dataloader=DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return labeled_dataloader, unlabeled_dataloader, test_dataloader, MAP


if __name__ == '__main__':
    load_deep_features('wikipedia')
    #load_data('wikipedia')