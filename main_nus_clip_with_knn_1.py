import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import copy
import time
import numpy as np
import random
import sys
import os
import pickle
import matplotlib.pyplot as plt
from model import Model
from evaluate import fx_calc_map_label,fx_calc_map_multilabel_k
from metrics import PAN, Triplet_Loss, Contrastive_Loss, Label_Regression_Loss, Modality_invariant_Loss,cross_modal_contrastive_ctriterion,CrossModalCenterLoss,Supervised_CrossModal_Contrastive_Loss,Supervised_IntraModal_Contrastive_Loss
from torch.autograd import Function
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from extract_clip_feature import CustomDataSet
from tabulate import tabulate
import faiss
from sklearn.utils import shuffle
from idecutils import cluster_acc
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from metrics import Loss
from sklearn.cluster import KMeans
from loss_n import Contrastive
from make_mask import *

def setup_seed(seeds):
    import random as rn
    import os
    os.environ['PYTHONHASHSEED'] = str(seeds)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    rn.seed(seeds)
    np.random.seed(seeds)
    torch.manual_seed(seeds)
    torch.cuda.manual_seed(seeds)
    torch.cuda.manual_seed_all(seeds)
    from torch.backends import cudnn
    cudnn.enabled = False
    cudnn.benchmark = False  # True accelarating the training
    cudnn.deterministic = True

def evaluate1(model):
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
    model.eval()
    running_loss = 0.0
    t_imgs, t_txts, t_labels = [], [], []
    predict=[]
    with torch.no_grad():
        for  xs,label,idx in loader:
            for v in range(view):
                xs[v] = xs[v].to(device)
            idx=idx.to(device)
            z0, z1,_ = model(xs[0], xs[1])
            label_realvalue = label.float().to(device)
            z0=l2norm(z0,dim=1)
            z1=l2norm(z1,dim=1)

            from metrics import Loss
            criterion = Loss(data_size, args.temperature_f, device).to(device)
            loss=criterion.forward_feature(z0,z1)

            t_imgs.append(z0.cpu().numpy())
            t_txts.append(z1.cpu().numpy())
            t_labels.append(label_realvalue.cpu().numpy())
            running_loss+= loss.item()
    t_imgs = np.concatenate(t_imgs)  # for visualization
    t_txts = np.concatenate(t_txts)  # for visualization
    t_labels = np.concatenate(t_labels)
    i_map = fx_calc_map_label(t_imgs, t_txts, t_labels)
    t_map = fx_calc_map_label(t_txts, t_imgs, t_labels)

    running_loss=running_loss/len(loader)
    print('Eval loss: ',running_loss)
    print('Image to Text: MAP: {:.4f}'.format(i_map))
    print('Text to Image: MAP: {:.4f}'.format(t_map))
    return i_map, t_map, t_imgs, t_txts, t_labels,running_loss
1

def evaluate(model):
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    loader = torch.utils.data.DataLoader(dataset, batch_size=data_size, shuffle=False)
    model.eval()
    running_loss = 0.0
    t_imgs, t_txts, t_labels = [], [], []
    predict=[]
    with torch.no_grad():
        for  xs,label,idx in loader:
            for v in range(view):
                xs[v] = xs[v].to(device)
            idx=idx.to(device)
            z, q, z1, z2 = model(xs[0], xs[1])
            #p=target_distribution(q)
            #kl_loss = F.kl_div(q.log(), p)
            ypred=q.cpu().numpy().argmax(1)
            ypred=torch.LongTensor(ypred)
            label_realvalue = label.float().to(device)
            z1=l2norm(z1,dim=1)
            z2=l2norm(z2,dim=1)
            #from loss_n import Contrastive
            #contrastive=Contrastive()
            #contrastive_loss=contrastive(model.ae,ypred)
            #loss=kl_loss+args.beta*contrastive_loss

            t_imgs.append(z1.cpu().numpy())
            t_txts.append(z2.cpu().numpy())
            t_labels.append(label_realvalue.cpu().numpy())
            predict.append(ypred)
            #running_loss+= loss.item()
    t_imgs = np.concatenate(t_imgs)  # for visualization
    t_txts = np.concatenate(t_txts)  # for visualization
    t_labels = np.concatenate(t_labels)
    predict=np.concatenate(predict)
    #i_map = fx_calc_map_multilabel_k(retrieval_txts, retrieval_labs, query_imgs, query_labs, k=0, metric='cosine')
    #t_map=fx_calc_map_multilabel_k(retrieval_imgs, retrieval_labs, query_txts, query_labs, k=0, metric='cosine')
    i_map = fx_calc_map_label(t_imgs, t_txts, t_labels)
    t_map = fx_calc_map_label(t_txts, t_imgs, t_labels)
    #from evaluate1 import calc_metrics
    #cluster_metrics=calc_metrics(t_labels,predict)
    acc, f1 = cluster_acc(t_labels,predict)
    nmi = nmi_score(t_labels,predict)
    ari = ari_score(t_labels,predict)

    #running_loss=running_loss/len(loader)
    print('*************Eval*****************' )
    print('Image to Text: MAP: {:.4f}'.format(i_map))
    print('Text to Image: MAP: {:.4f}'.format(t_map))
    print('cluster acc : {:.4f}'.format(acc))
    print('cluster nmi : {:.4f}'.format(nmi))
    print('cluster ari : {:.4f}'.format(ari))


    return i_map, t_map, t_imgs, t_txts, t_labels

def make_imputation(mask, indices, incomplete_ind):
    global data_list
    K=3


    for v in range(2):
        for i in range(data_size):
            if mask[i, v] == 0:
                predicts = []
                for w in range(2):
                    # only the available views are selected as neighbors
                    if w != v and mask[i, w] != 0:
                        neigh_w = indices[w][i]
                        for n_w in range(neigh_w.shape[0]):
                            if mask[neigh_w[n_w], v] != 0 and mask[neigh_w[n_w], w] != 0:
                                predicts.append(data_list[v][neigh_w[n_w]])
                            if len(predicts) >= K:
                                break
                #print(len(predicts))
                assert len(predicts) >= K
                fill_sample = np.mean(predicts, axis=0)
                data_list[v][i] = fill_sample

    global full_loader
    full_dataset = FullDataset(2, data_list,y)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)


    #global incomplete_loader
    #incomplete_data = []
    #for v in range(2):
         #incomplete_data.append(data_list[v][incomplete_ind])
    #incomplete_label = y[incomplete_ind]
    #incomplete_dataset = MultiviewDataset(2, incomplete_data, incomplete_label)
    #incomplete_loader = DataLoader(incomplete_dataset, args.batch_size, drop_last=True)
    #return full_loader
    #return full_loader,incomplete_loader

class Memory:
    def __init__(self):
        self.features = None
        self.alpha = 0.5
        self.interval = 1
        self.bi = False

    def cal_cur_feature(self, model, loader):
        features = []
        for v in range(2):
            features.append([])

        for _, (xs, y, _) in enumerate(loader):
            for v in range(2):
                xs[v] = xs[v].to(device)
            with torch.no_grad():
                #_,_,hs1,hs2,_, _, _ = model(xs[0],xs[1])
                hs1,hs2, _ = model(xs[0],xs[1])
                hs=[hs1,hs2]
                for v in range(2):
                    fea = hs[v].detach().cpu().numpy()
                    features[v].extend(fea)

        for v in range(2):
            features[v] = np.array(features[v])

        return features

    def update_feature(self, model, loader, mask, incomplete_ind, epoch):
        topK = 500
        model.eval()
        cur_features = self.cal_cur_feature(model, loader)
        indices = []
        if epoch == 1:
            self.features = cur_features
            for v in range(2):
                fea = np.array(self.features[v])
                n, dim = fea.shape[0], fea.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(fea)
                _, ind = index.search(fea, topK + 1)  # Sample itself is included
                indices.append(ind[:, 1:])
            return indices
        elif epoch % self.interval == 0:
            for v in range(2):
                f_v = (1-self.alpha)*self.features[v] + self.alpha*cur_features[v]
                self.features[v] = f_v/np.linalg.norm(f_v, axis=1, keepdims=True)

                n, dim = self.features[v].shape[0], self.features[v].shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(self.features[v])
                _, ind = index.search(self.features[v], topK + 1)  # Sample itself is included
                indices.append(ind[:, 1:])
            if self.bi:
                make_imputation(mask, indices, incomplete_ind)
            return indices

def figure_plt(Train_Loss, Valid_Loss):
    plt.figure()
    Epoch = len(Train_Loss)
    X = range(1, Epoch + 1)
    #Valid_Loss=Valid_Loss.data.cpu().numpy()
    plt.plot(X, Train_Loss, label='Train loss')
    plt.plot(X, Valid_Loss, label='Valid loss')
    plt.legend()
    # plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('result800/{}_loss.png'.format(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
    # plt.show()
def figure_plt_map_epoch(mAP,dataset_name):
    #plt.figure()
    Epoch=len(mAP)
    X=range(1,Epoch+1)
    plt.plot(X,mAP,label='mAP ')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('0.1*mAP')

    plt.savefig('result800/{}_map.png'.format(dataset_name))

# def load_dataset(name, bsz):
#     #train_loc = 'data/'+name+'/train.pkl'
#     #test_loc 'data/'+name+'/test.pkl'
#     train_loc = 'data/'+name+'/train_bert_feature.pkl'
#     test_loc = 'data/'+name+'/test_bert_feature.pkl'
#     with open(train_loc, 'rb') as f_pkl:
#         data = pickle.load(f_pkl)
#         train_labels = data['label']
#         train_label=np.array([np.argmax(item) for item in train_labels])
#         train_texts = data['text_feature']
#         train_images = data['image_feature']
#         #train_ids = data['ids']
#     with open(test_loc, 'rb') as f_pkl:
#         data = pickle.load(f_pkl)
#         test_labels = data['label']
#         test_label = np.array([np.argmax(item) for item in test_labels])
#         test_texts =  data['text_feature']
#         test_images = data['image_feature']
#         #test_ids = data['ids']
#     imgs = {'train': train_images, 'test': test_images}
#     texts = {'train': train_texts,  'test': test_texts}
#     labs = {'train': train_label, 'test': test_label}
#     #ids = {'train': train_ids, 'test': test_ids}
#
#     dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labs=labs[x])
#                for x in ['test']}
#
#     shuffle = { 'test': False}
#
#     dataloader = {x: DataLoader(dataset[x], batch_size=bsz,
#                                 shuffle=shuffle[x], num_workers=0) for x in ['test']}
#
#     return dataloader
# def load_data(name):
#     """
#     :param name: name of dataset
#     :return:
#     data_list: python list containing all views, where each view is represented as numpy array
#     labels: ground_truth labels represented as numpy array
#     dims: python list containing dimension of each view
#     num_views: number of views
#     data_size: size of data
#     class_num: number of category
#     """
#     train_loc = 'data/'+name+'/train_bert_feature.pkl'
#     test_loc = 'data/'+name+'/test_bert_feature.pkl'
#     data_list = []
#     with open(train_loc, 'rb') as f_pkl:
#
#         data = pickle.load(f_pkl)
#         train_labels = data['label']
#         train_label=np.array([np.argmax(item) for item in train_labels])
#         train_texts = data['text_feature']
#         train_images = data['image_feature']
#         data_list.append(train_images)
#         data_list.append(train_texts.astype(np.float64))
#
#     with open(test_loc, 'rb') as f_pkl:
#         data = pickle.load(f_pkl)
#         test_labels = data['label']
#         test_label = np.array([np.argmax(item) for item in test_labels])
#         test_texts =  data['text_feature']
#         test_images = (data['image_feature']).astype(np.float64)
#         #test_ids = data['ids']
#
#     imgs = {'test': test_images}
#     texts = {'test': test_texts}
#     labs = { 'test': test_label}
#     dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labs=labs[x])
#                for x in [ 'test']}
#     shuffle = {'test': False}
#     text_dataloader = {x: DataLoader(dataset[x], batch_size=100,shuffle=shuffle[x], num_workers=0) for x in ['test']}
#
#     return data_list,train_label,text_dataloader
# class UnLabeledDataSet(Dataset):
#     def __init__(self, images, texts,lab):
#         self.images = images
#         self.texts = texts
#         self.labels=lab
#         self.labels=self.labels.squeeze()
#
#     def __getitem__(self, index):
#         img = self.images[index]
#         text = self.texts[index]
#         # print('---------------------------')
#         # print(index)
#         lab=self.labels[index]
#         return img,text,lab
#
#     def __len__(self):
#         count = len(self.images)
#         return count
#
# def load_dataset(data_name, bsz):
#     from scipy.io import loadmat
#     if data_name == 'wikipedia':
#         path = './datasets/wikipedia/'
#         MAP = -1
#         img_train = loadmat(path + "train_img.mat")['train_img']
#         img_test = loadmat(path + "test_img.mat")['test_img']
#         text_train = loadmat(path + "train_txt.mat")['train_txt']
#         text_test = loadmat(path + "test_txt.mat")['test_txt']
#         label_train = loadmat(path + "train_img_lab.mat")['train_img_lab']
#         label_test = loadmat(path + "test_img_lab.mat")['test_img_lab']
#
#         loc=231
#
#         #train_data = [img_train, text_train]
#         #train_sup_data = [img_train[:sup_size], text_train[:sup_size]]
#         #valid_data = [img_test[:231], text_test[:231]]
#         #test_data = [img_test[231:], text_test[231:]]
#
#         #train_labels = [label_train, label_train]
#         #train_sup_labels = [label_train[:sup_size], label_train[:sup_size]]
#         #valid_labels = [label_test[:231], label_test[:231]]
#         #test_labels = [label_test[231:], label_test[231:]]
#
#     elif data_name == 'nus_wide':
#         path = './datasets/nus_wide/'
#         MAP = -1
#         img_train = loadmat(path + "train_img.mat")['train_img']
#         img_test = loadmat(path + "test_img.mat")['test_img']
#         text_train = loadmat(path + "train_txt.mat")['train_txt']
#         text_test = loadmat(path + "test_txt.mat")['test_txt']
#         label_train = loadmat(path + "train_img_lab.mat")['train_img_lab']
#         label_test = loadmat(path + "test_img_lab.mat")['test_img_lab']
#
#         # label_train = ind2vec(label_train).astype(int)
#         # label_test = ind2vec(label_test).astype(int)
#         loc=1000
#         # size = img_train.shape[0]
#         # sup_size = int(size * sup_rate)
#         # train_data = [img_train, text_train]
#         # train_sup_data = [img_train[:sup_size], text_train[:sup_size]]
#         #
#         # valid_data = [img_test[:1000], text_test[:1000]]
#         # test_data = [img_test[1000:], text_test[1000:]]
#         #
#         # train_labels = [label_train, label_train]
#         # train_sup_labels = [label_train[:sup_size], label_train[:sup_size]]
#         # valid_labels = [label_test[:1000], label_test[:1000]]
#         # test_labels = [label_test[1000:], label_test[1000:]]
#     elif data_name == 'mscoco':
#         path='./datasets/MSCOCO/'
#         img_train = loadmat(path + "train_img.mat")['train_img']
#         img_test = loadmat(path + "test_img.mat")['test_img']
#         text_train = loadmat(path + "train_txt.mat")['train_txt']
#         text_test = loadmat(path + "test_txt.mat")['test_txt']
#         label_train = loadmat(path + "train_lab.mat")['train_lab']
#         label_test = loadmat(path + "test_lab.mat")['test_lab']
#         #loc = 1000
#
#     elif data_name == 'pascal':
#         path='./datasets/pascal/'
#         train_image = loadmat(path + "images_pascal-sentences_vgg19_4096d.mat")
#         labels = loadmat(path + "labels_pascal-sentences.mat")
#         train_text = loadmat(path + "texts_pascal-sentences_doc2vec_300.mat")
#         img_train=train_image['images'][:800]
#         img_test = train_image['images'][800:]
#         text_train=train_text['texts'][:800]
#         text_test = train_text['texts'][800:]
#         label_train=labels['labels'][:,:800]
#         label_test=labels['labels'][:,800:]
#         # img_train = loadmat(path + "train_img.mat")['train_img']
#         # img_test = loadmat(path + "test_img.mat")['test_img']
#         # text_train = loadmat(path + "train_txt.mat")['train_txt']
#         # text_test = loadmat(path + "test_txt.mat")['test_txt']
#         # label_train = loadmat(path + "train_lab.mat")['train_lab']
#         # label_test = loadmat(path + "test_lab.mat")['test_lab']
#         #loc = 1000
#     imgs = {'train': img_train, 'test': img_test}
#     texts = {'train': text_train,  'test': text_test}
#     labs = {'train': label_train, 'test': label_test}
#     #unlabeled_dataset = UnLabeledDataSet(train_data[0], train_data[1])
#     #unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#
#     dataset = {x: UnLabeledDataSet(images=imgs[x], texts=texts[x],lab=labs[x]) for x in ['train', 'test']}
#     shuffle = {'train': True, 'test': False}
#
#     dataloader = {x: DataLoader(dataset[x], batch_size=bsz,
#                                 shuffle=shuffle[x], num_workers=0) for x in ['train', 'test']}
#
#     return dataloader



# from dataloader import load_data
# dataset, dims, view, data_size, class_num = load_data(args.dataset)
# data_loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True,drop_last=True,)

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

class CCMR(nn.Module):
    def __init__(self,img_dim,text_dim,n_clusters,n_h,alpha,pretrain_path='result300/pascal/_128_49_UCMR.pt'):
        super(CCMR, self).__init__()
        self.pretrain_path=pretrain_path
        self.alpha =alpha
        self.ae=Model(img_dim,text_dim)
        self.cluster_layer=nn.Parameter(torch.Tensor(n_clusters,n_h))
        torch.nn.init.xavier_normal(self.cluster_layer)

    def pretrain_MLP(self,path=''):
        if path=='':
           pretrain(self.ae)
        else:
           #load pretrain weights
           self.ae.load_state_dict(torch.load(path))
        print('load pretrained ae from',path)

    def forward(self,image,text):

        z0,z1,z = self.ae(image,text)
        #cluster
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return z, q, z0, z1

def pretrain(model):

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_schedu = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10,threshold=0.0001, threshold_mode='rel', cooldown=0,min_lr=0, eps=1e-08, verbose=False)
    criterion = Loss(args.batch_size, args.temperature_f, device).to(device)
    data_loader = torch.utils.data.DataLoader(com_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True )
    criterion1 = lambda x, y: (((x - y) ** 2).sum(1).sqrt()).mean().to(device)

    i_map = []
    t_map = []
    best_map = 0.0
    best_map_list = []
    besti2_t = 0.0
    bestt2_i = 0.0
    no_up = 0
    early_stop = 20
    for epoch in range(150):
        tot_loss = 0.0
        for batch_idx,(xs,_,_) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            z0,z1,_= model(xs[0],xs[1])
            z0=l2norm(z0,dim=1)
            z1=l2norm(z1,dim=1)
            c_loss=criterion.forward_feature(z0,z1)
            mse_loss=criterion1(z0,z1)
            #loss=args.alpha*c_loss+mse_loss
            loss=c_loss
            loss.backward()
            optimizer.step()
            tot_loss+=loss.item()
        print('Pretrain_epoch {}'.format(epoch), 'loss:{:.6f}'.format(tot_loss / len(data_loader)))

        print('...............evaluate...................')
        img2text, text2img, t_imgs, t_txts, t_labels,_=evaluate1(model)
        i_map.append(img2text)
        t_map.append(text2img)

        if (img2text + text2img) / 2. > best_map:
            best_map = (img2text + text2img) / 2.
            besti2_t = img2text
            bestt2_i = text2img
            print('New Best model')
            best_map_list.append(best_map)
            best_model_wts = copy.deepcopy(model.state_dict())
            dir = os.getcwd()
            dirs_path = os.path.join(dir,'result300/{}'.format(args.dataset))
            if not os.path.exists(dirs_path):
                os.makedirs(dirs_path)
            torch.save(model.state_dict(),'result300/{}/_{}_{}_UCMR.pt'.format(args.dataset,args.batch_size, epoch))
            np.savez('result300/{}/_{}_{}_{}.npz'.format(args.dataset,args.batch_size, epoch, best_map),image=t_imgs, text=t_txts, label=t_labels)
            no_up = 0
        else:
            no_up += 1
        lr_schedu.step(best_map)
        if no_up >= early_stop:
            break
    print(f'pretrain_Best average MAP: {best_map:.4f}, Epoch: {epoch + 1 - early_stop}')
    print(f'pretrain_Best I2T MAP: {besti2_t:.4f}, Epoch: {epoch + 1 - early_stop}')
    print(f'pretrain_Best T2I MAP: {bestt2_i:.4f}, Epoch: {epoch + 1 - early_stop}')




def cluster_contrastive(img_dim,text_dim):

    #data_list = copy.deepcopy(record_data_list)
    model=CCMR(img_dim,text_dim,n_clusters=args.n_clusters,n_h=args.n_h,alpha=1.0,pretrain_path='result300/pascal/_128_49_UCMR.pt').to(device)
    print(model)
    #model.pretrain_MLP()

    model.pretrain_MLP('result300/nus_wide/_128_32_UCMR.pt')

    memory = Memory()
    memory.interval = 1



    #full_dataloader,incomplete_data_loader=make_imputation(mask, indices, incomplete_ind)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    #data1 = data_list[0]
    #data2 = data_list[1]
    data1=mv_data[0]
    data2 = mv_data[1]
    y=mv_label
    data1 = torch.Tensor(data1).to(device)
    data2 = torch.Tensor(data2).to(device)
    model.eval()
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    z0, z1, z = model.ae(data1, data2)
    z = z.cpu().detach().numpy()
    h = scaler.fit_transform(z)
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=20)
    y_pred = kmeans.fit_predict(h)
    del h
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    acc, f1 = cluster_acc(y, y_pred)
    nmi = nmi_score(y, y_pred)
    ari = ari_score(y, y_pred)
    print('Kmeans performance')
    print(':Acc {:.4f}'.format(acc), 'nmi {:.4f}'.format(nmi), 'ari {:.4f}'.format(ari))
    y_pred_last = y_pred
    #indices = memory.update_feature(model.ae, full_loader, mask, incomplete_ind, epoch=1)
    #make_imputation(mask, indices, incomplete_ind)
    model.train()
    i_map = []
    t_map = []
    best_map = 0.0
    best_map_list = []
    besti2_t = 0.0
    bestt2_i = 0.0
    no_up = 0
    early_stop = 11

    best_acc2 = 0
    best_nmi2 = 0
    best_ari2 = 0
    best_f12 = 0
    best_epoch = 0
    total_loss = 0
    memory.bi = True
    iteration = 0
    for epoch in range(150):
        iteration += 1
        data1 = data_list[0]
        data2 = data_list[1]
        #data1 = mv_data[0]
        #data2 = mv_data[1]
        data1 = torch.Tensor(data1).to(device)
        data2 = torch.Tensor(data2).to(device)
        if epoch % args.update_interval == 0:
            _,tmp_q,_,_  = model(data1, data2)
            p = target_distribution(tmp_q)
            y_pred = tmp_q.cpu().detach().numpy().argmax(1)
            acc,f1 = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)
        if acc>best_acc2:
            best_acc2 = np.copy(acc)
            best_nmi2 = np.copy(nmi)
            best_ari2 = np.copy(ari)
            best_f12 = np.copy(f1)
            best_epoch = epoch
        print('best epoch {}'.format(epoch), ':best_acc {:.4f}'.format(acc),',best_nmi {:.4f}'.format(nmi), ',best_ari {:.4f}'.format(ari),':Acc {:.4f}'.format(acc),
                  ',nmi {:.4f}'.format(nmi),',ari {:.4f}'.format(ari))

        delta_y = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if epoch > 200 and delta_y < args.tol:
            print('Training stopped: epoch=%d, delta_label=%.4f, tol=%.4f' % (epoch, delta_y, args.tol))
            break

        z, q, z1, z2 = model(data1, data2)
        kl_loss = F.kl_div(q.log(), p)
        tot_loss=0.0
        tot_con=0.0
        tot_kl=0.0

        yred = torch.LongTensor(y_pred)
        cat_feature=torch.cat((z1,z2),dim=0)

        contrastive = Contrastive()
        contrastive_loss = contrastive(model.ae,cat_feature, yred)
        loss = kl_loss + args.beta * contrastive_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_con += contrastive_loss.item()
        tot_kl += kl_loss.item()
        tot_loss += loss.item()
        print('epoch {}'.format(epoch),'contrastive Loss:{:.6f}'.format(tot_con),'kl Loss:{:.6f}'.format(tot_kl), 'Loss:{:.6f}'.format(tot_loss))

        #memory.interval = 1
        #indices = memory.update_feature(model.ae,full_loader, mask, incomplete_ind, iteration)

        img2text, text2img, t_imgs, t_txts, t_labels=evaluate(model)

        i_map.append(img2text)
        t_map.append(text2img)

        if (img2text + text2img) / 2. > best_map:
            best_map = (img2text + text2img) / 2.
            besti2_t = img2text
            bestt2_i = text2img
            print('New Best model')
            best_map_list.append(best_map)
            no_up = 0
            best_model_wts = copy.deepcopy(model.state_dict())
            dir = os.getcwd()
            dirs_path = os.path.join(dir,'result600/{}/{}_{}_{}'.format(args.dataset, args.dataset,args.batch_size,args.beta))
            if not os.path.exists(dirs_path):
                os.makedirs(dirs_path)
            torch.save(model.state_dict(),'result600/{}/{}_{}_{}/{}_UCMR.pt'.format(args.dataset, args.dataset,args.batch_size,args.beta, epoch))
            np.savez('result600/{}/{}_{}_{}/{}_{}.npz'.format(args.dataset, args.dataset,args.batch_size,args.beta, epoch, best_map),image=t_imgs, text=t_txts, label=t_labels)
            no_up = 0
        else:
            # lr_schedu.step(best_map)
            no_up += 1
        if no_up >= early_stop:
            break
    print(f'Best average MAP: {best_map:.4f}, Epoch: {epoch + 1 - early_stop}')
    print(f'Best I2T MAP: {besti2_t:.4f}, Epoch: {epoch + 1 - early_stop}')
    print(f'Best T2I MAP: {bestt2_i:.4f}, Epoch: {epoch + 1 - early_stop}')


if __name__ == '__main__':

    import argparse

    Dataname = 'nus_wide' #wikipedia,nus_wide,pascal,xmedianet
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default=Dataname)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument("--temperature_f", default=0.5)
    parser.add_argument("--beta", default=1)
    parser.add_argument("--alpha", default=10)
    parser.add_argument("--n_clusters", default=10)
    parser.add_argument("--n_h", default=128)
    parser.add_argument("--learning_rate", default=0.0001)
    parser.add_argument("--lr", default=0.0001)
    parser.add_argument('--tol', default=1e-7, type=float)
    parser.add_argument('--load', default=True)
    parser.add_argument("--update_interval", default=1)
    parser.add_argument('--miss_rate', type=int, default=1, help='rate of miss')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "pascal":
        args.pretrain_epochs = 50
        seed = 1
    if args.dataset == "wikipedia":
        args.pretrain_epochs = 50
        seed = 1
    if args.dataset == "nus_wide":
        args.pretrain_epochs = 50
        seed = 10
    if args.dataset == "xmedianet":
        args.pretrain_epochs = 50
        seed = 1

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(seed)

    from dataloader import load_data,MultiviewDataset,RandomSampler,FullDataset
    dataset, dims, view, data_size, class_num = load_data(args.dataset,args.load)

    data1 = dataset.x1
    data2 = dataset.x2
    y = dataset.y
    datasize = data1.shape[0]

    mask = get_mask(2, datasize, args.miss_rate)
    sum_vec = np.sum(mask, axis=1, keepdims=True)
    complete_index = (sum_vec[:, 0]) == 2

    mv_data = []
    mv_data.append(data1[complete_index])
    mv_data.append(data2[complete_index])
    mv_label = y[complete_index]
    data_list=[]
    data_list.append(data1)
    data_list.append(data2)
    #record_data_list = copy.deepcopy(data_list)

    com_dataset = MultiviewDataset(2, mv_data, mv_label)
    full_dataset = MultiviewDataset(2, data_list, y)

    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)
    incomplete_ind = (sum_vec[:, 0]) != 2

    cluster_contrastive(dims[0],dims[1])

