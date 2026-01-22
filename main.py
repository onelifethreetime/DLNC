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

def make_imputation(mask, indices, incomplete_ind):
    global data_list
    K=5


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

                assert len(predicts) >= K
                fill_sample = np.mean(predicts, axis=0)
                data_list[v][i] = fill_sample

    global full_loader
    full_dataset = FullDataset(2, data_list,Y)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False)


    global incomplete_loader
    incomplete_data = []
    for v in range(2):
         incomplete_data.append(data_list[v][incomplete_ind])
    incomplete_label = Y[incomplete_ind]
    incomplete_dataset = MultiviewDataset(2, incomplete_data, incomplete_label)
    incomplete_loader = DataLoader(incomplete_dataset, args.batch_size, drop_last=True)

    return full_loader,incomplete_loader

class Memory:
    def __init__(self):
        self.features = None
        self.alpha = 0.5
        self.interval = 4
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
        topK = 100
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



def train(Net, com_dataset, full_loader,optimizer , mask, incomplete_ind):

    loader = DataLoader(com_dataset, batch_size=128, shuffle=True, drop_last=True)
    mse_loader = DataLoader(com_dataset, batch_size=128, shuffle=True)

    Net.train()


    for epoch in range(1,args.epochs):

        running_loss = 0.0

        for xs, _, _ in loader:
            img=xs[0].to(device)
            txt = xs[1].to(device)
            img_c, text_c, output_cluster = Net(img,txt)
            #img_c, text_c, view1_feature, view2_feature, view1_feature_rec, view2_feature_rec, output_cluster = Net(img,txt)

            loss=Net.loss(Net)
            #recon1 = F.mse_loss(view1_feature_rec,img)
            #recon2 = F.mse_loss(view2_feature_rec, txt)
            #reconstruction_loss = recon1 + recon2
            # if epoch >= args.incomplete_epoch:
            #     img_missing_idx_eval = mask[:, 0] == 0
            #     txt_co=x2_train[img_missing_idx_eval]
            #     txt_co_=Net.ecoders[1](txt_co)
            #     img_co_input = Net.decoders[0](txt_co_)
            #     txt_co_com=Net.encoders[0](img_co_input)
            #
            #     txt_missing_idx_eval = mask[:, 1] == 0
            #     img_co=x1_train[txt_missing_idx_eval]
            #     img_co_=Net.ecoders[0](img_co)
            #     txt_co_input = Net.decoders[1](img_co_)
            #     img_co_com=Net.encoders[1](txt_co_input)
            #
            #     img2text_loss=F.mse_loss(txt_co_, txt_co_com)
            #     text2img_loss=F.mse_loss(img_co_, img_co_com)
            #     loss3=img2text_loss+text2img_loss
            #
            #     loss = loss_c_c + reconstruction_loss +loss3
            #if epoch >= args.incomplete_epoch and epoch%args.t==0:
                #loss = loss_c_c + reconstruction_loss
                #pass
            #loss = loss_c_c +args.lambda1* reconstruction_loss

            optimizer.zero_grad()
            loss['tot'].backward()
            optimizer.step()

            running_loss += loss['tot'].item()


        print('log--------------------------------------------')
        params_dict = {}
        weights = Net.fusion.get_weights(softmax=True)
        weights=weights.cpu().detach().numpy()
        for i, w in enumerate(weights):
            params_dict[f"fusion/weight_{i}"] = w
            #params_dict[f"epoch_{epoch}"] = epoch
        headers = ["Key", "Value"]
        values = list(params_dict.items())
        print(tabulate(values, headers=headers), "\n")

    memory = Memory()
    memory.interval = 1

    indices = memory.update_feature(Net, full_loader, mask, incomplete_ind, epoch=1)
    full_dataloader,incomplete_data_loader=make_imputation(mask, indices, incomplete_ind)


    for epoch in range(args.epochs):

        running_loss = 0.0
        memory.bi = True
        iteration = 0
        for xs, _, _ in full_dataloader:
            iteration += 1
            img=xs[0].to(device)
            txt = xs[1].to(device)
            #img_c, text_c, view1_feature, view2_feature, view1_feature_rec, view2_feature_rec, output_cluster = Net(img,txt)
            img_c, text_c, output_cluster = Net(img,txt)

            loss=Net.loss(Net)

            #recon1 = F.mse_loss(view1_feature_rec,img)
            #recon2 = F.mse_loss(view2_feature_rec, txt)
            #reconstruction_loss = recon1 + recon2

            #loss = loss_c_c +args.lambda1* reconstruction_loss

            optimizer.zero_grad()
            loss['tot'].backward()
            optimizer.step()

            memory.interval = 1
            indices = memory.update_feature(Net, full_loader, mask, incomplete_ind, iteration)
            #full_dataloader,incomplete_data_loader=make_imputation(mask, indices, incomplete_ind)

            running_loss += loss['tot'].item()


def evaluate(Net, loader):
    Net.eval()
    running_loss = 0.0
    t_imgs, t_txts, t_labels = [], [], []
    predict=[]
    with torch.no_grad():
        for  img, text,label in loader:
            text = text.float().to(device)
            img = img.float().to(device)
            label_realvalue = label.float().to(device)
            img_features, text_feature, C = Net(img, text)

            loss = Net.loss(Net)
            eval_loss = loss['tot']

            t_imgs.append(img_features.cpu().numpy())
            t_txts.append(text_feature.cpu().numpy())
            t_labels.append(label_realvalue.cpu().numpy())
            predict.append(C.cpu().numpy().argmax(axis=1))
            eval_loss=eval_loss.item()
            running_loss+=eval_loss
    t_imgs = np.concatenate(t_imgs)  # for visualization
    t_txts = np.concatenate(t_txts)  # for visualization
    t_labels = np.concatenate(t_labels)
    predict=np.concatenate(predict)
    #i_map = fx_calc_map_multilabel_k(retrieval_txts, retrieval_labs, query_imgs, query_labs, k=0, metric='cosine')
    #t_map=fx_calc_map_multilabel_k(retrieval_imgs, retrieval_labs, query_txts, query_labs, k=0, metric='cosine')
    i_map = fx_calc_map_label(t_imgs, t_txts, t_labels)
    t_map = fx_calc_map_label(t_txts, t_imgs, t_labels)
    from evaluate1 import calc_metrics
    cluster_metrics=calc_metrics(t_labels,predict)

    running_loss=running_loss/len(loader)
    print('Eval loss: ',running_loss)
    print('Image to Text: MAP: {:.4f}'.format(i_map))
    print('Text to Image: MAP: {:.4f}'.format(t_map))
    print('cluster acc : {:.4f}'.format(cluster_metrics['acc']))
    print('cluster nmi : {:.4f}'.format(cluster_metrics['nmi']))
    return i_map, t_map, t_imgs, t_txts, t_labels,running_loss


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

if __name__ == '__main__':

    import argparse

    Dataname = 'pascal' #wikipedia,nus_wide,pascal,xmedianet
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--dataset', default=Dataname)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument("--temperature_f", default=0.5)
    parser.add_argument("--pretrain_epochs", default=50)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "pascal":
        args.con_epochs = 50
        seed = 1
    if args.dataset == "wikipedia":
        args.con_epochs = 50
        seed = 1
    if args.dataset == "nus_wide":
        args.con_epochs = 50
        seed = 1
    if args.dataset == "xmedianet":
        args.con_epochs = 50
        seed = 1

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    setup_seed(seed)

    from dataloader import load_data
    dataset, dims, view, data_size, class_num = load_data(args.dataset)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=args.batch_size,shuffle=True,drop_last=True,)

    accs = []
    nmis = []
    purs = []


    def target_distribution(q):
        weight = q ** 2 / q.sum(0)
        return (weight.t() / weight.sum(1)).t()

    class CCMR(nn.Module):
        def __init__(self):
            super(CCMR, self).__init__(img_dim,text_dim,n_clusters,n_h)
            self.ae=Model(img_input_dim=img_dim,text_input_dim=text_dim)
            #n_clusters=10,n_h=1024,
            self.cluster_layer=nn.Parameter(torch.Tensor(n_clusters,n_h))
            torch.nn.init.xavier_normal(self.cluster_layer)
        #def pre_train(self):
            #pretrain(self.ae)

        def forward(self,image,text):
            z0,z1,z = self.ae(image,text)
            q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
            q = q.pow((self.v + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
            return z, q, z0, z1


    model=Model(img_input_dim=img_dim,text_input_dim=text_dim).to(device)
    def pretrain(epoch):

        # for m in model.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         nn.init.constant_(m.bias, 0.0)
        args.lr_train=0.0001
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        from metrics import Loss
        criterion = Loss(args.batch_size, args.temperature_f, device).to(device)
        tot_loss=0.0
        for batch_idx,(xs,_,) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            z0,z1,_= model(xs[0],xs[1])
            loss=criterion.forward_feature(z0,z1)
            loss.backward()
            optimizer.step()
            tot_loss+=loss.item()
        print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))

    nh=128
    model_CCMR=CCMR(dims[0],dims[1],class_num,nh)
    epoch = 1
    while epoch <= args.pretrain_epochs:
        pretrain(epoch)
        epoch += 1


    def cluster_contrastive(epoch):

        loader = torch.utils.data.DataLoader(dataset,batch_size=data_size,shuffle=False,)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
        model.eval()
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        for step, (xs, y, _) in enumerate(loader):
            for v in range(view):
                xs[v] = xs[v].to(device)
            with torch.no_grad():
                z,z0,z1 = model(xs[0],xs[1])
                z=z.cpu().detach().numpy()
                h = scaler.fit_transform(z)

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=20)
        y_pred = kmeans.fit_predict(h)
        model_CCMR.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
        del h
        model_CCMR.train()

    #acc, f1 = cluster_acc(label, y_pred)
    #nmi = nmi_score(label, y_pred)
    #ari = ari_score(label, y_pred)
        if epoch % args.update_interval == 0:
            _,_, tmp_q = model_CCMR()
            # update target distribution p
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)
            y_pred = tmp_q.cpu().numpy().argmax(1)
            acc = cluster_acc(np.array(labels.cpu()), y_pred)
            nmi = nmi_score(np.array(labels.cpu()), y_pred)
            ari = ari_score(np.array(labels.cpu()), y_pred)
            if acc > acc_clu:
                acc_clu = acc
                nmi_clu = nmi
                ari_clu = ari
                torch.save(model_UCGL.state_dict(), args.model_path)
            for  batch_idx, (x, _, idx) in enumerate(loader):
                x=x.to(device)
                idx=idx.to(device)
                Z0,Z1,q =modal(x)

                kl_loss = F.kl_div(q.log(), p[idx])
                y=torch.from_numpy(y_pred).type(torch.long)
                from loss_n import Contrastive
                contrastive=Contrastive()
                contrastive_loss=contrastive(modal,y)
                loss=kl_loss+contrastive_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



