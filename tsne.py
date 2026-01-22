from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
import pickle
import seaborn as sns

def load_dataset():
    '''
    train_loc = 'data/'+name+'/clip_train.pkl'
    test_loc = 'data/'+name+'/clip_test.pkl'

    with open(test_loc, 'rb') as f_pkl:
    '''
    f=open('/home/yh/code/PAN/data3/nus-wide/clip_test.pkl','rb')

    data = pickle.load(f)
    test_labels = data['label']
    test_texts = data['text']
    test_images = data['image']
    test_ids = data['ids']
    return test_images,test_texts,test_labels

def load_result():
    tsne=np.load('/home/yh/code/PAN/result600/nus_wide/nus_wide_128_0.01/49_0.555250050746708.npz')
    tsne_img_feature=tsne['image']
    tsne_txt_feature = tsne['text']
    tsne_labels = tsne['label']

    return tsne_img_feature, tsne_txt_feature,tsne_labels


def plot_embedding(data,data_t, label):
    # x_min, x_max = np.min(data, 0), np.max(data, 0)
    # data = (data - x_min) / (x_max - x_min)
    #
    # x_min, x_max = np.min(data_t, 0), np.max(data_t, 0)
    # data_t = (data_t - x_min) / (x_max - x_min)
    embeddings = np.vstack([data, data_t])
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    vectors=tsne.fit_transform(embeddings)

    fig = plt.figure()
    ax = plt.subplot(111)
    #data['labels']=
    #data_t['labels']='text'

    #plt.scatter(data[:,0],data[:,1])
    plt.scatter(vectors[:1000, 0], vectors[:1000, 1],s=15, color=plt.cm.Set1(label/9),marker='+')
    plt.scatter(vectors[1000:, 0], vectors[1000:, 1], s=15,color=plt.cm.Set1(label/9),marker='*')
    #plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    #plt.colorbar()
    plt.savefig('result/1tsne.pdf')
    plt.show()

    plt.scatter(vectors[:1000, 0], vectors[:1000, 1], s=15, color=plt.cm.Set1(label /9), marker='+')
    #plt.axis('off')
    #plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.savefig('result/2tsne.pdf')
    plt.show()

    plt.scatter(vectors[1000:, 0], vectors[1000:, 1], s=15,color=plt.cm.Set1(label/9),marker='*')
    #plt.colorbar()
    #plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('result/3tsne.pdf')
    plt.show()

    #plt.scatter(data_t[:, 0], data_t[:, 1], c=label, cmap='tab10',marker='*')
    #sns.scatterplot(data[:,0],data[:,1],hue=label,c=label,marker='+',ax=ax)
    #sns.scatterplot(data_t[:, 0], data_t[:, 1],hue=label, c=label,marker='*',ax=ax)
    #x=list(range(0,200))
    #for i in range(data.shape[0]):
        #plt.scatter(data[i:,0],data[i:,1],marker='+')
        #plt.scatter(data_t[i:, 0], data_t[i:, 1], marker='*')
        #plt.text(data[i, 0], data[i, 1], str(label[i]),fontdict={'weight': 'bold', 'size': 9})
        #plt.text(data[i, 0], data[i, 1], str(label[i]),color=plt.cm.Set1(label[i] / 200.),fontdict={'weight': 'bold', 'size': 5})
        #plt.text(data_t[i, 0], data_t[i, 1], str(label[i]), color=plt.cm.Set1(label[i] / 200.),fontdict={'weight': 'bold', 'size': 5})


    #plt.title(title)


    #plt.savefig('result/tsne.png')
    #plt.savefig('./result/tsne.pdf')
    #plt.show()
    return fig

def main():
    name='nus_wide' #wiki-10, nus-wide-10, pascal-20, xmedianet-200
    img_feature,txt_feature,labels = load_dataset()
    tsne_img_feature, tsne_txt_feature,tsne_labels=load_result()

    #result_img = tsne.fit_transform(img_feature)
    #result_txt = tsne.fit_transform(txt_feature)
    #result_img=tsne.fit_transform(tsne_img_feature)
    #result_txt=tsne.fit_transform(tsne_txt_feature)
    fig = plot_embedding(tsne_img_feature,tsne_txt_feature,tsne_labels)
    plt.savefig('result/tsne.pdf')



if __name__=='__main__':
    main()


