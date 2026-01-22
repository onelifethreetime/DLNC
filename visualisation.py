import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.io import loadmat
import pickle

def load_dataset():
    '''
    train_loc = 'data/'+name+'/clip_train.pkl'
    test_loc = 'data/'+name+'/clip_test.pkl'

    with open(test_loc, 'rb') as f_pkl:
    '''
    f=open('/home/gtt/CODE_WJS/code_SDCML/data3/nus-wide/clip_test.pkl','rb')

    data = pickle.load(f)
    test_labels = data['label']
    test_texts = data['text']
    test_images = data['image']
    test_ids = data['ids']
    return test_images,test_texts,test_labels


img_feature,txt_feature,labels = load_dataset()
tsne = TSNE(n_components=2, init='pca', random_state=0)
result_img = tsne.fit_transform(img_feature)
result_txt = tsne.fit_transform(txt_feature)

#orange Image samples in the common subspace
color = ['black', 'red', 'gold', 'orange', 'yellow', 'green', 'blue', 'violet', 'pink', 'c']
plt.figure(figsize=(8,8))
for i in range(0, img_feature.shape[0]):
    plt.scatter(result_img[i, 0], result_img[i, 1], edgecolor=color[int(labels[i])], c='none',
    s=30, marker='^', linewidth=1)
plt.savefig('result/OrangImagetsne3.pdf')
plt.show()

#orange text samples in the common subspace
color = ['black', 'red', 'gold', 'orange', 'yellow', 'green', 'blue', 'violet', 'pink', 'c']
plt.figure(figsize=(8,8))
for i in range(0, img_feature.shape[0]):
    plt.scatter(result_txt[i, 0], result_txt[i, 1], edgecolor=color[int(labels[i])], c='none',
    s=30, marker='o', linewidth=1)
plt.savefig('result/OrangTexttsne3.pdf')
plt.show()

image_co = np.loadtxt('./feature/nus_wide3/common_image_all.tsv')
text_co = np.loadtxt('./feature/nus_wide3/common_text_all.tsv')
proxy_co = np.loadtxt('./feature/nus_wide3/proxy.tsv')
all_co = np.concatenate((image_co, text_co, proxy_co))
label_co = np.loadtxt('./feature/nus_wide3/label_all.tsv')
lable_cos = np.concatenate((label_co, label_co))

ts = TSNE(n_components=2, init='pca', random_state=0, metric='cosine', early_exaggeration=50)
t_feat_all = ts.fit_transform(all_co)

#Image samples in the common subspace
color = ['black', 'red', 'gold', 'orange', 'yellow', 'green', 'blue', 'violet', 'pink', 'c']

plt.figure(figsize=(8,8))
for i in range(0, image_co.shape[0]):
    plt.scatter(t_feat_all[i, 0], t_feat_all[i, 1], edgecolor=color[int(lable_cos[i])], c='none',
    s=30, marker='^', linewidth=1)
plt.savefig('result/Imagetsne3.pdf')
plt.show()

#Text samples in the common subspace
color = ['black', 'red', 'gold', 'orange', 'yellow', 'green', 'blue', 'violet', 'pink', 'c']
plt.figure(figsize=(8,8))

for i in range(image_co.shape[0], image_co.shape[0] +text_co.shape[0]):
    plt.scatter(t_feat_all[i, 0], t_feat_all[i, 1], edgecolor=color[int(lable_cos[i])], c='none',
    s=30, marker='o', linewidth=1)

plt.savefig('result/Texttsne3.pdf')
plt.show()

#Image and text samples in the common subspace

color = ['black', 'red', 'gold', 'orange', 'yellow', 'green', 'blue', 'violet', 'pink', 'c']
plt.figure(figsize=(8,8))

for i in range(0, image_co.shape[0]):
    plt.scatter(t_feat_all[i, 0], t_feat_all[i, 1], edgecolor=color[int(lable_cos[i])], c='none',
    s=30, marker='^', linewidth=1)
for i in range(image_co.shape[0], image_co.shape[0] +text_co.shape[0]):
    plt.scatter(t_feat_all[i, 0], t_feat_all[i, 1], edgecolor=color[int(lable_cos[i])], c='none',
    s=30, marker='o', linewidth=1)
for i in range(image_co.shape[0] +text_co.shape[0], all_co.shape[0]):
    plt.scatter(t_feat_all[i, 0], t_feat_all[i, 1], edgecolor='black', c='deeppink',
    s=150, marker='*', linewidth=1)

plt.savefig('result/image_Texttsne3.pdf')
plt.show()