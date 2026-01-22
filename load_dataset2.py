from torch.utils.data.dataset import Dataset
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os
import json
from transformers import BertConfig, BertTokenizer, BertModel
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
import string
import torch
import re
punctuation_string = string.punctuation     # CLIP的分词器对字符长度很不友好，尤其是数字和字符，每一个单个数字都会被识别成一个token
                                            # CLIP对带有中文拼音、罕见词的分词也很不友好，容易导致token超出最大长度76，比如PsittacosaurusYou被切分成了7个单词

device = "cuda" if torch.cuda.is_available() else "cpu"
#model, preprocess = clip.load("RN50", device=device)  #['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']

class CustomDataSet(Dataset):
    def __init__(
            self,
            images,
            texts,
            labs,
            ids,
            mask,
            text_id
    ):
        self.images = images
        self.texts = texts
        self.labs = labs
        self.ids = ids
        self.mask = mask
        self.text_id = text_id



    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        lab = self.labs[index]
        id = self.ids[index]
        mask = self.mask[index]
        text_id = self.text_id[index]
        return img, text, mask, text_id, lab, id

    def __len__(self):
        count = len(self.texts)
        return count


def cap_2_bertid(text, max_seq_length):
    from transformers import BertTokenizer
    #vocab_path = '/home/yh/code/PAN_L/bert_data/bert-base-uncased/vocab.txt'
    #tokenizer = BertTokenizer(vocab_path)
    text_dict = tokenizer.encode_plus(text, truncation=True, max_length=10)
    text_ids = text_dict['input_ids']  # 使用谷歌的词典
    attention_mask = text_dict['attention_mask']
    cap_len = len(text_ids)
    if cap_len < max_seq_length:
        pad = np.zeros((max_seq_length - cap_len, 1), dtype=np.int64)
        text_ids = np.append(text_ids, pad)
        attention_mask = np.append(attention_mask, pad)
    else:
        text_ids =np.array(text_ids[:max_seq_length])
        attention_mask = np.array(attention_mask[:max_seq_length])
        cap_len = max_seq_length

    return text_ids, attention_mask, cap_len


def load_dataset(max_text_len=10, name='NUS-WIDE-10K'):
    """
    Load captions and image features
    Possible options: wikipedia, NUS-WIDE-10K,Pascal-Sentence, xmedia
    """
    # loc = '../benchpark/' + name +'/'
    loc = '/home/yh/code/PAN_L/benchmark/' + name + '/'

    # Captions
    train_imgs, dev_imgs, test_imgs = [], [], []
    train_caps, dev_caps, test_caps = [], [], []
    train_labs, dev_labs, test_labs = [], [], []
    train_ids, dev_ids, test_ids = [], [], []
    train_cap_len,dev_cap_len,test_cap_len=[],[],[]
    train_mask,dev_mask,test_mask=[],[],[]
    train_file_path = os.path.join(loc, 'split/train.txt')
    dev_file_path = os.path.join(loc, 'split/valid.txt')
    test_file_path = os.path.join(loc, 'split/test.txt')
    print(test_file_path)

    # / home / yh / code / PAN_L / benchmark / wikipedia_dataset / split/test.txt
    with open(test_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            id, label = line.replace('\n', '').split(':')
            text = json.load(open(os.path.join(loc, 'texts', id + '.json'), 'r', encoding='UTF_8'))
            text = text.split(' ')
            text = ' '.join(text)
            text = re.sub('[{}]'.format(punctuation_string), "", text)
            text = re.sub('[\d]', '', text)
            text = text.strip()
            text = text.lstrip()
            try:
                # text_ids = clip.tokenize(text).detach().cpu()
                # from transformers import BertTokenizer
                # vocab_path='/home/yh/code/PAN_L/bert_data/bert-base-uncased/vocab.txt'
                # tokenizer = BertTokenizer(vocab_path)
                # tokenized_text = tokenizer(text, max_length=max_text_len, add_special_tokens=True, truncation=True, padding=False,return_tensors="pt")
                text_ids, attention_mask, cap_len=cap_2_bertid(text,max_text_len)
                test_caps.append(text_ids)
                test_mask.append(attention_mask)
                test_cap_len.append(cap_len)
                test_labs.append(int(label))
                test_ids.append(id)
            except:
                continue
            image = Image.open(os.path.join(loc, 'images', id + '.jpg'))  # 使用open函数打开后返回的图像模式都是‘RGB’或者灰度图其模式为‘L’

            # -----------****************************************_________________
            '''
            def _convert_image_to_rgb(image):
                return image.convert("RGB")

            def _transform(n_px):
                from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
                try:
                    from torchvision.transforms import InterpolationMode
                    BICUBIC = InterpolationMode.BICUBIC
                except ImportError:
                    BICUBIC = Image.BICUBIC

                return Compose([
                    Resize(224, interpolation=BICUBIC),
                    CenterCrop(224),
                    _convert_image_to_rgb,
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
                ])
            '''
            from torchvision import transforms as T
            transform = T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(224),
                # T.C
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ])

            def _convert_image_to_rgb(image):
                return image.convert("RGB")

            image = _convert_image_to_rgb(image)
            image = transform(image)
            test_imgs.append(image.numpy())

    # -----------****************************************_________________
    #print(len(test_imgs))
    print(len(test_caps))
    print(' Successfully process test data')
    test_data = {'image': test_imgs, "text": test_caps,"mask":test_mask,"text_ids":test_cap_len, "label": test_labs, 'ids': test_ids}
    with open('test.pkl', 'wb') as f:
        pickle.dump(test_data, f)

    with open(train_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            id, label = line.replace('\n', '').split(':')
            text = json.load(open(os.path.join(loc, 'texts', id + '.json'), 'r', encoding='UTF_8'))
            text = text.split(' ')
            text = ' '.join(text)
            text = re.sub('[{}]'.format(punctuation_string), "", text)  # 去掉符号
            text = re.sub('[\d]', '', text)  # 去掉数字
            text = text.strip()
            text = text.lstrip()
            try:
                text_ids, attention_mask, cap_len=cap_2_bertid(text,max_text_len)
                train_caps.append(text_ids)
                train_mask.append(attention_mask)
                train_cap_len.append(cap_len)
                train_labs.append(int(label))
                train_ids.append(id)
            except:
                continue

            image = Image.open(os.path.join(loc, 'images', id + '.jpg'))  # 使用open函数打开后返回的图像模式都是‘RGB’或者灰度图其模式为‘L’
            from torchvision import transforms as T
            transform = T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(224),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ])

            def _convert_image_to_rgb(image):
                return image.convert("RGB")

            image = _convert_image_to_rgb(image)
            image = transform(image)
            train_imgs.append(image.numpy())
    print(len(train_imgs))
    print(len(train_caps))
    #test_data = {'image': test_imgs, "text": test_caps,"mask":test_mask,"text_ids":test_text_ids, "label": test_labs, 'ids': test_ids}

    train_data = {'image': train_imgs, "text": train_caps,"mask":train_mask,"text_ids":train_cap_len, "label": train_labs, 'ids': train_ids}
    with open('train.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    print(' Successfully process training data')

    with open(dev_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            id, label = line.replace('\n', '').split(':')
            text = json.load(open(os.path.join(loc, 'texts', id + '.json'), 'r', encoding='UTF_8'))
            text = text.split(' ')
            text = ' '.join(text)
            text = re.sub('[{}]'.format(punctuation_string), "", text)
            text = re.sub('[\d]', '', text)
            text = text.strip()
            text = text.lstrip()
            try:
                text_ids, attention_mask, cap_len=cap_2_bertid(text,max_text_len)
                dev_caps.append(text_ids)
                dev_mask.append(attention_mask)
                dev_cap_len.append(cap_len)
                dev_labs.append(int(label))
                dev_ids.append(id)
            except:
                continue
            image = Image.open(os.path.join(loc, 'images', id + '.jpg'))  # 使用open函数打开后返回的图像模式都是‘RGB’或者灰度图其模式为‘L’
            transform = T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(224),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ])

            def _convert_image_to_rgb(image):
                return image.convert("RGB")

            image = _convert_image_to_rgb(image)
            image = transform(image)
            # image = transform(image).unsqueeze(0)
            dev_imgs.append(image.numpy())
    print(len(dev_imgs))
    print(len(dev_caps))

    dev_ims = np.array(dev_imgs)
    dev_caps = np.array(dev_caps)
    #train_data = {'image': train_imgs, "text": train_caps,"mask":train_mask,"text_ids":train_text_ids, "label": train_labs, 'ids': train_ids}
    valid_data = {'image': dev_imgs, "text": dev_caps,"mask":dev_mask,"text_ids":dev_cap_len, "label": dev_labs, 'ids': dev_ids}
    with open('dev.pkl', 'wb') as f:
        pickle.dump(valid_data, f)
    print('Successfully process dev data')

def load_dataloader(name, bsz):
    train_loc = '/opt/data/private/text/PAN_L/benchmark/'+name+'/train.pkl'
    test_loc='/opt/data/private/text/PAN_L/benchmark/'+name+'/test.pkl'
    dev_loc='/opt/data/private/text/PAN_L/benchmark/'+name+'/dev.pkl'
    #train_loc = 'data/'+name+'/train_bert_feature.pkl'
    #test_loc = 'data/'+name+'/test_bert_feature.pkl'
    with open(train_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        train_labels = data['label']
        #train_label=np.array([np.argmax(item) for item in train_labels])
        train_texts=data['text']
        train_images = data['image']
        train_ids = data['ids']
        train_mask=data['mask']
        train_text_id=data['text_ids']

    with open(test_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        test_labels = data['label']
        #test_label = np.array([np.argmax(item) for item in test_labels])
        #train_label=np.array([np.argmax(item) for item in train_labels])
        test_texts= data['text']
        test_images = data['image']
        test_ids = data['ids']
        test_mask=data['mask']
        test_text_id=data['text_ids']

    with open(dev_loc, 'rb') as f_pkl:
        data = pickle.load(f_pkl)
        dev_labels = data['label']
        #train_label=np.array([np.argmax(item) for item in train_labels])
        dev_texts=data['text']
        dev_images = data['image']
        dev_ids = data['ids']
        dev_mask=data['mask']
        dev_text_id=data['text_ids']

    imgs = {'train': train_images, 'test': test_images,'dev':dev_images}
    texts = {'train': train_texts,  'test': test_texts,'dev':dev_texts}
    labs = {'train': train_labels, 'test': test_labels,'dev':dev_labels}
    ids = {'train': train_ids, 'test': test_ids,'dev':dev_ids}
    mask={'train':train_mask,'test':test_mask,'dev':dev_mask}
    text_id={'train':train_text_id,'test':test_text_id,'dev':dev_text_id}

    dataset = {x: CustomDataSet(images=imgs[x], texts=texts[x], labs=labs[x],ids=ids[x],mask=mask[x],text_id=text_id[x])
               for x in ['train', 'dev','test']}

    shuffle = {'train': True, 'test': False,'dev': False}

    from torch.utils.data import DataLoader
    dataloader = {x: DataLoader(dataset[x], batch_size=bsz,
                                shuffle=shuffle[x], num_workers=0) for x in ['train','dev','test']}

    return dataloader
if __name__ == '__main__':
    names = 'NUS-WIDE-10K'
    #load_dataset(name=names)

    dataloader=load_dataloader(names,4)  #name
    train_loader=dataloader['train']
    #for imgs,texts,labs,ids in train_loader:
    for img, text, mask, _, lab, _ in train_loader:
        print(text)

        if type(text)==list:
            print('error')
        #print('data loader process')

        pass
