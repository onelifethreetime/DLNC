from transformers import BertConfig, BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import transforms
import torch.utils.data as data
from pytorch_pretrained_bert.optimization import BertAdam
import os
#from load_data import load_dataset
import numpy as np
from tqdm import tqdm
# 读取bert模型
model_name = 'bert-base-uncased'
model_path = '/opt/data/private/text/PAN_L/bert_data/bert-base-uncased'
# model_name = 'bert-base-uncased'
# model_path = 'model/DictBERT-base'
# b. 导入配置文件
model_config = BertConfig.from_pretrained(model_name)
# 修改配置
model_config.output_hidden_states = True
model_config.output_attentions = True
# 通过配置和路径导入模型
# bert_model = BertModel.from_pretrained(model_path, config=model_config)


class Model(nn.Module):  # 模型代码
    def __init__(self, label_dim):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(model_path, config=model_config)  # 调用你下载的预训练模型
        for param in self.bert.parameters():
            param.requires_grad = True
        self.dropout=nn.Dropout(0.7)
        self.fc = nn.Linear(768, label_dim)  # 转换输出

    def forward(self, text, mask):
        device_num = text.device.index
        token_type_ids = torch.zeros(text.shape[0], text.shape[1]).long().to(device=device_num)
        # text = text.unsqueeze(1)
        # mask = mask.unsqueeze(1)
        res = self.bert(text, token_type_ids=token_type_ids, attention_mask=mask)  # 返回结果向量
        encoder_out = res[0]  # [1, length, 768]
        pooled = res[1]  # [1, 768]
        # pooled = torch.mean(encoder_out, dim=1)
        pooled=self.dropout(pooled)
        out = self.fc(pooled)
        return out, pooled, encoder_out

class VGGNet(nn.Module):
    def __init__(self,label_dim):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        from torchvision import models
        self.vgg = models.vgg19_bn(pretrained=True)
        for parameter in self.vgg.parameters():
            parameter.required_grad = True
        self.vgg_features = self.vgg.features
        self.fc_features = nn.Sequential(*list(self.vgg.classifier.children())[:-2])
        self.dropout=nn.Dropout(0.5)
        self.fc=nn.Linear(4096,label_dim)
    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = self.vgg_features(x).view(x.shape[0], -1)
        features = self.fc_features(features)
        feature_classfier=self.dropout(features)
        out=self.fc(feature_classfier)
        return features,out


#vgg = VGGNet().cuda()
#vgg = vgg.eval()
#fea = vgg(img).cpu().data.numpy()

def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    max_preds = preds.argmax(dim=1, keepdim=True)  # 获得第二个维度最大值的index，keepdim保持原来的preds维度不变
    correct = max_preds.squeeze(1).eq(y).float()   # 如果不float的话acc一直是0，因为correct应该是布尔型
    return correct.sum() / max_preds.size(0)


def train(model, loader, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()
    for img, text, mask, _, label, _ in loader:
        optimizer.zero_grad()
        text = text.to(device)
        label = label.long().to(device)
        img=img.to(device)
        #predictions, pooled, encoder_out = model(text, mask.to(device))  # 不需要压缩维度，交叉熵损失函数的期望输入时（batch_size, n_class),label is [batch_size]
        feature,predictions = model(img)
        loss = criterion(predictions, label)
        acc = categorical_accuracy(predictions, label)
        loss.backward()
        optimizer.step()

        epoch_loss += float(loss)
        epoch_acc += float(acc)
    return epoch_loss / len(loader), epoch_acc / len(loader)


def evaluate(model, loader, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()
    with torch.no_grad():
        for img, text, mask, _, label, _ in loader:
            text = text.to(device)
            label = label.long().to(device)
            img=img.to(device)
            #predictions, _, encoder_out = model(text, mask.to(device))  # 不需要压缩维度，交叉熵损失函数的期望输入时（batch_size, n_class),label is [batch_size]
            feature, predictions = model(img)
            loss = criterion(predictions, label)
            acc = categorical_accuracy(predictions, label)

            epoch_loss += float(loss)
            epoch_acc += float(acc)
    return epoch_loss / len(loader), epoch_acc / len(loader)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == '__main__':
    '''
    #vgg=VGGNet()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # data parameters
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    # data parameters
    MAX_EPOCH = 20
    batch_size = 160
    max_length = 10  # nuswide-10, pascal-56, wikidata-150, xmedia-70
    num_class = 10  # nuswide-10, pascal-20, wikidata-10, xmedia-200
    betas = (0.5, 0.999)
    weight_decay = 0
    best_valid_acc = 0.0
    print('...Data loading is beginning...')

    from load_dataset2 import load_dataloader
    name = 'NUS-WIDE-10K'
    #load_dataset(name=names)
    dataloader=load_dataloader(name,batch_size)  #name

    #dataloader = load_dataset(bsz=batch_size, max_text_len=max_length, name='wikipedia_dataset')
    train_loader = dataloader['train']
    dev_loader = dataloader['dev']
    test_loader = dataloader['test']

    #model_ft = Model(label_dim=num_class).to(device)
    model_ft = VGGNet(label_dim=num_class).to(device)
    model_ft = nn.DataParallel(model_ft, device_ids=[0, 1])
    model_ft = model_ft.to(device)
    params_to_update = list(model_ft.parameters())

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # Observe that all parameters are being optimized
    lr=5e-5
    optimizer = optim.Adam(params_to_update, lr=lr, betas=betas)

    #optimizer = BertAdam(params_to_update, lr=5e-5, warmup=0.05)  # 优化器  论文推荐的学习率5e-5, 3e-5, 2e-5

    for epoch in range(MAX_EPOCH):
        start_time = time.time()

        train_loss, train_acc = train(model_ft, train_loader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(model_ft, dev_loader, criterion, device)  # 注意valid集是评估train训练好的模型

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print('================================================')
        print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'Train loss: {train_loss:.3f} | Train accuracy: {train_acc * 100:.2f}%')
        print(f'Valid loss: {valid_loss:.3f} | Valid accuracy: {valid_acc * 100:.2f}%')
        #print(f'Test loss: {test_loss:.3f} | Test accuracy: {test_acc * 100:.2f}%')
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_epoch = epoch+1
            torch.save(model_ft.state_dict(), 'nus_wide_bert.pt')  # text_trans.pt
            print('New best model')

    print('================================================')
    print(f'Best valid accuracy: {best_valid_acc * 100:.2f}, Epoch: {best_epoch}')


    '''
    # # fine-tune
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    # # data parameters
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    # # data parameters
    batch_size = 160
    # max_length = 150  # nuswide-10, pascal-56, wikidata-150, xmedia-70
    num_class = 10  # nuswide-10, pascal-20, wikidata-10, xmedia-200
    # betas = (0.5, 0.999)
    # print('...Data loading is beginning...')
    from load_dataset2 import load_dataloader
    name = 'NUS-WIDE-10K'
    #load_dataset(name=names)
    dataloader=load_dataloader(name,batch_size)

    #dataloader = load_dataset(bsz=batch_size, max_text_len=max_length, name='wikipedia_dataset')
    # print(dataloader)
    train_loader = dataloader['train']
    # dev_loader = dataloader['dev']
    test_loader = dataloader['test']

    model_ft = VGGNet(label_dim=num_class).to(device)
    #model_ft = Model(label_dim=num_class).to(device)

    #model_ft = nn.DataParallel(model_ft, device_ids=[0, 1])
    #model_ft = model_ft.to(device)

    model_ft.load_state_dict(torch.load('/opt/data/private/text/PAN_L/nus_wide_vgg_b160.pt', map_location=torch.device("cpu")), False)
    #model_ft.load_state_dict(torch.load('/opt/data/private/text/PAN_L/nus_wide_bert_66_61.pt', map_location=torch.device("cpu")), False)

    model_ft = nn.DataParallel(model_ft, device_ids=[0,1])
    model_ft = model_ft.to(device)
    # criterion = nn.CrossEntropyLoss()
    # criterion = criterion.to(device)
    # test_loss, test_acc = evaluate(model_ft, test_loader, criterion, device)
    # print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')


    def extractor_features(model, loader):
         model.eval()
         with torch.no_grad():
             for img, text, mask, _, _, ids in tqdm(loader):
                 text = text.to(device)
                 img=img.to(device)
                 #predictions, pooled, encoder_out = model(text, mask.to(device))
                 feature, predictions = model(img)
                 for i in range(img.shape[0]):
                     # id = ids[i].item()
                     id = list(ids)[i]  # wikipedia的id是字符串
                     img_feature=feature[i].cpu()
                     img_feature=img_feature.numpy().reshape(1,4096)
                     #text_feature = pooled[i].cpu()  # TENSOR
                     #text_feature = text_feature.numpy()
                     #np.savetxt("/opt/data/private/text/PAN_L/data/nus_wide/img_feature/" + str(id) + ".txt",img_feature,fmt='%f',delimiter=',')
                     np.savetxt("/opt/data/private/text/PAN_L/data/nus_wide/img_feature/" + str(id) + ".txt",img_feature,fmt='%f',delimiter=', ',newline='',header='[',footer=']',comments='')
                     #np.savetxt("/opt/data/private/text/PAN_L/data/nus_wide/img_feature/" + str(id) + ".txt",img_feature,fmt='%f', delimiter=',',newline=',')
                     #np.savetxt("/opt/data/private/text/PAN_L/data/nus_wide/img_feature/" + str(id) + ".txt",img_feature,fmt='%f', delimiter=',',header='[',footer=']',comments='')
                     #np.loadtxt("/opt/data/private/text/PAN_L/data/nus_wide/img_feature/" + str(id) + ".txt")
                     #np.savetxt("/u01/isi/sy/Discrete/data/wikipedia/text_features/" + str(id) + ".txt",text_feature, fmt='%f', delimiter=',')
                     #np.savetxt("/u01/isi/sy/bert/data_without_finetune/wikipedia/text_features/" + str(id) + ".txt",text_feature, fmt='%f', delimiter=',')

    '''
    def extractor_features(model, loader):
        model.eval()
        with torch.no_grad():
            for img, text, mask, _, _, ids in tqdm(loader):
                text = text.to(device)
                img = img.to(device)
                predictions, pooled, encoder_out = model(text, mask.to(device))
                for i in range(img.shape[0]):
                    # id = ids[i].item()
                    id = list(ids)[i]  # wikipedia的id是字符串
                    text_feature = pooled[i].cpu()  # TENSOR
                    text_feature = text_feature.numpy()
                    np.savetxt("/opt/data/private/text/PAN_L/data/nus_wide/text_feature/" + str(id) + ".txt",text_feature, fmt='%f', delimiter=',')
                    # np.savetxt("/opt/data/private/text/PAN_L/data/nus_wide/text_feature/" + str(id) + ".txt",text_feature, fmt='%f', delimiter=',')
    '''
    print('Train Set')
    extractor_features(model_ft, train_loader)
    print('Test Set')
    # extractor_features(model_ft, dev_loader)
    extractor_features(model_ft, test_loader)
