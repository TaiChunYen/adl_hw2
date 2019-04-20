import torch
import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader

from BCN.dataset import Part1Dataset
from common.vocab import Vocab
from pytorch_pretrained_bert import BertConfig
from pytorch_pretrained_bert import BertForSequenceClassification


def collate_fn(train):

    batch = {}

    batch['tid']=[]
    #vid=[]
    batch['tlab']=[]
    #vlab=[]
    
    batch['tid']+=[x['text_word'] for x in train]
    #vid+=[x['text_word'] for x in valid]
    batch['tlab']+=[x['label'] for x in train]
    #vlab+=[x['label'] for x in valid]
    batch['tlab']=torch.LongTensor(batch['tlab'])
    #vlab=torch.LongTensor(vlab)

    tml=max(map(len,batch['tid']))
    #vml=max(map(len,vid))
    #llll=max(tml,vml)
    
    for i in range(len(batch['tid'])):
        if len(batch['tid'][i])<tml:
            for j in range(tml-len(batch['tid'][i])):
                batch['tid'][i].append(0)
    batch['tid']=torch.LongTensor(batch['tid'])

    '''for i in range(len(vid)):
        if len(vid[i])<vml:
            for j in range(vml-len(vid[i])):
                vid[i].append(0)
    vid=torch.LongTensor(vid)'''

    return batch

if __name__ == '__main__':


    with open('./dataset/classification/train.pkl','rb') as f:
        train=pickle.load(f)

    with open('./dataset/classification/dev.pkl','rb') as f:
        valid=pickle.load(f)
    epoch=10
   
    num_labels = 5

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config = BertConfig(vocab_size_or_config_json_file=17970, hidden_size=300,
        num_hidden_layers=6, num_attention_heads=6, intermediate_size=1024)
    
    model = BertForSequenceClassification(config, num_labels).to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
                                                                               

    for e in range(epoch):
        model.train()
        loss = 0
        dataloader = DataLoader(train,batch_size=64,collate_fn=collate_fn,shuffle=True)
        iter_in_epoch = len(dataloader)
        description = ('training %d' % e)
        trange = tqdm(enumerate(dataloader),total=iter_in_epoch,desc=description)

        for i, batch in trange:

            batch_loss=model(batch['tid'].to(device), token_type_ids=None, attention_mask=None, labels=batch['tlab'].to(device))
                     
            optimizer.zero_grad()         
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss.item()
            trange.set_postfix(loss=loss / (i + 1))
                
        loss /= iter_in_epoch
        print('loss=%f\n' % loss)


        model.eval()
        loss = 0
        dataloader = DataLoader(valid,batch_size=64,collate_fn=collate_fn,shuffle=False)
        iter_in_epoch = len(dataloader)
        description = 'validing'
        trange = tqdm(enumerate(dataloader),total=iter_in_epoch,desc=description)

        for i, batch in trange:

            batch_loss=model(batch['tid'].to(device), token_type_ids=None, attention_mask=None, labels=batch['tlab'].to(device))                                 

            loss += batch_loss.item()
            trange.set_postfix(loss=loss / (i + 1))
                
        loss /= iter_in_epoch
        print('loss=%f\n' % loss)






