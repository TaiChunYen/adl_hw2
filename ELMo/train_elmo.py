import torch
import numpy as np
import pdb
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from elmo_wo_cnn import ELMo
from emb_layer import Emb_layer
import pickle
import os
import random

def collate_fn(data,embedding):
    batch = {}
    batch['fwdata']=[sum(data['fwdata'],[]) for data in datas]
    batch['bwdata']=[sum(data['bwdata'],[]) for data in datas]
    batch['fwlabel']=[sum(data['fwlabel'],[]) for data in datas]
    batch['bwlabel']=[sum(data['bwlabel'],[]) for data in datas]
    batch['sen_len']=[]
    fwd=[]
    bwd=[]
    fwl=[]
    bwl=[]
    sen_len=[]
    for i in range(len(batch['fwdata'])):
        
        batch['fwdata'][i].insert(0,embedding.to_index('<bos>'))
        while len(batch['fwdata'][i])>64:
              a=batch['fwdata'][i][0:64]
              #a.insert(0,embedding.to_index('<bos>'))
              fwd.append(a)
              sen_len.append(len(a))
              batch['fwdata'][i]=batch['fwdata'][i][64:]
        if len(batch['fwdata'][i])<=64:
              sen_len.append(len(batch['fwdata'][i]))
              for j in range(64-len(batch['fwdata'][i])):
                  batch['fwdata'][i].append(embedding.to_index('<pad>'))
              #batch['fwdata'][i].insert(0,embedding.to_index('<bos>'))
        fwd.append(batch['fwdata'][i])
    #pdb.set_trace()   
    batch['fwdata']=torch.tensor(fwd)
    batch['sen_len']=sen_len
     

    for i in range(len(batch['bwdata'])):
        
        batch['bwdata'][i].insert(0,embedding.to_index('<eos>'))
        while len(batch['bwdata'][i])>64:
              b=batch['bwdata'][i][0:64]
              #b.insert(0,embedding.to_index('<eos>'))
              bwd.append(b)
              batch['bwdata'][i]=batch['bwdata'][i][64:]
        if len(batch['bwdata'][i])<=64:
              for j in range(64-len(batch['bwdata'][i])):
                  batch['bwdata'][i].append(embedding.to_index('<pad>'))
              #batch['bwdata'][i].insert(0,embedding.to_index('<eos>'))
        bwd.append(batch['bwdata'][i])
    batch['bwdata']=torch.tensor(bwd)

    for i in range(len(batch['fwlabel'])):
        
        batch['fwlabel'][i].insert(-1,embedding.to_index('<eos>'))
        while len(batch['fwlabel'][i])>64:
              c=batch['fwlabel'][i][0:64]
              #c.insert(-1,embedding.to_index('<eos>'))
              fwl.append(c)
              batch['fwlabel'][i]=batch['fwlabel'][i][64:]
        if len(batch['fwlabel'][i])<=64:
              
              for j in range(64-len(batch['fwlabel'][i])):
                  batch['fwlabel'][i].append(embedding.to_index('<pad>'))
              #batch['fwlabel'][i].insert(-1,embedding.to_index('<eos>'))
        fwl.append(batch['fwlabel'][i])
    batch['fwlabel']=torch.tensor(fwl)

    for i in range(len(batch['bwlabel'])):
        
        batch['bwlabel'][i].insert(-1,embedding.to_index('<bos>'))
        while len(batch['bwlabel'][i])>64:
              d=batch['bwlabel'][i][0:64]
              #d.insert(-1,embedding.to_index('<bos>'))
              bwl.append(d)
              batch['bwlabel'][i]=batch['bwlabel'][i][64:]
        if len(batch['bwlabel'][i])<=64:
              
              for j in range(64-len(batch['bwlabel'][i])):
                  batch['bwlabel'][i].append(embedding.to_index('<pad>'))
              #batch['bwlabel'][i].insert(-1,embedding.to_index('<bos>'))
        bwl.append(batch['bwlabel'][i])
    batch['bwlabel']=torch.tensor(bwl)

    return batch



if __name__ == '__main__':

    with open('test-300000.pkl', 'rb') as f:
         datas=pickle.load(f)
      
    with open('embedding.pkl', 'rb') as f:
         embedding=pickle.load(f)
    
    emb = torch.nn.Embedding(embedding.vectors.size(0),embedding.vectors.size(1)).cpu()
    emb.weight = torch.nn.Parameter(embedding.vectors)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=ELMo(embedding.vectors.size(0),embedding.vectors.size(1)).to(device)#.cpu()
    #outemb=Emb_layer(512,embedding.vectors.size(0)).to(device)
    if os.path.isfile('./elmo.pkl'):
         model.load_state_dict(torch.load('./elmo.pkl'))
    if os.path.isfile('./outemb.pkl'):
         outemb.load_state_dict(torch.load('./outemb.pkl'))
    #pdb.set_trace() 
    
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    #optimizer2 = torch.optim.Adam(outemb.parameters(),lr=1e-3)
    #criterion = torch.nn.CrossEntropyLoss(ignore_index=embedding.to_index('<pad>'))
    criterion = torch.nn.NLLLoss(ignore_index=embedding.to_index('<pad>'))
    data=collate_fn(datas,embedding)
    '''dataloader = DataLoader(datas,batch_size=30,collate_fn=collate_fn)#,num_workers=4,collate_fn=
         
    trange = tqdm(enumerate(dataloader),
                 total=len(dataloader),
                 desc='traing')'''

    
    count=0
    print(len(data['fwdata']))
    #for i, batch in trange:
    '''randnum = random.randint(0,100)
    random.seed(randnum)
    random.shuffle(data['fwdata'])
    random.shuffle(data['bwdata'])
    random.shuffle(data['fwlabel'])
    random.shuffle(data['bwlabel'])
    random.shuffle(data['sen_len'])'''
    
    for i in range(len(data['fwdata'])//30):
         batch={}
         batch['fwdata']=data['fwdata'][i*30:i*30+30]
         batch['bwdata']=data['bwdata'][i*30:i*30+30]
         batch['fwlabel']=data['fwlabel'][i*30:i*30+30]
         batch['bwlabel']=data['bwlabel'][i*30:i*30+30]
         batch['sen_len']=data['sen_len'][i*30:i*30+30]
         
         vec=emb(batch['fwdata']).to(device)
         vec2=emb(batch['bwdata']).to(device)     
         y1,y2=model(vec,vec2,batch['fwlabel'].view(-1).to(device),batch['bwlabel'].view(-1).to(device))#

         y1=y1.view(vec.shape[0],vec.shape[1])
         y2=y2.view(vec2.shape[0],vec2.shape[1])
         loss1=0
         loss2=0
         ttlen=0
         #pdb.set_trace()
         for j in range(y1.shape[0]):
             loss1+=torch.sum(-y1[j][:batch['sen_len'][j]])
             loss2+=torch.sum(-y2[j][:batch['sen_len'][j]])
             ttlen+=batch['sen_len'][j]
         loss1=loss1/ttlen
         loss2=loss2/ttlen
         #pdb.set_trace()
         
         #z1=outemb(y1)#.to(device)
         #z2=outemb(y2)#.to(device)
         #loss1=criterion(y1,batch['fwlabel'].view(-1).to(device))#         
         #loss2=criterion(y2,batch['bwlabel'].view(-1).to(device))#        
         loss=(loss1+loss2)/2
              
         optimizer.zero_grad()
         #optimizer2.zero_grad()
         print(i, loss.item())
         loss.backward()
         optimizer.step()
         #optimizer2.step()
         count+=1
         if count>=100:
             torch.save(model.state_dict(),'./elmo.pkl')
             #torch.save(outemb.state_dict(),'./outemb.pkl')
             count=0
   
    torch.save(model.state_dict(),'./elmo.pkl')
    #torch.save(outemb.state_dict(),'./outemb.pkl')




