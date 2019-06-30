import torch
import numpy as np
import pdb
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

class ELMo(torch.nn.Module):

    def __init__(self,embedding0,embedding1):
        super(ELMo, self).__init__()
        self.embin = torch.nn.Linear(embedding1,512)
        self.embin2 = torch.nn.Linear(embedding1,512)
        self.lstm = torch.nn.LSTM(512, 512)
        self.p = torch.nn.Linear(512,512)
        self.lstm_2 = torch.nn.LSTM(512,512)
        self.p_2 = torch.nn.Linear(512,512)
        self.lstm2 = torch.nn.LSTM(512, 512)
        self.p2 = torch.nn.Linear(512,512)
        self.lstm2_2 = torch.nn.LSTM(512,512)
        self.p2_2 = torch.nn.Linear(512,512)
        self.asoft = torch.nn.AdaptiveLogSoftmaxWithLoss(512,50000,cutoffs = [10, 100, 1000])#

    def forward(self, vec, vec2, lab, lab2):#
        
        embd=self.embin(vec.permute(1,0,2))
        out,self.hidden  = self.lstm(embd)
        out=self.p(out)
        out_2,self.hidden_2 = self.lstm_2(out)
        out_2=self.p_2(out_2)
        outt=out_2.permute(1,0,2)
        outt=outt.contiguous().view(-1,512)
        outt=self.asoft(outt,lab)[0]#
        #outt=outt.view(-1,1)
        #pdb.set_trace()
        embd2=self.embin2(vec2.permute(1,0,2))
        out2,self.hidden2  = self.lstm2(embd2)
        out2=self.p2(out2)
        out2_2,self.hidden2_2 = self.lstm2_2(out2)
        out2_2=self.p2_2(out2_2)
        outt2=out2_2.permute(1,0,2)
        outt2=outt2.contiguous().view(-1,512)
        outt2=self.asoft(outt2,lab2)[0]#
        #outt2=outt2.view(-1,1)
        #pdb.set_trace()
        return outt,outt2
        #return embd, embd2,out, out2,out_2, out2_2
