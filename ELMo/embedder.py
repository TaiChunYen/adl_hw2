import numpy as np
from ELMo.elmo_wo_cnn import ELMo
from ELMo.embedding import Embedding
import torch
import pdb
import pickle
import sys

class Embedder:
    """
    The class responsible for loading a pre-trained ELMo model and provide the ``embed``
    functionality for downstream BCN model.

    You can modify this class however you want, but do not alter the class name and the
    signature of the ``embed`` function. Also, ``__init__`` function should always have
    the ``ctx_emb_dim`` parameter.
    """

    def __init__(self, n_ctx_embs, ctx_emb_dim):
        """
        The value of the parameters should also be specified in the BCN model config.
        """
        self.n_ctx_embs = n_ctx_embs
        self.ctx_emb_dim = ctx_emb_dim
        # TODO
        
        sys.path.append('./ELMo/')
        with open('./ELMo/embedding.pkl', 'rb') as f:
            embedding=pickle.load(f)
        self.emb=embedding
        #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model=ELMo(self.emb.vectors.size(0),self.emb.vectors.size(1))#.to(self.device)
        self.model.load_state_dict(torch.load('./ELMo/elmo.pkl'))
        self.model.eval()

        

    def __call__(self, sentences, max_sent_len):
        """
        Generate the contextualized embedding of tokens in ``sentences``.

        Parameters
        ----------
        sentences : ``List[List[str]]``
            A batch of tokenized sentences.
        max_sent_len : ``int``
            All sentences must be truncated to this length.

        Returns
        -------
        ``np.ndarray``
            The contextualized embedding of the sentence tokens.

            The ndarray shape must be
            ``(len(sentences), min(max(map(len, sentences)), max_sent_len), self.n_ctx_embs, self.ctx_emb_dim)``
            and dtype must be ``np.float32``.
        """
        # TODO
        fwd=[]
        bwd=[]        
        llll=min(max(map(len, sentences)), max_sent_len)
        #pdb.set_trace()
        for i in range(len(sentences)):
              f=[]
              f=sentences[i][:]
              b=[]
              b=sentences[i][:]
              for j in range(len(sentences[i])):
                  f[j]=self.emb.to_index(f[j].lower())
                  b[j]=self.emb.to_index(b[j].lower())
              #list(map(lambda x:self.embeddding.to_index(x),sentences[i]))
              if len(sentences[i])<=llll:
                  for j in range(llll-len(sentences[i])):
                      f.append(self.emb.to_index('<pad>'))
                      b.append(self.emb.to_index('<pad>'))
              
              f.insert(0,self.emb.to_index('<bos>'))
              fwd.append(f)
              b.insert(0,self.emb.to_index('<eos>'))
              bwd.append(b)
        #pdb.set_trace()
        fwd=torch.tensor(fwd)
        bwd=torch.tensor(bwd)

        emb = torch.nn.Embedding(self.emb.vectors.size(0),self.emb.vectors.size(1))
        emb.weight = torch.nn.Parameter(self.emb.vectors)
        vec=emb(fwd)
        vec2=emb(bwd)
        embd,embd2,out,out_2,out2,out2_2=self.model(vec,vec2)
        #pdb.set_trace()
        embd=embd[1:]
        pembd=embd.permute([1,0,2])
        embd2=embd2[1:]
        pembd2=embd2.permute([1,0,2])
        out=out[1:]
        pout=out.permute([1,0,2])
        out_2=out_2[1:]
        pout_2=out_2.permute([1,0,2])
        out2=out2[1:]
        pout2=out2.permute([1,0,2])
        out2_2=out2_2[1:]
        pout2_2=out2_2.permute([1,0,2])
        #pdb.set_trace()
        '''sembd=torch.split(pembd,[1,pembd.shape[1]-1],1)[1]
        sembd2=torch.split(pembd2,[1,pembd2.shape[1]-1],1)[1]        
        sout=torch.split(pout,[1,pout.shape[1]-1],1)[1]
        sout_2=torch.split(pout_2,[1,pout_2.shape[1]-1],1)[1]
        sout2=torch.split(pout2,[1,pout2.shape[1]-1],1)[1]
        sout2_2=torch.split(pout2_2,[1,pout2_2.shape[1]-1],1)[1]'''

        l0=torch.cat((pembd,pembd2),-1)
        l1=torch.cat((pout,pout2),-1)
        l2=torch.cat((pout_2,pout2_2),-1)
        l0=l0.view(l0.shape[0],l0.shape[1],1,l0.shape[2])
        l1=l1.view(l1.shape[0],l1.shape[1],1,l1.shape[2])
        l2=l2.view(l2.shape[0],l2.shape[1],1,l2.shape[2])
        output=torch.cat((l0,l1,l2),2)
        #pdb.set_trace()
        output=output.detach().numpy()
        output=output.astype(np.float32)

        return output
