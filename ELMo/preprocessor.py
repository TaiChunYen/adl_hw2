import json
#import logging
from multiprocessing import Pool
from tqdm import tqdm
import pdb
import time
import sys
from itertools import chain
from collections import Counter

class Preprocessor:
    """

    Args:
        embedding_path (str): Path to the embedding to use.
    """
    def __init__(self, embedding):
        self.embedding = embedding
        
        #self.logging = logging.getLogger(name=__name__)

    def tokenize(self, sentence):
        """ Tokenize a sentence.
        Args:
            sentence (str): One string.
        Return:
            indices (list of str): List of tokens in a sentence.
        """        
        # TODO
        
        
                         
                             
        tokens=sentence.split()

                                                                                                        
        #tokens = WordPunctTokenizer().tokenize(sentence)
        
        #pdb.set_trace()
        return tokens

    def sentence_to_indices(self, sentence):
        """ Convert sentence to its word indices.
        Args:
            sentence (str): One string.
        Return:
            indices (list of int): List of word indices.
        """
        # TODO
        # Hint: You can use `self.embedding`
        
        indices=[]
        
        for word in self.tokenize(sentence):
            indice=self.embedding.to_index(word)            
                
            indices.append(indice)   
               
        return indices

    def collect_words(self, data_path, n_workers=1):
        with open(data_path) as f:
            data=[]
            for line in f:
                data.append([line])                     

        utterances = []
        utterances = list(chain(*data[300000:600000]))
        #utterances = data[0:1000000]#
        '''tokei=0
        for sample in data[0:10]:
            
            time.sleep(0.25)
            sys.stdout.write("\r%d%%" % tokei)
            sys.stdout.flush()
            tokei+=1
            utterances += sample'''
        #pdb.set_trace()
        utterances = list(set(utterances))
        #pdb.set_trace()
        chunks = [
            ' '.join(utterances[i:i + len(utterances) // n_workers])
            for i in range(0, len(utterances), len(utterances) // n_workers)
        ]
        with Pool(n_workers) as pool:
            chunks = pool.map_async(self.tokenize, chunks)
            words = sum(chunks.get(), [])#
            dicts = {}
            dicts=Counter(words)
        words=set()
        list(map(lambda x:words.add(x[0]), dicts.most_common(50000)))
        #words=set(words)    
        #pdb.set_trace()
		
        

        return words

    def get_dataset(self, data_path, n_workers=1, dataset_args={}):
        """ Load data and return Dataset objects for training and validating.

        Args:
            data_path (str): Path to the data.
            valid_ratio (float): Ratio of the data to used as valid data.
        """
        #self.logging.info('loading dataset...')
        with open(data_path) as f:
            dataset = []
            for line in f:
                dataset.append([line])

        #self.logging.info('preprocessing data...')
        dataset=dataset[300000:600000]#
        results = [None] * n_workers
        with Pool(processes=n_workers) as pool:
            for i in range(n_workers):
                batch_start = int((len(dataset) // n_workers) * i)
                if i == n_workers - 1:
                    batch_end = int(len(dataset) // 1)
                else:
                    batch_end = int((len(dataset) // n_workers) * (i + 1))

                batch = dataset[batch_start: batch_end]
                results[i] = pool.apply_async(self.preprocess_samples, [batch])

                # When debugging, you'd better not use multi-thread.
                #results[i] = self.preprocess_samples(batch)

            pool.close()
            pool.join()
        
        
        processed = []
        '''getresult=[]
        getresult=map(lambda x:x.get(),results)
        processed = list(chain(*getresult))'''
        
        for result in results:
            processed += result.get()
        #pdb.set_trace()

        padding = self.embedding.to_index('</s>')
        return processed

    def preprocess_samples(self, dataset):
        """ Worker function.

        Args:
            dataset (list of dict)
        Returns:
            list of processed dict.
        """
        processed = []
        for sample in tqdm(dataset):
            processed.append(self.preprocess_sample(sample))

        return processed

    def preprocess_sample(self, data):
        """
        Args:
            data (dict)
        Returns:
            dict
        """
        processed = {}
        

        # process messages-so-far
        processed['fwdata'] = []
        processed['bwdata'] = []
        processed['fwlabel'] = []
        processed['bwlabel'] = []
        for message in data:#message is a word,data are words
            a=self.sentence_to_indices(message.lower())
            #a.insert(0,self.embedding.to_index('<bos>'))            
            processed['fwdata'].append(a)
            b=self.sentence_to_indices(message.lower())
            #b.reverse()
            #b.insert(0,self.embedding.to_index('<bos>'))
            processed['bwdata'].append(b)
            

            c=self.sentence_to_indices(message.lower())
            #c.insert(-1,self.embedding.to_index('<eos>'))          
            processed['fwlabel'].append(c)            

            d=self.sentence_to_indices(message.lower())
            #d.reverse() 
            #d.insert(-1,self.embedding.to_index('<eos>'))        
            processed['bwlabel'].append(d)

        #processed['fwdata'].insert(0,[self.embedding.to_index('<bos>')]) 
        processed['bwdata'].reverse()
        #processed['bwdata'].insert(0,[self.embedding.to_index('<eos>')]) 
        #processed['fwlabel'].insert(-1,[self.embedding.to_index('<eos>')]) 
        processed['bwlabel'].reverse() 
        #processed['bwlabel'].insert(-1,[self.embedding.to_index('<bos>')])
        # process options
        

        

        return processed
