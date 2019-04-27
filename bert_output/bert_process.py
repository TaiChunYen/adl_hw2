import pandas as pd
import numpy as np

bertans=pd.read_csv('test_results.tsv', sep='\t',header=None)
ans=pd.read_csv('../data/classification/sample_submission.csv')

for i in range(len(ans)):
    ans['label'][i]=pd.Series.idxmax(bertans.iloc[i])

ans.to_csv(r'./bert_ans.csv',columns=['Id','label'],index=False,sep=',')
