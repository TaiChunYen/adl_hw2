import pandas as pd

test=pd.read_csv('./data/classification/test.csv')
test.to_csv('./try.tsv', sep='\t', index=False)

