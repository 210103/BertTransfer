import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import OrderedDict

hyperParamslist = ['BATCH_SIZE','MAX_LEN','LEARNING_RATE',"EPOCHS"]

def get_accuracy(preds,labels):
    pred_flat=preds.flatten()
    pred_flat = (pred_flat>=0.5).astype(int)
    labels_flat=labels.flatten()
    return accuracy_score(labels_flat,pred_flat)

     
def full_permutation(paramDictLists):
    
    paramLists = [v for k,v in paramDictLists.items()]
    table = pd.DataFrame(columns=hyperParamslist)
    prefix = [0]*len(paramLists)
    paramNum = [len(v) for v in paramLists]    

    def full_permutation_inner(i:int):

        if (i==len(paramNum)):
            addedLine = [paramLists[loc][v]  for loc,v in enumerate(prefix)]
            table.loc[len(table)] = addedLine
            return 
        for j in range(paramNum[i]):
            prefix[i]=j
            full_permutation_inner(i+1)
    

    full_permutation_inner(0)
    return table

def merge_df(path_lists):
    table_lists = [pd.read_csv(path) for path in path_lists]
    return pd.concat(table_lists)

if __name__=="__main__":
    hyperParams = OrderedDict({
    'BATCH_SIZES':[16,32,64],
    'MAX_LENS':[32,64,128],
    'LEARNING_RATES':[2e-5,2e-4,1e-4,2e-3,1e-3],
    "EPOCHSS":[100,200]
    })
    table = full_permutation(hyperParams)
    print(table)