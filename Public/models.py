import pandas as pd
import torch.nn as nn
import torch
from transformers import BertModel,BertTokenizer
from torch.utils.data import Dataset
from utils import merge_df

class SentimentClassifier(nn.Module):

    def __init__(self, freeze_bert = True):
        super(SentimentClassifier, self).__init__()
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')
        
        #Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False
        
        #Classification layer
        self.cls_layer = nn.Linear(768, 1)

    def forward(self, seq, attn_masks):
        
        #Feeding the input to BERT model to obtain contextualized representations
        out = self.bert_layer(seq, attention_mask = attn_masks).last_hidden_state
        logits = self.cls_layer(out[:,0,:])

        return logits


class BertDataset(Dataset):

    def __init__(self, path_or_pd, maxlen):
        if (type(path_or_pd)==str):
            self.df = pd.read_csv(path_or_pd)
        elif (type(path_or_pd)==pd.DataFrame):
            self.df = path_or_pd

        #Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        #Selecting the sentence and label at the specified index in the data frame
        sentence = self.df.loc[index, 'text']
        label = self.df.loc[index, 'label']

        #Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(sentence) #Tokenize the sentence
        tokens = ['[CLS]'] + tokens + ['[SEP]'] #Insering the CLS and SEP token in the beginning and end of the sentence
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] #Padding sentences
        else:
            tokens = tokens[:self.maxlen-1] + ['[SEP]'] #Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens) #Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids) #Converting the list to a pytorch tensor

        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label


if __name__ == "__main__":
    table_list = ['../Data/sst2/train.csv','../Data/sst2/val.csv']
    BertDataset(table_list,32)