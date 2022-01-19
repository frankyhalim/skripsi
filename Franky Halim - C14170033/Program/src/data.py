import logging
import torch
from helpers import pad
from transformers import BertModel, BertTokenizer 

logger = logging.getLogger(__name__)

# Calculating padding per batch 
def pad_batch_collate(batch):
    elem = batch[0] 
    final_dictionary = {} 
    
    # Iterate through all key dictionary
    for key in elem:
        
        # For each data key in batch append to list of feature 
        feature_list = [d[key] for d in batch]
        if key == "sent_rep_token_ids":
            
            feature_list = pad(feature_list, -1) 
            sent_rep_token_ids = torch.tensor(feature_list, dtype=torch.long)

            sent_rep_mask = ~(sent_rep_token_ids == -1) 
            sent_rep_token_ids[sent_rep_token_ids == -1] = 0 

            final_dictionary["sent_rep_token_ids"] = sent_rep_token_ids
            final_dictionary["sent_rep_mask"] = sent_rep_mask
            continue  
        if key == "input_ids":
            input_ids = feature_list

            # Attention
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [[1] * len(ids) for ids in input_ids]

            input_ids_width = max([len(ids) for ids in input_ids])
            input_ids = pad(input_ids, 0, width=input_ids_width) 
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            
            attention_mask = pad(attention_mask, 0) 
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            
            final_dictionary["input_ids"] = input_ids
            final_dictionary["attention_mask"] = attention_mask
            
            continue
        
        if key in ("source", "target"):
            final_dictionary[key] = feature_list
            continue
        
        if key in ("labels", "token_type_ids"):
            feature_list = pad(feature_list, 0) 

        feature_list = torch.tensor(feature_list, dtype=torch.long)
        final_dictionary[key] = feature_list

    return final_dictionary

# Bert data class
class BertData():
    def __init__(self, pre_trained_bert_model):
        self.model = BertModel.from_pretrained(pre_trained_bert_model) 
        self.tokenizer = BertTokenizer.from_pretrained(pre_trained_bert_model) 
        self.sep_token = '[SEP]' 
        self.cls_token = '[CLS]' 
        self.pad_token = '[PAD]' 
        self.sep_vid = self.tokenizer.vocab[self.sep_token] 
        self.cls_vid = self.tokenizer.vocab[self.cls_token] 
        self.pad_vid = self.tokenizer.vocab[self.pad_token] 
        
    def get_input_features(self, source, target, labels, min_seq_length=1, max_seq_length=512, delimiter='<q>'):
        source = source.split(delimiter)
        original_src_txt = source 
        
        source = [sent.strip() for sent in source]
        idxs = [i for i, s in enumerate(source) if (len(s) >= min_seq_length)] 
        
        labels = labels.split('<q>')
        labels = [int(l) for l in labels]
    
        tokenized_source = self.tokenizer.tokenize(self.sep_token.join(source))         
        temp = []
        tokens = []
        flag = True
        for sub_token in tokenized_source:
            if flag:
                tokens.append(self.cls_token)
                flag = False
                
            tokens.append(sub_token)
            
            if sub_token == self.sep_token:
                temp.append(tokens)
                tokens = []
                flag = True     
        
        # Check exceeded length (max. token length = 512)
        res = []
        total_len = 0
        for idx, l in enumerate(temp):
            total_len+=len(l)
            if total_len > max_seq_length:
                break
            res.append(temp[idx])
            
        source_subtokens = [t for s in res for t in s]  
        input_ids = self.tokenizer.convert_tokens_to_ids(source_subtokens) 
        
        # Segments ids / Token type ids  
        _segs = [-1] + [i for i, t in enumerate(input_ids) if t == self.sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]        
        segments_ids = [] 
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]        
        
        cls_ids = [i for i, t in enumerate(input_ids) if t == self.cls_vid]
        labels = labels[:len(cls_ids)]
        source = [original_src_txt[i] for i in idxs]
        
        return input_ids, segments_ids, labels, cls_ids, source, target  
    
    def add_special_token(self, text, delimiter='<q>'): 
        text = text.split(delimiter)
        text = ' '.join([f'{self.cls_token} {sent} {self.sep_token}' for sent in text])  
        return text
    
    def get_input_ids(self, text, max_seq_length=512):
        sep_token = str(self.sep_token)
        cls_token = str(self.cls_token)

        # adds a '[CLS]' token between each sentence and outputs `input_ids`
        # If the CLS or SEP tokens exist in the document as part of the dataset, then
        # set them to UNK
        unk_token = str(self.tokenizer.unk_token)
        source = [
            sent.replace(sep_token, unk_token).replace(cls_token, unk_token)
            for sent in text
        ]
        tokenized_source = self.tokenizer.tokenize(sep_token.join(source))         
        temp = []
        tokens = []
        flag = True
        for sub_token in tokenized_source:
            if flag:
                tokens.append(self.cls_token)
                flag = False
                
            tokens.append(sub_token)
            
            if sub_token == self.sep_token:
                temp.append(tokens)
                tokens = []
                flag = True     
        
        # Check exceeded length (max. token length = 512)
        res = []
        total_len = 0
        for idx, l in enumerate(temp):
            total_len+=len(l)
            if total_len > max_seq_length:
                break
            res.append(temp[idx])
        source_subtokens = [t for s in res for t in s]  
        input_ids = self.tokenizer.convert_tokens_to_ids(source_subtokens) 
        return input_ids
    
    def get_sentences_length(self, text, delimiter='<q>'):
        text = text.split(delimiter)
        text = [f'{self.cls_token} {sent} {self.sep_token}' for sent in text]
        sentences_length = [len(self.tokenize(sent)) for sent in text]
        return sentences_length
    
    def tokenize(self, text): 
        return self.tokenizer.tokenize(text) 
    
    def convert_tokens_to_ids(self,tokens): 
        return self.tokenizer.convert_tokens_to_ids(tokens)
    
    def get_model(self):
        return self.model
    
    def get_model_with_config(self, model_name, model_config):
        return BertModel.from_pretrained(model_name, config=model_config)
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_special_token(self):
        return self.cls_token, self.sep_token, self.pad_token

# Dataloader
class DatasetIndoNews(torch.utils.data.Dataset):
    def __init__(self, doc, bert, max_seq_length=512):
        super(DatasetIndoNews,self).__init__()
        self.doc = doc
        self.bert = bert
        self.features = self.get_input_features(doc)   
        
    def get_input_features(self, doc):
        input_features = []
        for item in doc:
            input_ids, token_type_ids, labels, cls_ids, source, target = self.bert.get_input_features(item['source'], item['target'], item['labels'])
            bert_features = {"input_ids": input_ids, "token_type_ids": token_type_ids,
                           "labels": labels, "sent_rep_token_ids": cls_ids, 
                           "source":source, "target":target}
            input_features.append(bert_features)
        return input_features
    
    def get_bert(self):
        return self.bert
    
    def get_doc(self, idx):
        return self.doc[idx]
    
    def get_len_doc(self):
        return len(self.doc)
    
    def __getitem__(self, idx):
        return self.features[idx]
    
    def __len__(self):
        return len(self.features)
