# -*- coding: utf-8 -*-
import os
import shutil
import json
from time import time
import torch
from datetime import timedelta
from argparse import ArgumentParser

parser = ArgumentParser(description='Data Preparation for Indonesian News Summarization')
parser.add_argument('--original_data_dir',type=str,default='./indosum_original', help='path to original dataset')
parser.add_argument('--save_dir', type=str, default='./prepared_data_basic/indo_news_', help='path to save prepared data') 
args = parser.parse_args()

def preprocessing(text):         
    # Delimiter of split sentence
    delimiter = '<q>'
        
    # Case folding (lowercase text)
    text = text.lower()
    
    # Clean excessive space, tab, and newline
    text = delimiter.join([' '.join(sent.split()) for sent in text.split(delimiter)])
    
    return text
    
def read(fname):
    data = []
    for line in open(fname, 'r').readlines():
        datum = json.loads(line)
        label=''
        source=''
        target= ''
        
        # Source paragraphs document and gold labels
        for idx in range(len(datum['paragraphs'])):
            for idy in range(len(datum['paragraphs'][idx])):
                for idz in range(len(datum['paragraphs'][idx][idy])):
                    source+=datum['paragraphs'][idx][idy][idz]+' '
                source+='<q>'
                label+=str(int(datum['gold_labels'][idx][idy])) + '<q>'
        
        # Gold summaries
        for idx in range(len(datum['summary'])):
            for idy in range(len(datum['summary'][idx])):
                target+=datum['summary'][idx][idy] + ' '
            target+='<q>'
            
        source = source[:-3]
        target = target[:-3]
        label = label[:-3]
        data.append((source, target, label))  
    return data   
      
def process(path, save_dir):
    dataset = []
    data = read(path)
    corpus_type = path.split('/')[-1].split('.')[0]
    fold = path.split('/')[-1].split('.')[1]
    start_time = time()
    for idx, datum in enumerate(data): 
        source, target, sent_labels = datum
        source = preprocessing(source) 
        target = '<q>'.join([' '.join(sent.split()) for sent in target.split('<q>')])
        b_data_dict = {"source": source,"labels": sent_labels, "target": target}
        dataset.append(b_data_dict)
        if (idx+1) % 500 == 0:
            end_time = time()
            print(f'{(idx+1)} data processed | {corpus_type}-{fold} | runtime: {end_time-start_time} seconds') 
            start_time = end_time
        
    if len(dataset) > 0:          
        pt_file = save_dir + "{:s}.indonews.bert.pt".format(corpus_type)
        torch.save(dataset, pt_file)

fold = [1,2,3,4,5]
start = time()
for i in fold:
    save_dir = args.save_dir + str(i) + '/' 
    print('Create ', save_dir)
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)  
    os.makedirs(save_dir) 
    process(args.original_data_dir + '/train.0'+str(i)+'.jsonl', save_dir)
    process(args.original_data_dir + '/dev.0'+str(i)+'.jsonl', save_dir)
    process(args.original_data_dir + '/test.0'+str(i)+'.jsonl', save_dir)
end = time()
elapsed_time = timedelta(seconds=end-start)
print(f'Elapsed time: {elapsed_time}')


