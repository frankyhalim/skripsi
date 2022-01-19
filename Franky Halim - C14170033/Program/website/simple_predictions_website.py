import os
import sys
import requests
import torch
from argparse import ArgumentParser
from newspaper import Article

sys.path.insert(0, os.path.abspath("./src"))
from extractive import ExtractiveSummarizer

parser = ArgumentParser(description='Indonesian News Summarization')
parser.add_argument('--model', type=str, default='E:/models/indolem-indobert-base-uncased_use-token-type-ids_2_1_3e-05_0.5_8_1.bin',help='trained model')
parser.add_argument('--source',type=str, default='./kumpulan_berita/berita_covid_omicron_cnbc_2.txt', help='source of news')
parser.add_argument('--percentages',type=float, default='80.0',help='percentages of summary sentences')
parser.add_argument('--save_dir', type=str, default='./kumpulan_berita', help='directory of saved news article')

args = parser.parse_args()

checkpoint = torch.load(args.model, map_location=torch.device('cpu'))
state_dict = checkpoint['model_state_dict']
model = ExtractiveSummarizer(checkpoint['hyperparameters'])
model.load_state_dict(checkpoint['model_state_dict'])

# Check source
try:
    #Get news article
    assert(requests.get(args.source))
    news_article = Article(args.source)
    news_article.download()
    news_article.parse()
    temp = news_article.text.split('\n\n')
    contents = [sent+"\n" for sent in temp]
    file_name = news_article.title.replace(' ','-') + '.txt'
    source = 'url/' + file_name
    with open(os.path.join(args.save_dir, file_name), 'w', encoding='utf-8') as f:
        f.write('\n'.join(contents))
except:	
    # Open text file
    with open(args.source) as f:
        contents = f.readlines()
    source = 'file/' + os.path.basename(args.source)
 
# Convert percentages to number of sentences (based on number of sentences in news text)
num_sentences = int((args.percentages*0.01) * len(contents))

# Check if sentences less than 1, then set number of sentence to min. 1
if num_sentences < 1:
	num_sentences = 1
    
# Predict
output = model.predict(contents, source, num_sentences)
print(output)


# =============================================================================
# # If the number of sentences equal to length of content then show original news
# import json
# if num_sentences != len(contents): 
# 	output = model.predict(contents, num_sentences)
# else:
# 	output = json.dumps(contents)
# 
# print(output)
# =============================================================================
