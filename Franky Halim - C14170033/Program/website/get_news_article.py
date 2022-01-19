from argparse import ArgumentParser
from newspaper import Article
import requests
parser = ArgumentParser()
parser.add_argument('--source',type=str, default='', help='source of news file')
args = parser.parse_args()
try:
	assert(requests.get(args.source))
	news_article = Article(args.source) 
	news_article.download() 
	news_article.parse()
	original_text = news_article.text.split('\n\n')
	print(' '.join(original_text))
except:
	print('Failed to get news article')


