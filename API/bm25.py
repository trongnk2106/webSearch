# Bước 1: Tiền xử lý văn bản
import os
import math
from tqdm import tqdm
import numpy as np
import json
import string
import glob
import nltk # Thư viện xử lý ngôn ngữ tự nhiên
from nltk.corpus import stopwords # Tập hợp các stop words
from nltk.stem import PorterStemmer # Bộ stemmer
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords') # Tải về các stop words
# Khởi tạo bộ stemmer và tập hợp các stop words
stemmer = PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

stop_words = set(stopwords.words('english'))




def preprocessing_stem(tokens):
  tokens = nltk.word_tokenize(tokens)
  tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
  tokens = [stemmer.stem(token) for token in tokens]
  return tokens

def loadweight(path_weight):
  with open(path_weight, 'r') as f:
    content = f.readlines()

  content = [json.loads(x.strip().replace("'", '"')) for x in content]
  
  return content

indexing_dataset = loadweight('./indexing_bm25.txt') 



def eval(querry, indexing_dataset, top_n = 0, corpus = 1400):
  scores = []
  for i in range(len(corpus)):
    score = 0
    for term in querry:
      if term in indexing_dataset[i]:
        # print(indexing_dataset[i])
        score += indexing_dataset[i][term]
    scores.append(score)
  avl = 1.5 * (sum(scores) / len(scores))
  top_search = []
  res_docidx = []
  res_score = []

  if top_n == 0:

    for sc in scores:
      if sc > avl:
        res_docidx.append(scores.index(sc) + 1)
        res_score.append(sc)
  else:
    for idx in range(top_n):
      max_index = scores.index(max(scores))
      res_score.append(scores[max_index])
      res_docidx.append(str(max_index + 1))
      scores[max_index] = -1000
 
  return res_docidx,  res_score


async def search(querry):
    
    querry = preprocessing_stem(querry)
    
    res_docidx, res_score = eval(querry,indexing_dataset,top_n = 20)
    return res_docidx
    




