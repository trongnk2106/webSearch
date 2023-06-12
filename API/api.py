from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import re
from collections import Counter
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
shortword = re.compile(r'\W*\b\w{1,2}\b')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords') # Tải về các stop words
# Khởi tạo bộ stemmer và tập hợp các stop words
stemmer = PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()

stop_words = set(stopwords.words('english'))


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Danh sách các nguồn được phép gửi yêu cầu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




class BM25():
    
    def __init__(self):
        pass
        
    def loadweight_bm25(path_weight):
        with open(path_weight, 'r') as f:
            content = f.readlines()

        content = [json.loads(x.strip().replace("'", '"')) for x in content]
        
        return content

    indexing_dataset = loadweight_bm25('./indexing_bm25.txt') 
    
    def preprocessing_stem(self, tokens):
        tokens = nltk.word_tokenize(tokens)
        tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
        tokens = [stemmer.stem(token) for token in tokens]
        return tokens
    

    def eval(self, querry, indexing_dataset = indexing_dataset, top_n = 0, corpus = 1400):
        scores = []
        for i in range(corpus):
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
        
        document = []
        for i in res_docidx:
            with open('./Cranfield/' + i + '.txt', 'r') as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            content.append(f' \t -------> file {i}.txt')
            
            document.append(content)
            
        
        return res_docidx,  res_score, document
    
class VectorSpace():
    def __init__(self):
        pass
    
    def tokenize_process(self, data):
        lines = data.lower()
        lines = re.sub(r"[^a-zA-Z0-9]", " ", lines)
        tokens = lines.split()
        clean_tokens = [word for word in tokens if word not in stop_words]
        stem_tokens = [stemmer.stem(word) for word in clean_tokens]
     
        clean_stem_tokens = [word for word in stem_tokens if word not in stop_words]

        clean_stem_tokens = ' '.join(map(str,  clean_stem_tokens))
        clean_stem_tokens = shortword.sub('', clean_stem_tokens)
        return clean_stem_tokens

    
    def load_vocab(path_vocab):
        with open(path_vocab, 'r') as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        return content
    
    vocab = load_vocab('./vocab_vector.txt')
    
    def load_df(path_df):
        with open(path_df, 'r') as f:
            DF= json.load(f)
            
        return DF
    
    DF = load_df('./DF.json')
    
    
    
    
    def loadweight_vector(path_weight):
        return np.load(path_weight)
    
    D = loadweight_vector('./indexing_vector.npy')
    
    no_of_docs = len(os.listdir('./Cranfield'))
    
    def gen_vector(self, tokens, vocab = vocab, DF = DF, no_of_docs = no_of_docs):
        Q = np.zeros((len(vocab)))
        counter = Counter(tokens)
        words_count = len(tokens)
        

        for token in np.unique(tokens):
            tf = counter[token]/words_count
            df = DF[token] if token in vocab else 0
            idf = math.log((no_of_docs+1)/(df+1))
            try:
                ind = vocab.index(token)
                Q[ind] = tf*idf
            except:
                pass
        return Q
    
    def cosine_sim(self, x, y):
        if (np.linalg.norm(x)*np.linalg.norm(y)) == 0:
            return 0
        cos_sim = np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

        return cos_sim
    
    def cosine_similarity(self,k, query, D = D):
        tokens = query.split()
        d_cosines = []
        query_vector = self.gen_vector(tokens)
        # print('vector query: ', query_vector)
        for d in D:
            # print('in for 1: ', d)
            d_cosines.append(self.cosine_sim(query_vector, d))
            # print('d cosines: ', d_cosines)
        if k == 0:
            out = np.array(d_cosines).argsort()[::-1]
        else:
            out = np.array(d_cosines).argsort()[-k:][::-1]

        return out
    
    def search(self, query, k = 0):
        query = self.tokenize_process(query)
        cs = self.cosine_similarity(k, query)
        cs = cs.tolist()
        cs = [str(x+1) for x in cs]
        document = []
        for i in cs:
            with open('./Cranfield/' + i + '.txt', 'r') as f:
                content = f.readlines()
            content = [x.strip() for x in content]
            content.append(f' \t -------> file {i}.txt')
            
            document.append(content)
        return document


BM25 = BM25()
VectorSpace = VectorSpace()

@app.get("/vec")
async def searchvec(q: str = Query(None, min_length=1), k : int = Query(0)):
   
    result = VectorSpace.search(q, k)
    # print(type(result))
    return {"result": result}

@app.get("/bm25")
async def search(q: str = Query(None, min_length=1), k: int = Query(0)):
    
    querry = BM25.preprocessing_stem(q)
    res_docidx, res_score , document= BM25.eval(querry,top_n = k)
    res = document
    print(type(res))
    return {"result": res}

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="localhost")
