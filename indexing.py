# Giả sử bạn có 4 tài liệu văn bản như sau:
docs = [
    "This is a document about computer graphics.",
    "This is another document about machine learning.",
    "This document is about natural language processing.",
    "This is the last document about computer vision."
]

# Bước 1: Tiền xử lý văn bản
import nltk # Thư viện xử lý ngôn ngữ tự nhiên
from nltk.corpus import stopwords # Tập hợp các stop words
from nltk.stem import PorterStemmer # Bộ stemmer
nltk.download('stopwords') # Tải về các stop words

# Khởi tạo bộ stemmer và tập hợp các stop words
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Hàm tiền xử lý văn bản
def preprocess(text):
    # Chuyển văn bản thành chữ thường
    text = text.lower()
    # Tách văn bản thành các từ
    tokens = nltk.word_tokenize(text)
    # Loại bỏ các stop words và các ký tự không phải chữ
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    # Stemming các từ
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

# Tiền xử lý các tài liệu và lưu vào một danh sách mới
processed_docs = []
for doc in docs:
    processed_docs.append(preprocess(doc))

# In ra kết quả tiền xử lý
print(processed_docs)
# [['document', 'comput', 'graphic'], ['document', 'machin', 'learn'], ['document', 'natur', 'languag', 'process'], ['last', 'document', 'comput', 'vision']]

# Bước 2: Xây dựng inverted index

# Khởi tạo một từ điển rỗng để lưu trữ inverted index
inverted_index = {}

# Duyệt qua các tài liệu đã được tiền xử lý
for doc_id, doc in enumerate(processed_docs):
    # Duyệt qua các từ trong mỗi tài liệu
    for term in doc:
        # Nếu từ chưa có trong inverted index, thêm vào với posting list là danh sách chứa doc_id hiện tại
        if term not in inverted_index:
            inverted_index[term] = [doc_id]
        # Nếu từ đã có trong inverted index, kiểm tra xem doc_id hiện tại đã có trong posting list chưa
        else:
            posting_list = inverted_index[term]
            # Nếu chưa có, thêm vào posting list
            if doc_id not in posting_list:
                posting_list.append(doc_id)

# In ra kết quả inverted index
print(inverted_index)
# {'document': [0, 1, 2, 3], 'comput': [0, 3], 'graphic': [0], 'machin': [1], 'learn': [1], 'natur': [2], 'languag': [2], 'process': [2], 'last': [3], 'vision': [3]}

# Bước 3: Tính toán trọng số cho các từ

# Sử dụng công thức tf-idf để tính trọng số cho các từ

import math # Thư viện toán học

# Hàm tính inverse document frequency của một từ
def idf(term):
    # Số lượng tài liệu trong ngữ liệu
    N = len(docs)
    # Số lượng tài liệu chứa từ đó (độ dài của posting list)
    df = len(inverted_index[term])
    # Công thức idf theo cơ sở logarit tự nhiên
    return math.log(N / df)

# Khởi tạo một ma trận rỗng để lưu trọng số của các từ trong các tài liệu
# Kích thước của ma trận là số tài liệu x số từ
weights = [[0 for term in inverted_index] for doc in docs]

# Duyệt qua các từ và các posting list trong inverted index
for term_id, (term, posting_list) in enumerate(inverted_index.items()):
    # Tính idf của từ đó
    term_idf = idf(term)
    # Duyệt qua các doc_id trong posting list
    for doc_id in posting_list:
        # Tính term frequency của từ đó trong tài liệu đó
        term_tf = processed_docs[doc_id].count(term)
        # Tính tf-idf của từ đó trong tài liệu đó
        term_tf_idf = term_tf * term_idf
        # Lưu vào ma trận trọng số
        weights[doc_id][term_id] = term_tf_idf

# In ra kết quả ma trận trọng số
print(weights)
# [[0.28768207245178085, 0.28768207245178085, 0.6931471805599453, 0, 0, 0, 0, 0, 0, 0], [0.28768207245178085, 0, 0, 0.6931471805599453, 0.6931471805599453, 0, 0, 0, 0, 0], [0.28768207245178085, 0, 0, 0, 0, 0.6931471805599453, 0.6931471805599453, 0.6931471805599453, 0, 0], [0.28768207245178085, 0.28768207245178085, 0, 0, 0, 0, 0, 0, 1.3862943611198906, 1.3862943611198906]]

# Bước 4: Lưu trữ inverted index và ma trận trọng số

# Sử dụng thư viện pickle để lưu trữ các cấu trúc dữ liệu vào file

import pickle # Thư viện pickle

# Mở một file để ghi dữ liệu
with open("index.pkl", "wb") as f:
    # Ghi inverted index vào file
    pickle.dump(inverted_index, f)
    # Ghi ma trận trọng số vào file
    pickle.dump(weights, f)

# Đóng file
f.close()
