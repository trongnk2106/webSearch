# Import math để tính logarit
import math

# Tạo một bộ dữ liệu đơn giản gồm 5 văn bản
data = ["This is a sentence about dogs",
        "This is another sentence about cats",
        "This is a third sentence about birds",
        "This is a fourth sentence about fish",
        "This is a fifth sentence about lions"]

# Tiền xử lý dữ liệu bằng cách tách từ và loại bỏ stop words
stop_words = ["a", "an", "the", "is", "about", "this"]
corpus = []
for doc in data:
  tokens = doc.lower().split()
  tokens = [token for token in tokens if token not in stop_words]
  corpus.append(tokens)
  
  
print(corpus)

# Khai báo các hằng số của BM25
k1 = 1.2
b = 0.75

# Tính độ dài trung bình của các văn bản
avgdl = sum([len(doc) for doc in corpus]) / len(corpus)

# Tính idf cho mỗi thuật ngữ trong tập hợp các văn bản
N = len(corpus) # Số lượng văn bản
idf = {} # Từ điển lưu trữ giá trị idf của mỗi thuật ngữ
for doc in corpus:
  for term in doc:
    if term not in idf:
      n = sum([1 for d in corpus if term in d]) # Số lượng văn bản chứa thuật ngữ
    #   if math.log((N - n + 0.5) / (n + 0.5)) <= 0:
    #     idf[term] = 0.02
    #   else:
          
    idf[term] = math.log(1 + ((N - n + 0.5) / (n + 0.5))) # Công thức tính idf

# Tính tf cho mỗi thuật ngữ trong mỗi văn bản
tf = [] # Danh sách lưu trữ giá trị tf của mỗi thuật ngữ trong mỗi văn bản
for doc in corpus:
  tf_doc = {} # Từ điển lưu trữ giá trị tf của mỗi thuật ngữ trong một văn bản
  for term in doc:
    if term not in tf_doc:
      frequency = doc.count(term) # Số lần xuất hiện của thuật ngữ trong văn bản
      tf_doc[term] = math.sqrt(frequency) # Công thức tính tf
  tf.append(tf_doc)

# Tính K cho mỗi văn bản
K = [] # Danh sách lưu trữ giá trị K của mỗi văn bản
for doc in corpus:
  dl = len(doc) # Độ dài của văn bản
  K.append(k1 * ((1 - b) + b * (float(dl) / float(avgdl)))) # Công thức tính K

# Tính BM25 cho mỗi thuật ngữ trong mỗi văn bản
BM25 = [] # Danh sách lưu trữ giá trị BM25 của mỗi thuật ngữ trong mỗi văn bản
for i in range(N):
  BM25_doc = {} # Từ điển lưu trữ giá trị BM25 của mỗi thuật ngữ trong một văn bản
  for term in corpus[i]:
    BM25_doc[term] = idf[term] * ((k1 + 1) * tf[i][term]) / (K[i] + tf[i][term]) # Công thức tính BM25
  BM25.append(BM25_doc)
  
print(BM25)

# Tạo một truy vấn đơn giản
query = "sentence about third dogs"

# Tiền xử lý truy vấn bằng cách tách từ và loại bỏ stop words
query_tokens = query.lower().split()
query_tokens = [token for token in query_tokens if token not in stop_words]

# Tính điểm BM25 cho truy vấn và lấy ra 3 văn bản có điểm cao nhất
scores = [] # Danh sách lưu trữ điểm BM25 của mỗi văn bản cho truy vấn
for i in range(N):
  score = 0 # Điểm BM25 của một văn bản cho truy vấn
  for term in query_tokens:
    if term in BM25[i]:
      score += BM25[i][term] # Cộng dồn điểm BM25 của các thuật ngữ trong truy vấn
  scores.append(score)
# print(scores)



top_n = [] # Danh sách lưu trữ 3 văn bản có điểm cao nhất
for i in range(3):
  max_index = scores.index(max(scores)) # Chỉ số của văn bản có điểm cao nhất
  top_n.append(data[max_index]) # Thêm văn bản có điểm cao nhất vào danh sách
  scores[max_index] = -1 # Đặt điểm của văn bản đã chọn thành -1 để tìm văn bản tiếp theo

# In kết quả
print("Query:", query)
print("Top 3 documents:", top_n)
