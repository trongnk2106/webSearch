import math
from collections import Counter

def bm25(n, f, qf, r, N, dl, avdl):
    """
    :param n: số lượng văn bản chứa từ
    :param f: tần suất xuất hiện của từ trong văn bản
    :param qf: tần suất xuất hiện của từ trong câu truy vấn
    :param r: số lượng văn bản liên quan chứa từ
    :param N: tổng số lượng văn bản
    :param dl: độ dài của văn bản
    :param avdl: độ dài trung bình của văn bản
    """
    k1 = 1.5
    b = 0.75
    K = k1 * ((1 - b) + b * (float(dl) / float(avdl)))
    first = ((r + 0.5) / (N - r + 0.5)) / ((n - r + 0.5) / (N - n - r + 0.5))
    second = ((k1 + 1) * f) / (K + f)
    third = ((k1 + 1) * qf) / (K + qf)
    return math.log(first) * second * third

def get_bm25_weights(docs, query):
    """
    :param docs: danh sách các văn bản
    :param query: câu truy vấn
    """
    N = len(docs)
    avgdl = sum([len(doc) for doc in docs]) / N
    query = query.split()
    
    scores = []
    
    for doc in docs:
        score = 0
        doc_dict = Counter(doc)
        print(doc_dict)
        for word in query:
            if word in doc_dict:
                n = sum([1 if word in _doc else 0 for _doc in docs])
                print(n)
                f = doc_dict[word]
                qf = 1
                r = 0
                dl = len(doc)
                score += bm25(n=n, f=f, qf=qf, r=r, N=N, dl=dl, avdl=avgdl)
        scores.append(score)
    
    return scores

docs = [
    ['black', 'cat', 'white', 'cat'],
    ['cat', 'outer', 'space'],
    ['wag', 'dog']
]
query = 'black'

print(get_bm25_weights(docs, query))
