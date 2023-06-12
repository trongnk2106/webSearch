# Import libraries
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Define documents and query
documents = ["The sky is blue and beautiful.",
             "Love this blue and beautiful sky!",
             "The quick brown fox jumps over the lazy dog.",
             "A king loves eating delicious brown food."]
query = "brown fox"

# Preprocess documents and query
vectorizer = TfidfVectorizer(stop_words="english", use_idf=False, binary=True)
# print('vectorize ' ,vectorizer)
X = vectorizer.fit_transform(documents) # document-term matrix
# print('data :', X)
q = vectorizer.transform([query]) # query vector

# Calculate probabilities
terms = vectorizer.get_feature_names_out() # list of terms in the vocabulary
# print(terms)
# print(q)
N_d = len(documents) # number of documents in the collection
print(X)
# N_t,d = X.sum(axis=0) # number of documents containing each term
# print(N_t, '\t', d)
# P_t,d = N_t,d / N_d # probability of each term in each document
# prob_table = pd.DataFrame(data=P_t,d.toarray(), index=["P(t|d)"], columns=terms)
# print(prob_table)0 