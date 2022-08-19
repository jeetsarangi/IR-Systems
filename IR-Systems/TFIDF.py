import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
import glob
import re
import os
import sys
import math
import pickle
import numpy as np
import pandas as pd
Stopwords = set(stopwords.words('english'))  
ps = PorterStemmer()

#Importing all pre-processed files that are required for model 
with open('file_idx.pkl','rb') as file:
    file_index = pickle.load(file)
    file.close()
    
with open('document_words.pkl','rb') as file:
    document_words=pickle.load(file)
    file.close()
    
with open('document_freq.pkl','rb') as file:
    document_freq=pickle.load(file)
    file.close()
    
with open('doc_len.pkl','rb') as file:
    doc_len=pickle.load(file)
    file.close()

file = open('unique_words.pkl','rb')
unique_words=pickle.load(file)
file.close()

file = open('posting_list.pkl','rb')
term_freq=pickle.load(file)
file.close()

file = open('document_vector_mag.pkl','rb')
doc_mag = pickle.load(file)
file.close()


#This function takes string as input and removes all non ascii and special characters
def filter_special_characters(text):
    regex = re.compile('[^a-zA-Z0-9\s]')
    text_returned = regex.sub('',text)
    return text_returned

#The below function as input a query and calculate tfidf scores for all documents and return top 5 most relevant documents in ranked order.
def TFIDF(query):    

    #using nltk library we word tokenize but before that remove special characters and digits.
    sol = []
    text = filter_special_characters(query)
    text = re.sub(re.compile('\d'),'',text)

    words = word_tokenize(text)
    
    #make all words lower and perform stemming then remove stopwords
    words = [word.lower() for word in words]
    words=[ps.stem(word) for word in words]
    words=[word for word in words if word not in Stopwords]
    words=[word for word in words if word in unique_words]

    #creating query vector
    query_vector_mag = 0
    query_vector = []
    N = len(file_index)

    for word in words:
        tfidf = words.count(word) * math.log(N/document_freq[word])
        query_vector.append(tfidf)
        query_vector_mag += (tfidf)**2

    query_vector_mag = math.sqrt(query_vector_mag)
    query_vector = np.array(query_vector)/query_vector_mag

    #calculate the score cosine similarity of documents with query vector using there tdidf values. 
    score = {}
    for file_idx in range(len(file_index)):
        document_vector = []

        # for each word in for particular file add zero if word not present else add tfidf score after calculating it document vector 
        # magnitude is already preprocessed.
        for word in words:
            if not term_freq[word].__contains__(file_idx):
                document_vector.append(0)
            else:
                tfidf = term_freq[word][file_idx] * math.log(N/document_freq[word])
                document_vector.append(tfidf)

        document_vector = np.array(document_vector)/doc_mag[file_idx]
        # calculate scores using cosine similarity. 
        score[file_idx]  = np.dot(document_vector,query_vector)

    #sort the documents according to there cosine similarity scores
    score=sorted(score.items(),key=lambda x:x[1],reverse=True)
    top_count = 5
    for i in range(top_count):
        sol.append(file_index[score[i][0]])
    return sol


#In below snippet we get the queries and the run Boolean retrieval on them and stores the result in a dictionary.
solution = {}
def dirback():
    m = os.getcwd()
    n = m.rfind("/")
    d = m[0: n+1]
    os.chdir(d)
    return None

dirback()

query_file_name=sys.argv[1]
query_file_name+=".txt"


query_file = open(query_file_name,"r")
queries = query_file.readlines()

for i in queries:
    temp = i.split("\t")
    key = temp[0]
    query = temp[1]

    
    solution[key] = TFIDF(query)

#The below snippet converts the solution dictionary to Qrels format list of list then loads it to the file.
Qrels = []
for i in solution:
    for doc in solution[i]:
        Qrels.append([i,1,doc,1])
df = pd.DataFrame(Qrels)
df.to_csv("TFIDF_output.txt",header = False,sep = ",",index = False)