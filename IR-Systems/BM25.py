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

#calculating average documents length we have already stored pre-processed document lengths. 
total = 0
N=len(doc_len)
for i in range(N):
    total += doc_len[i]
Lavg=total/N
Lavg

#below are hyperparameters.
k=1.2
b=0.75

'''
“idf, computed as log(1 + (N — n + 0.5) / (n + 0.5)) from:”

“tf, computed as freq / (freq + k1 * (1 — b + b * dl / avgdl)) from:”
'''


#below function takes input a word and return its IDF in BM25 way
def IDF(word):
    df = document_freq[word]
    ans=math.log((N-df+0.5)/(df+0.5))
    return ans

#Below function takes input word and file index and return term frequency in BM25 way.

def TF(word,file_idx):
    tf = term_freq[word][file_idx]
    ans = ((k+1)*tf)/(tf+ k*(1-b+b*(doc_len[file_idx] / Lavg)))
    return ans
 
#Below function calculates the similarity scores for all documents given a particular query.
def calculate_score(query):
    #using nltk library we word tokenize but before that remove special characters and digits.
    query = filter_special_characters(query)
    query = re.sub(re.compile('\d'),'',query)
    words = word_tokenize(query)
    
    #make all words lower and perform stemming then remove stopwords
    words = [word.lower() for word in words]
    words=[ps.stem(word) for word in words]
    words=[word for word in words if word not in Stopwords]
    words=[word for word in words if word in unique_words]
    
    #calculate the score based on BM25 way tf-idf values sum. 
    score = {}
    for file_idx in range(len(document_words)):
        score[file_idx] = 0
        for word in words:
            temp = 0
            if file_idx in term_freq[word]:
                temp = TF(word,file_idx)*IDF(word)
            score[file_idx] += temp
    return score

# Below is the main function to retrieve the documents based on query.
def BM25(query):

    #return the resulting top 5 most relevant documents in res list.
    res = []
    score = calculate_score(query)
    score=sorted(score.items(),key=lambda item: item[1],reverse=True)
    # print(score)
    count = 5
    for i in score:
        if count == 0:
            break

        res.append(file_index[i[0]])
        count-=1
    return res

#In below snippet we get the queries and the run Boolean retrieval on them and stores the result in a dictionary.
solution = {}
# argumentList = sys.argv
# print (argumentList)
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
    solution[key] = BM25(query)

solution

#The below snippet converts the solution dictionary to Qrels format list of list then loads it to the file.
Qrels = []
for i in solution:
    for doc in solution[i]:
        Qrels.append([i,1,doc,1])
df = pd.DataFrame(Qrels)
df.to_csv("bm25_output.txt",header = False,sep = ",",index = False)

