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
Stopwords = set(stopwords.words('english'))  
ps = PorterStemmer()

#This function filters the non ascii and special characters from text(string) input
def filter_special_characters(text):
    regex = re.compile('[^a-zA-Z0-9\s]')
    text_returned = regex.sub('',text)
    return text_returned
#Below function finds all unique words and there frequencies in given list of words
def unique_words_with_freq(words):
    words_unique = []
    word_freq = {}
    for word in words:
        if word not in words_unique:
            words_unique.append(word)
    for word in words_unique:
        word_freq[word] = words.count(word)
    return word_freq

#Below snippet preprocess the documents and creates a dictionary of unique words and there occurences and also creates a list 
# of all the words in each documents.
document_words=[]
dict_global = {}
file_folder = 'english-corpora/*'
idx = 0
files_with_index = {}
ps = PorterStemmer()
for file in glob.glob(file_folder):
    print(file+" "+str(idx+1))
    fname = file
    file = open(file , "r",encoding='UTF-8')
    text = file.read()
    text = filter_special_characters(text)
    text = re.sub(re.compile('\d'),'',text)
    words = word_tokenize(text)
    words = [word for word in words if len(words)>1]
    words = [word.lower() for word in words]
    words = [ps.stem(word) for word in words]
    words = [word for word in words if word not in Stopwords]
    #stores documents words as it will be required later.
    document_words.append(words)
    #maintains global word dictionary so that all words and frequencies gets stored.
    dict_global.update(unique_words_with_freq(words))
    files_with_index[idx] = os.path.basename(fname)
    idx = idx + 1

#get all unique words from the dictionary keys.   
unique_words = set(dict_global.keys())

#By default creating two dictionary to store term frequency and document frequencies of words.
term_freq={}
document_freq={}
for i in unique_words:
    term_freq[i]={}
    document_freq[i]=0   

file_folder = 'english-corpora/*'
idx=0
tot_len=0
doc_len={}
for file in glob.glob(file_folder):
    file = open(file , "r",encoding="utf8")
    text = file.read()
    text = filter_special_characters(text)
    text = re.sub(re.compile('\d'),'',text)
    words = word_tokenize(text)
    words = [word.lower() for word in words]
    words=[ps.stem(word) for word in words]
    doc_len[idx] = len(words) #len of current doc
    tot_len = tot_len + len(words) #sum of lens of all the docs
    words = [word for word in words if word not in Stopwords]
    counter = Counter(words)
    for i in counter.keys():
        #document freq is based on all files and overall so add 1
        document_freq[i]=document_freq[i]+1
        #term frequency depends on the per documents hence store file_index as well.
        term_freq[i][idx]=counter[i]   
    print(idx)
    idx=idx+1

#pre-process and store document vector magnitude as it will be required in scored retrieval models.
document_vector_mag={}
idx=0
N = len(files_with_index)
for i in document_words:
    l2=0
    for word in set(i):
        l2 = l2 + (i.count(word)*math.log(N/document_freq[word]))**2
    document_vector_mag[idx]=(math.sqrt(l2))
    print(idx)
    idx += 1

#storing all pre-processed data in corresponding files.
file = open("file_idx.pkl", "wb")
pickle.dump(files_with_index, file)
file.close()
file = open("unique_words.pkl", "wb")
pickle.dump(unique_words, file)
file.close()
file = open("posting_list.pkl", "wb")
pickle.dump(term_freq, file)
file.close()
file = open("document_freq.pkl", "wb")
pickle.dump(document_freq, file)
file.close()
file = open("doc_len.pkl", "wb")
pickle.dump(doc_len, file)
file.close()
file = open("document_words.pkl", "wb")
pickle.dump(document_words, file)
file.close()
file = open("document_vector_mag.pkl", "wb")
pickle.dump(document_vector_mag, file)
file.close()
print(len(unique_words))