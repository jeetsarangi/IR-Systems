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
import pandas as pd
Stopwords = set(stopwords.words('english'))  
ps = PorterStemmer()

#Importing all pre-processed files that are required for model 
temp=open('file_idx.pkl','rb')
file_index=pickle.load(temp)

temp=open('posting_list.pkl',"rb")
posting_lists=pickle.load(temp)

temp=open('unique_words.pkl','rb')
unique_words=pickle.load(temp)

#This function takes string as input and removes all non ascii and special characters
def filter_special_characters(text):
    regex = re.compile('[^a-zA-Z0-9\s]')
    text_returned = regex.sub('',text)
    return text_returned

'''
This function takes as input list of words and then removes or and not and if and is not present in between any two words then
adds it
'''
def append_and(words):
    
    res = []
    total_words = len(words)
    
    if total_words > 0:
        res.append(words[0])
    for i in range(1,total_words):
        #if and is not present then append one else append the current word and if word is or then also append and instead.
        if words[i] not in ["and","or"]:
            if res[-1] not in ["and","or"]:
                res.append("and")
                res.append(words[i])
            else:
                res.append(words[i])
        elif res[-1] not in ["and","or"]:
            res.append(words[i])
            
    return res

#This function takes as input list of words and separates the connectors and meaningfull words
def find_connectors(words):
    connector = []
    query_word = []
    
    for word in words:
        word = word.lower()
        if word in ["or","and"]:
            connector.append(word)
        else:
            query_word.append(word)
    return connector,query_word

#This function takes as input query words set without any connectors and returns word vector matrix representation of files
def make_words_vector(query_words):   
    n=len(file_index)

    temp_vector=[]
    word_vector_matrix=[]

    for word in query_words:
        temp_vector=[0]*n

        if word in unique_words:
            for x in posting_lists[word].keys():
                temp_vector[x]=1

        word_vector_matrix.append(temp_vector)
    return word_vector_matrix

# this function performs the and operation on all the word vector bitmaps and return there resultant vector
def perform_operation(connector,word_vector_matrix):
    for word in connector:
        v1=word_vector_matrix[0]
        v2=word_vector_matrix[1]

        if word == "and":
            temp=[b1&b2 for b1,b2 in zip(v1,v2)]

            word_vector_matrix.pop(0)
            word_vector_matrix.pop(0)

            word_vector_matrix.insert(0,temp)
        else:
            temp=[b1|b2 for b1,b2 in zip(v1,v2)]

            word_vector_matrix.pop(0)
            word_vector_matrix.pop(0)

            word_vector_matrix.insert(0,temp)
            
    return word_vector_matrix[0]

# this fuction is used to convert final files vector to files list
def get_files(files_vector):
    idx=0
    files=[]
    for i in files_vector:
        if i==1:
            files.append(file_index[idx])
        idx+=1
    return files

#This is the main function that perform boolean retrieval given an query and returns the result files
def BRM(query):   
    
    #using nltk library we word tokenize but before that remove special characters and digits.
    text = filter_special_characters(query)
    text = re.sub(re.compile('\d'),'',text)
    words = word_tokenize(text)
    
    #make all words lower and perform stemming then remove stopwords
    words = [word.lower() for word in words] 
    words = [ps.stem(word) for word in words]
    words = [word for word in words if word not in Stopwords]

    #perform retrieval using the above functions
    words = append_and(words)
    connector,query_words = find_connectors(words)
    words_vectors = make_words_vector(query_words)
    files_vector = perform_operation(connector,words_vectors)
    files = get_files(files_vector)
    return files

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
    solution[key] = BRM(query)

solution

#The below snippet converts the solution dictionary to Qrels format list of list then loads it to the file.
Qrels = []
for i in solution:
    count = 5
    for doc in solution[i]:
        Qrels.append([i,1,doc,1])
        count -= 1
        if(count == 0):
            break
df = pd.DataFrame(Qrels)
df.to_csv("BRM_output.txt",header = False,sep = ",",index = False)
