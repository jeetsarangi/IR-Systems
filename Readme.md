#                                  CS657 Assignment 1
## Author
Jeet Sarangi - 21111032 - jeets21@iitk.ac.in <br>


## Directory Structure:
#### IR-Systems
'BM25.py' : Code for BM25 based prediction.
'BRM.py' : Code for Boolean Retrieval Model based prediction.
'preprocess.py' : Code for preprocessing the data dump.
'TFIDF.py' : Code for TFIDF Model based prediction. 
'Intermediate result files after preprocess' : doc_len.pkl,document_freq.pkl,document_vector_mag.pkl,file_idx.pkl,posting_list.pkl,unique_words.pkl.
#### Qrels:
'query.txt' : queries for the model.
'Question3_Qrels.txt' : Grount truth results for all the queries in ranked order.
## Link to Dataset:
Link to Dataset: https://www.cse.iitk.ac.in/users/arnabb/ir/english/

## Steps to run:
#### 1:
##### Download and dump the Dataset folder in the same directory and update the link in code.<br>
##### Dump the query file in the same folder.<br>
##### Run the 'mid.sh' script to obtain the results in the Qrels format.
#### 2:
##### Or else zip IR-Systems and Qrels then load the datafile and the query file and run the make to obtain the results.

