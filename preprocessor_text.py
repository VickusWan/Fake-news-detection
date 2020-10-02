# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 01:35:27 2020

@author: Victor
"""

import re
import nltk
from nltk.corpus import stopwords  
nltk.download('stopwords')
stop = set(stopwords.words('english')) 

def normalize(df_text):
    temp =[]
    
    #snow = nltk.stem.SnowballStemmer('english')
    for sentence in df_text:
        sentence = sentence.lower()                 # Converting to lowercase
        sentence = re.sub(r'<.*?>', ' ', sentence)        #Removing HTML tags
        sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)     #Removing Punctuations (? ! ' " #)
        sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence)        #Removing Punctuations (. , ) ( / \)
        
        
        words = stem(sentence)
# =============================================================================
#         words = []
#         for word in sentence.split():
#             if word not in stopwords.words('english'):
#                 words.append(snow.stem(word))
# =============================================================================
        
        #words = [snow.stem(word) for word in sentence.split() if word not in stopwords.words('english')]   # Stemming and removing stopwords

        temp.append(words)
    final_X = temp 
    
    txt = []
    for row in final_X:
        seq = ''
        for word in row:
            seq = seq + ' ' + word
        txt.append(seq)
    return txt

def stem(sentence):
    snow = nltk.stem.SnowballStemmer('english')
    words = []
    for word in sentence.split():
        if word not in stopwords.words('english'):
            words.append(snow.stem(word))
    return words
