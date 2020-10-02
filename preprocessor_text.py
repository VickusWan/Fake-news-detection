# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 01:35:27 2020

@author: Victor
"""

import re
import nltk
from nltk.corpus import stopwords  
from nltk.stem import PorterStemmer 


def preprocessor_text(df_text):
    temp =[]
    snow = nltk.stem.SnowballStemmer('english')
    for sentence in df_text:
        sentence = sentence.lower()                 # Converting to lowercase
        cleanr = re.compile('<.*?>')
        sentence = re.sub(cleanr, ' ', sentence)        #Removing HTML tags
        sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)
        sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence)        #Removing Punctuations
        
        words = [snow.stem(word) for word in sentence.split() if word not in stopwords.words('english')]   # Stemming and removing stopwords
        temp.append(words)
        
    final_X = temp 
    
    sent = []
    for row in final_X:
        sequ = ''
        for word in row:
            sequ = sequ + ' ' + word
        sent.append(sequ)
    return sent