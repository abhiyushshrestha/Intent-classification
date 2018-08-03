#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:12:09 2018

@author: abhiyush
"""
from flask import Flask, request, jsonify
import pickle
import nltk
from nltk.stem import WordNetLemmatizer

open_class_words = open('/home/abhiyush/mPercept/Natural Language Processing/Intent detection/models/class_words.pickle', 'rb')
class_words = pickle.load(open_class_words)
open_class_words.close()

class_words
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def index():
    return("Welcome to intent classification API")

@app.route('/predict', methods = ['GET','POST'])
def predict():
    sentence = request.json['sentence']
    predictions = classify(sentence)
    return jsonify({'predictions' : predictions})


def calculate_jaccard_similarity(sentence, class_name):
    test_word = []
    for word in nltk.word_tokenize(sentence):
        test_word.append(lemmatizer.lemmatize(word).lower())
    a = set(test_word) 
    b = set(class_words[class_name])
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def classify(sentence):
    high_jaccard_score = 0
    high_class = None
    score = 0
    
    for class_name in class_words.keys():
        score = calculate_jaccard_similarity(sentence, class_name)
        
        if(score > high_jaccard_score):
            high_jaccard_score = score
            high_class = class_name
            
    return high_class

if __name__ == '__main__':
    app.run(debug=True)
