#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 18:08:02 2018

@author: abhiyush
"""
import numpy as np
import pandas as pd
import nltk

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix


training_data = []

training_data.append({"class":"greeting", "sentence":"how are you?"})
training_data.append({"class":"greeting", "sentence":"How is your day?"})
training_data.append({"class":"greeting", "sentence":"Good day."})
training_data.append({"class":"greeting", "sentence":"How it is going today?"})
training_data.append({"class":"greeting", "sentence":"Hey,"})
training_data.append({"class":"greeting", "sentence":"Hey man"})
training_data.append({"class":"greeting", "sentence":"Hi"})
training_data.append({"class":"greeting", "sentence":"Hello"})
training_data.append({"class":"greeting", "sentence":"How are you doing"})
training_data.append({"class":"greeting", "sentence":"What's up"})
training_data.append({"class":"greeting", "sentence":"What's new"})
training_data.append({"class":"greeting", "sentence":"What's going on?"})
training_data.append({"class":"greeting", "sentence":"How's everything?"})
training_data.append({"class":"greeting", "sentence":"how are things?"})
training_data.append({"class":"greeting", "sentence":"How's life"})
training_data.append({"class":"greeting", "sentence":"how was your day?"})
training_data.append({"class":"greeting", "sentence":"how's your day going?"})
training_data.append({"class":"greeting", "sentence":"Good to see you"})
training_data.append({"class":"greeting", "sentence":"Nice to see you"})
training_data.append({"class":"greeting", "sentence":"Long time no see"})
training_data.append({"class":"greeting", "sentence":"Good morning"})
training_data.append({"class":"greeting", "sentence":"Good afternoon"})
training_data.append({"class":"greeting", "sentence":"Good evening"})
training_data.append({"class":"greeting", "sentence":"Nice to meet you"})
training_data.append({"class":"greeting", "sentence":"Pleased to meet you"})
training_data.append({"class":"greeting", "sentence":"How do you do?"})
training_data.append({"class":"greeting", "sentence":"Yo!"})
training_data.append({"class":"greeting", "sentence":"Are you okay?"})
training_data.append({"class":"greeting", "sentence":"You alright?"})
training_data.append({"class":"greeting", "sentence":"Alright mate?"})
training_data.append({"class":"greeting", "sentence":"Whazzup?"})
training_data.append({"class":"greeting", "sentence":"Sup?"})
training_data.append({"class":"greeting", "sentence":"Howdy!"})
training_data.append({"class":"greeting", "sentence":"G'day mate!"})



training_data.append({"class":"goodbye", "sentence":"Have a nice day."})
training_data.append({"class":"goodbye", "sentence":"see you later."})
training_data.append({"class":"goodbye", "sentence":"have a nice day"})
training_data.append({"class":"goodbye", "sentence":"talk to you soon"})
training_data.append({"class":"goodbye", "sentence":"Bye"})
training_data.append({"class":"goodbye", "sentence":"Bye bye!"})
training_data.append({"class":"goodbye", "sentence":"See you later"})
training_data.append({"class":"goodbye", "sentence":"talk to you later"})
training_data.append({"class":"goodbye", "sentence":"See you soon"})
training_data.append({"class":"goodbye", "sentence":"I must be going"})
training_data.append({"class":"goodbye", "sentence":"I have got to get going"})
training_data.append({"class":"goodbye", "sentence":"I am off"})
training_data.append({"class":"goodbye", "sentence":"Take it easy"})
training_data.append({"class":"goodbye", "sentence":"Goodbye"})
training_data.append({"class":"goodbye", "sentence":"Have a good day"})
training_data.append({"class":"goodbye", "sentence":"I look forward to our next meeting"})
training_data.append({"class":"goodbye", "sentence":"Take care"})
training_data.append({"class":"goodbye", "sentence":"It was nice to see you again"})
training_data.append({"class":"goodbye", "sentence":"It was nice seeing you"})
training_data.append({"class":"goodbye", "sentence":"Good night"})
training_data.append({"class":"goodbye", "sentence":"Later"})
training_data.append({"class":"goodbye", "sentence":"Catch you later"})
training_data.append({"class":"goodbye", "sentence":"Peace out"})
training_data.append({"class":"goodbye", "sentence":"I am out of here"})
training_data.append({"class":"goodbye", "sentence":"have a sweet dreams"})
training_data.append({"class":"goodbye", "sentence":"I gotta jet"})
training_data.append({"class":"goodbye", "sentence":"I gotta take off"})
training_data.append({"class":"goodbye", "sentence":"I gotta hit the road"})
training_data.append({"class":"goodbye", "sentence":"I gotta head out"})



training_data.append({"class":"order", "sentence":"make me a sandwich"})
training_data.append({"class":"order", "sentence":"can you make a sandwich?"})
training_data.append({"class":"order", "sentence":"having a sandwich today?"})
training_data.append({"class":"order", "sentence":"what's for lunch?"})
training_data.append({"class":"order", "sentence":"What's the menu for today"})
training_data.append({"class":"order", "sentence":"Would you like cheese with that"})
training_data.append({"class":"order", "sentence":"Would you like this to go"})
training_data.append({"class":"order", "sentence":"Is that all you will be ordering"})
training_data.append({"class":"order", "sentence":"Is that all for you today?"})
training_data.append({"class":"order", "sentence":"Can I take you order?"})
training_data.append({"class":"order", "sentence":"Are you ready to order?"})
training_data.append({"class":"order", "sentence":"Can we have a bill, please?"})



print ("%s sentences of training data" % len(training_data))

stemmer = PorterStemmer()
corpus_words = {}
class_words = {}

#Taking a value with key class from the training_data and converting it into set so as
# to remove duplicate data and again converting it into list
classes = list(set([a['class'] for a in training_data]))

for c in classes:
    class_words[c] = []

#Loop through each data in our training data
for data in training_data:
    #tokenize each words in a sentence
    for word in nltk.word_tokenize(data['sentence']):
        print(word)
        #ignoring a words which are listed below
        if word not in ["?", ".", "'s", ",","!", "'"]:
            #stemming and converting to lower case for each word 
            stemmed_word = stemmer.stem(word.lower())
            
            #if stemmed_word corpus_words ma chaina bhane 1 halne and if cha bhanne
            # count increase gardai jane
            if stemmed_word not in corpus_words:
                corpus_words[stemmed_word] = 1
            else:
                corpus_words[stemmed_word] += 1
                
            class_words[data["class"]].extend([stemmed_word])

print("Corpus words and counts:\n", corpus_words)
print("Class words:\n", class_words)

#calculating the score of the given class
def calculate_class_score(sentence, class_name, show_details = True):
    score = 0
    #tokenizing each words in the sentence
    for word in nltk.word_tokenize(sentence):
        #checking if the stemmed word is in any of the class
        if stemmer.stem(word.lower()) in class_words[class_name]:
            #treating each word with the same weight
            score += 1  
            
            if show_details:
                print("Match word: ", stemmer.stem(word.lower()))
                
    return score

sentence = "good day for us to have lunch"

for c in class_words.keys():
    print("Class: %s , Score: %s \n" % (c, calculate_class_score(sentence, c)))


    
#calculating a score for a given class taking into account word commonality

def calculate_class_score_commonality(sentence, class_name, show_details = True):
    score = 0
    #tokenizing each words in the sentences
    for word in nltk.word_tokenize(sentence):
        if stemmer.stem(word.lower()) in class_words[class_name]:
            # treating each word with the relative weight
            score += (1/corpus_words[stemmer.stem(word.lower())])

            if show_details:
                print("Match word: ", stemmer.stem(word.lower()))
    return score

#return a class with highest score for sentence
    
def classify(sentence):
    high_class = None
    high_score = 0
    
    #loop through our class
    for class_name in class_words.keys():
        score = calculate_class_score_commonality(sentence, class_name, show_details = False)
        
        #keep track of the highest score
        if (score > high_score):
            high_score = score
            high_class = class_name
    
    return high_class #, high_score


test_data_set = ["take care",
"The day is beautiful today",
"Who are you?",
"Hi",
"Whats up?",
"How do you do?",
"talk to you later",
"bye bye!!!",
"Sweet dreams",
"I need to go",
"Could you please tell me menu for today",
"Our order should be ready by now, right",
"all good?",   
"take care",
"How is your life",
"Hey dude",
"whatzz up bro!!!", "how is your day going", 
"Is everything good",
"I hope you are fine",
"How was your day",
"Nice day",
"Have a perfect day"]
classify("take care")
classify("The day is beautiful today")
classify("Who are you?")
classify("Hi")
classify("Whats up?")
classify("How do you do?")
classify("talk to you later")
classify("bye bye!!!")
classify("Sweet dreams")
classify("I need to go")
classify("Could you please tell me menu for today")
classify("Our order should be ready by now, right")
classify("all good?")   
classify("take care")
classify("How is your life")
classify("Hey dude")
classify("whatzz up bro!!!")

len(test_data_set)


test_data_sets = pd.read_csv("/home/abhiyush/mPercept/Natural Language Processing/Intent detection/datasets/test_intent_datasets.csv")
test_data_sets

#converting categorical to numeric
numeric_mapper = {
        "class" : {"greeting" : 1, "goodbye" : 2, "order" : 3}
        }

test = test_data_sets.replace(numeric_mapper)
test_class = test['class']
type(test_class)

pred = []

for sen in test['sentence']:
    y = classify(sen)
    pred.append(y)

pred_class = pd.DataFrame(pred, columns = ['class'])

pred_class = pred_class.replace(numeric_mapper)

cm = confusion_matrix(test_class, pred_class)
cm


















]
