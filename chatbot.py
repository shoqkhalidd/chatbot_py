import tkinter
from tkinter import *
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import random
import numpy as np
import logging
from silence_tensorflow import silence_tensorflow

silence_tensorflow()


intents = json.loads(open('intents.json').read())
model = load_model('chatbot.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


def bow(sentence):
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
	bag = [0]*len(words)
	for s in sentence_words:
		for i, w in enumerate(words):
			if w == s:
				bag[i]=1
	return (np.array(bag))

def predict_class(sentence):
	sentence_bag = bow(sentence)
	res = model.predict(np.array([sentence_bag]))[0]
	ERROR_THRESHOLD = 0.25
	results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
	#sort by probablity
	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append({'intent':classes[r[0]], 'probablity':str(r[1])})
	return return_list

def getResponse(ints):
	tag = ints[0]['intent']
	list_of_intents = intents['intents']
	for i in list_of_intents:
		if(i['tag']==tag):
			resp=i['responses']
			result=random.choice(resp)
			break

	return result

def chatbot_response(msg):
	ints = predict_class(msg)
	res = getResponse(ints)
	return res

def isBlank (myString):
    if myString and myString.strip():
        #myString is not None AND myString is not empty or blank
        return False
    #myString is None OR myString is empty or blank
    return True

print("Hey i'm a customer service for Fighting Robot controllers")
while TRUE:
	try:
		msg=input("You: ")
		if isBlank(msg):
			print("Sorry, could you please repeat that?")
		else:
			res=chatbot_response(msg)
			print("Bot: " + res)
	except(KeyboardInterrupt,EOFError,SystemExit):
		break;