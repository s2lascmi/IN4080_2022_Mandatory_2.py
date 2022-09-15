import numpy
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
import spacy
import re

with open("in4080_2022_ex2/crisis.txt", 'r') as f:
    raw = f.read()
#print(raw)

"""Exercise 2a"""

sents = nltk.sent_tokenize(raw)
#print(len(sents))

"""Exercise 2b"""

tokenized = [nltk.word_tokenize(s) for s in sents]
print(tokenized)


"""Exercise 2c"""
tokens = [[w for w in s] for s in tokenized]
#print("\n: ", tokens)
print(sum(len(s)))
#??????????????????????????????????????????????????????????????

cleaned = [[w for w in s if not(w in ".,:;-_?!()" or w in ['"', "''", "``"])] for s in tokens]
number_words = sum(len(s) for s in cleaned)


