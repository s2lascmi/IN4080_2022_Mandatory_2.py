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




"""Exercise 1a"""
tokens = word_tokenize(raw)


# print(len(tokens)) #1093
# print(len(set(tokens))) #448




"""Exercise 1b"""

words = []
numbers = []
puncts = []
others = []

for t in tokens:
    if t.isalpha():
        words.append(t)
    elif t.isnumeric():
        numbers.append(t)
    elif t in ".,:;-_?!()" or t in ['"', "''", "``"]:
        puncts.append(t)
    else:
        others.append(t)
# print(len(words)) #961
# print(len(numbers)) #4
# print(len(puncts)) #110
# print(len(others)) #18


"""Exercise 1c"""

print(set(puncts))
print(set(others))
print(set(numbers))
print(set(words))


words = []
numbers = []
puncts = []
others = []

for t in tokens:
    if t.isalpha():
        words.append(t)
    elif re.search("\w-\w", t):
        words.append(t)
    elif t.isnumeric():
        numbers.append(t)
    elif t in ".,:;-_?!()" or t in ['"', "''", "``"]:
        puncts.append(t)
    else:
        others.append(t)


print(len(words)) #965
print(len(numbers)) #4
print(len(puncts)) #110
print(len(others)) #14

