import numpy
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
import spacy
import re
from nltk.corpus import brown
nltk.download('brown')
nltk.download('universal_tagset')

"""Exercise 4a"""

uni_tag_words = [x for x in brown.tagged_words(tagset="universal")]
#print("uni tag words: " , uni_tag_words)
uni_tag_freq = nltk.FreqDist([t for w,t in uni_tag_words])
#print("\n", "uni tag freq: ", uni_tag_freq)
# for t in uni_tag_freq:
#     print("{:7}{:10"})print(....)



"""Exercise 4b"""
uni_distr = nltk.ConditionalFreqDist(uni_tag_words)
number_of_tags = {w: len(uni_distr[w]) for w in uni_distr}
freq_freqs = nltk.FreqDist([number_of_tags[w] for w in number_of_tags])
for numb in sorted(freq_freqs):
    print("{:10} words have {} different tags".format(freq_freqs[numb], numb))


n = max(freq_freqs)
for w in number_of_tags:
    if number_of_tags[w] == n:
        print(w)