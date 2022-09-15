import nltk
import random
import numpy as np
import scipy as sp
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import statistics
from nltk.corpus import brown
from tabulate import tabulate
import re

""" Exercise 2 """
""" In this exercise, we will consider Zipf’s law, which is explained in exercise 23 in NLTK chapter 2, 
and more thoroughly in the Wikipedia article: Zipf's law, which you are advised to read. We will use 
the text Tom Sawyer. """

""" 2a """
""" First, you need to get hold of the text. You can download it from project Gutenberg as explained in 
section 1 in chapter 3 in the NLTK book. You find it here: https://www.gutenberg.org/files/74/74-0.txt """

from urllib import request
url = "https://www.gutenberg.org/files/74/74-0.txt"
response = request.urlopen(url)
raw = response.read().decode('utf8')



""" 2b """
""" Then you have to do some clean up. The downloaded text contains a preamble and a long appendix 
about the Gutenberg project and copyrights that should be removed. """

def clean_texts(text):
    """ Finds end of the preamble and beginning of the appendix, leaves only the text in between."""
    search_results = []
    preamble = re.finditer("1876.", text)
    appendix = re.finditer("part of their lives at present.", text)
    for item in preamble:
        search_results.append(item.end())
    for item in appendix:
        search_results.append(item.end())
    cleaned_text = text[search_results[0]:search_results[1]]
    return cleaned_text


only_text = clean_texts(raw)




""" 2c """
""" You can then extract the words. We are interested in the words used in the book and their distribution. 
We are, e.g. not interested in punctuation marks. Should you case fold the text? 
Explain the steps you take here and in point (b) above. """

def reduce_punctuation(text):
    no_punctuation = re.sub(r"\t", "", text)
    no_punctuation = re.sub(r"\.", "", no_punctuation)
    no_punctuation = re.sub(r";", "", no_punctuation)
    no_punctuation = re.sub(r":", "", no_punctuation)
    no_punctuation = re.sub(r",", "", no_punctuation)
    no_punctuation = re.sub(r"\?", "", no_punctuation)
    no_punctuation = re.sub(r"!", "", no_punctuation)
    no_punctuation = re.sub(r"\)", "", no_punctuation)
    no_punctuation = re.sub(r"\(", "", no_punctuation)
    no_punctuation = re.sub(r"\"", "", no_punctuation)
    no_punctuation = re.sub(r"“", "", no_punctuation)
    no_punctuation = re.sub(r"”", "", no_punctuation)
    no_punctuation = re.sub(r"—", " ", no_punctuation)
    no_punctuation = re.sub(r"_", " ", no_punctuation)
    no_punctuation = no_punctuation.lower()

    return no_punctuation


final_text = reduce_punctuation(only_text)



""" 2d """

""" 2e """

""" 2f """

""" 2g """