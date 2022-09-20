
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
import nltk


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
    no_punctuation = re.sub(r"’", "", no_punctuation)
    no_punctuation = no_punctuation.lower()

    return no_punctuation


final_text = reduce_punctuation(only_text)



""" 2d """
""" Use the nltk.FreqDist() to count the words. Report the 20 most frequent words in a table with 
their absolute frequencies. """



tokens = nltk.word_tokenize(final_text)
fdist1 = nltk.FreqDist(tokens)



most_common_20 = fdist1.most_common(20)
# print(data)

col_names = ["word", "absolute frequency"]

# print(tabulate(most_common_20, headers=col_names))
#
# word      absolute frequency
# ------  --------------------
# the                     3702
# and                     3087
# a                       1829
# to                      1711
# of                      1434
# he                      1197
# was                     1168
# it                      1149
# in                       941
# that                     905
# his                      815
# i                        781
# you                      777
# tom                      688
# with                     647
# but                      580
# they                     558
# for                      525
# had                      512
# him                      434



""" 2e """
""" Consider the frequencies of frequencies. How many words occur only 1 time? How many words occur n times,
 etc. for n = 1, 2, …, 10; how many words have between 11 and 50 occurrences; how many have 51-100 occurrences;
  and how many words have more than 100 occurrences? Report in a table! """


all_words = fdist1.most_common()

counter_1 = 0
counter_2 = 0
counter_3 = 0
counter_4 = 0
counter_5 = 0
counter_6 = 0
counter_7 = 0
counter_8 = 0
counter_9 = 0
counter_10 = 0
counter_11_50 = 0
counter_51_100 = 0
counter_100 = 0


for item in all_words:
    (word, freq) = item
    if freq == 1:
        counter_1 += 1
    elif freq == 2:
        counter_2 += 1
    elif freq == 3:
        counter_3 += 1
    elif freq == 4:
        counter_4 += 1
    elif freq == 5:
        counter_5 += 1
    elif freq == 6:
        counter_6 += 1
    elif freq == 7:
        counter_7 += 1
    elif freq == 8:
        counter_8 += 1
    elif freq == 9:
        counter_9 += 1
    elif freq == 10:
        counter_10 += 1
    elif 11 <= freq <= 50:
        counter_11_50 += 1
    elif 51 <= freq <= 100:
        counter_51_100 += 1
    else:
        counter_100 += 1

col_names = ["frequency", "number of words with this frequency"]
data = [["1", counter_1],
        ["2", counter_2],
        ["3", counter_3],
        ["4", counter_4],
        ["5", counter_5],
        ["6", counter_6],
        ["7", counter_7],
        ["8", counter_8],
        ["9", counter_9],
        ["10", counter_10],
        ["11-50", counter_11_50],
        ["51-100", counter_51_100],
        [">100", counter_100]]

# print(tabulate(data, headers=col_names))

# frequency      number of words with this frequency
# -----------  -------------------------------------
# 1                                             3767
# 2                                             1202
# 3                                              608
# 4                                              382
# 5                                              231
# 6                                              172
# 7                                              147
# 8                                              127
# 9                                               74
# 10                                              93
# 11-50                                          508
# 51-100                                          81
# >100                                           104






""" 2f """
""" We order the words by their frequencies, the most frequent word first. Let r be the
frequency rank for each word and n its frequency. Hence, the most frequent word gets rank
1, the second most frequent word gets rank two, and so on. According to Zipf’s law, r*n should be nearly constant. 
Calculate r*n for the 20 most frequent words and report in a
table. How well does this fit Zipf’s law? Answer in text. """


# print(most_common_20)
table_data = []
n = 1
for tuple in most_common_20:
    n = n
    (word, frequency) = tuple
    table_data.append([n, frequency, n*frequency])
    n += 1


col_names = ["rank r", "frequency n", "r*n"]
# print(tabulate(table_data, headers=col_names))

#   rank r    frequency n    r*n
# --------  -------------  -----
#        1           3702   3702
#        2           3087   6174
#        3           1829   5487
#        4           1711   6844
#        5           1434   7170
#        6           1197   7182
#        7           1168   8176
#        8           1149   9192
#        9            941   8469
#       10            905   9050
#       11            815   8965
#       12            781   9372
#       13            777  10101
#       14            688   9632
#       15            647   9705
#       16            580   9280
#       17            558   9486
#       18            525   9450
#       19            512   9728
#       20            434   8680





""" 2g """
""" Try to plot the rank against frequency for all words (not only the 20 most frequent ones).
First, make a plot where you use linear scale on the axes. Then try to make a plot similarly to
the Wikipedia figure below with logarithmic scales at both axes. Logarithms are available in
numpy, using functions functions like log(), for the natural logarithm, and log2() for the base-
2 logarithm. An alternative to using one of these functions is to explore loglog() plotting from
matplotlib.pyplot directly. """

import matplotlib.pyplot as plt
import numpy as np


rank = []
frq = []
n = 1
for tuple in all_words:
    n = n
    (word, frequency) = tuple
    rank.append(n)
    frq.append(frequency)
    n += 1

# print(frq)

# #plotting simply rank against frequency
# plt.plot(rank, frq)
# plt.xlabel('rank')
# plt.ylabel('frequency')

# #plotting rank against frequency with logarithmic scales
# plt.plot(np.log(rank), np.log(frq))
# plt.xlabel('log(rank)')
# plt.ylabel('log(frequency)')

#plotting rank against frequency with logarithmic scales (base-2 logarithm)
plt.plot(np.log2(rank), np.log2(frq))
plt.xlabel('log2(rank)')
plt.ylabel('log2(frequency)')
plt.show()