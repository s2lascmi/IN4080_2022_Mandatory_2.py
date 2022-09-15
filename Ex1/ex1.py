import nltk
#nltk.download('book')
#from nltk.book import *
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import brown
from tabulate import tabulate

# Exercise 1 from book
calc = 12 / (4 + 1)
#print(calc)

# Exercise 3 from book
#print(['Monty', 'Python'] * 20)

#  Exercise 8 from book
"""The set() function creates a set object. The items in a set list are unordered, so it will appear in random order."""



#  Exercise 16 from book
# print(list(range(10))) #prints list from 0-9
# print(list(range(10, 20))) #prints list from 10-19


#  Exercise 19 from book

#print(len(sorted(set(w.lower() for w in text1))))

#print(len(sorted(w.lower() for w in set(text1))))

#  Exercise 9 from book

#my_string = "Dummy string for test"
#my_string2 = "Dummy 2"
#my_string2
#print(my_string)


"""Exercises from the worksheet"""


"""Exercise 1"""

list = ["hello", "hi", "hallo", "hello"]
counter = {}

for letter in list:
    if letter not in counter:
        counter[letter] = 0
    counter[letter] += 1

#print(counter)

# list = ["hello", "hi", "hallo", "hello"]
# print(my_frequency(list))

#fd = nltk.FreqDist(text1)
# print(fd)
# print(fd.items())
# print(fd.keys())
# print(fd.values())
# fd.tabulate()
# fd.plot()




"""Exercise 2"""

"""26"""
#sum = sum(len(w) for w in text1)
#print(sum)

"""27"""

def vocab_size(text):
    length = len(text)
    return length

#print(vocab_size(text1))

"""28"""

def percent(term, text):
    text_length = len(text)
    counter = {}
    for word in text:
        if word not in counter:
            counter[word] = 0
        counter[word] += 1

    # print(counter[term], term)
    # print(text_length)
    percentage = round((counter[term] / text_length), 8)
    return percentage

# print(percent("Moby", text1))



"""Exercise 3"""


news_text = brown.words()
fdist = nltk.FreqDist(w.lower() for w in news_text)
pronouns = ["i", "he", "she", "we", "they"]
occurrences = []
for m in pronouns:
    #print(m + ':', fdist[m], end=' ')
    occurrences.append(fdist[m])

data = [[pronouns[0], occurrences[0]] , [pronouns[1], occurrences[1]], [pronouns[2], occurrences[2]], [pronouns[3],
                                                                                                       occurrences[3]], [pronouns[4],
                                                                                                                         occurrences[4]]]

# print(tabulate(data))
# plt.bar(pronouns, occurrences)
#
# plt.show()


"""Exercise 4"""

labels = ['news', 'romance']
cfd = nltk.ConditionalFreqDist((genre, word)
for genre in brown.categories()
for word in brown.words(categories=genre))
genres = ['news', 'science_fiction']
pronouns_4 = ["I", "he", "she", "we", "they"]
table = cfd.tabulate(conditions=genres, samples=pronouns_4)


"""Exercise 5"""


"""Exercise 6"""

"""Exercise 7"""
num_words = len(brown.words())
print(num_words)
num_sents = len(brown.sents())
print(num_sents)

print("Average Sentence Length: ", round(num_words/num_sents))

brown_sentences = brown.sents()
sentence_length = []
for item in brown_sentences:
    sentence_length.append(len(item))
print(sentence_length)

# plt.hist(sentence_length, bins=20)
# plt.show()


"""Exercise 8"""
plt.boxplot(sentence_length)
plt.show()




