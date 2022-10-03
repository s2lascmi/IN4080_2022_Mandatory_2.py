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

""" Exercise 1 """
""" The NLTK book, chapter 2, has an example in section 2.1, in the paragraph Brown Corpus, where they
compare the frequency of modal verbs across different genres. We will conduct a similar experiment,
We are in particular interested in to which degree the different genres use the masculine pronouns
(he, him) or the feminine pronouns (she, her). """


""" 1a """
""" Conduct a similar experiment as the one mentioned above with the genres: news, religion,
government, fiction, romance as conditions, and occurrences of the words: he, she, her, him,
as events. Make a table of the conditional frequencies and deliver code and table.
(Hint: Have you considered case folding?) """


pronouns = ["he", "him", "she", "her"]
genres = ['news', 'religion', 'government', 'fiction', 'romance']
texts = brown.words()
news_text = brown.words(categories=genres)


cfd = nltk.ConditionalFreqDist((genre, word.lower())
                               for genre in brown.categories()
                               for word in brown.words(categories=genre))
# cfd.tabulate(conditions=genres, samples=pronouns)

# Table as the code above produces it
#              he  him  she  her
#       news  642   93   77  121
#   religion  206   94   12    8
# government  169   26    1    3
#    fiction 1308  382  425  413
#    romance 1068  340  728  680



""" 1b """
""" Answer in words what you see. How does gender vary with the genres? """
""" Answer to the question can be found in the PDF file for Part A """

""" Maybe not so surprisingly, the masculine forms are more frequent than the feminine forms across all
genres. However, we also observe another pattern. The relative frequency of her compared to she
seems higher than the relative frequency of him compared to he. We want to explore this further
and make a hypothesis, which we can test. 

Ha: The relative frequency of the objective form, her, of the feminine personal pronoun (she or
her) is higher than the relative frequency of the objective form, him, of the masculine personal
pronoun, (he or him)."""


""" 1c """
""" First, consider the complete Brown corpus. Construct a conditional frequency distribution,
which uses gender as condition, and for each gender counts the occurrences of nominative
forms (he, she) and objective forms (him, her). Report the results in a two by two table. Then
calculate the relative frequency of her from she or her, and compare to the relative
frequency of him from he or him. Report the numbers. Submit table, numbers and code you
used. """


gender = ["male", "female"]

gender_word = [(gender_item, word.lower())
                               for gender_item in gender
                               for word in brown.words()]

cfd = nltk.ConditionalFreqDist(gender_word)
male_subject = cfd["male"]["he"]
male_object = cfd["male"]["him"]
female_subject = cfd["female"]["she"]
female_object = cfd["female"]["her"]

col_names = ["form", "male pronouns", "female pronouns"]
data = [["nominative forms (he, she)", male_subject, female_subject],
             ["objective forms (him, her)", male_object, female_object]]

# print(tabulate(data, headers=col_names))

rel_freq_her = female_object /(female_subject + female_object)
rel_freq_him = male_object / (male_subject + male_object)

# # Relative Frequency of Her is higher than that of Him --> Supports hypothesis
# print("\n","Relative Frequency \'Her\'", rel_freq_her) # 0.5149253731343284
# print("Relative Frequency \'Him\'", rel_freq_him) # 0.21525437659242214

# Table as the code above produces it
# form                          male pronouns    female pronouns
# --------------------------  --------------  -----------------
# nominative forms (he, she)            9548               2860
# objective forms (him, her)            2619               3036


""" It is tempting to conclude from this that the objective form of the feminine pronoun is relatively
more frequent than the objective form of the male pronoun. Beware, however, her is not only the
feminine equivalent of him, but also of his. So what can we do? We could do a similar calculation as
in point (b), comparing the relative frequency of her –not to the relative frequency of him –but
compare her + hers to him + his. That might give relevant information, but it does not check the
hypothesis, Ha. """


""" 1d """
""" What could work is to use a tagged corpus, which separates between the two forms of her,
i.e, if the corpus tags her as a personal pronoun differently from her as a possessive pronoun
(determiner). The tagged Brown corpus does that. Use this to count the occurrences of she,
he, her, him as personal pronouns and her, his, hers as possessive pronouns. See NLTK book,
Ch. 5, Sec. 2, for the tagged Brown corpus. Report in a two-ways table.

You could solve this by using the originally tagged Brown corpus with the original tags.

Alternatively, you may use the corpus with the universal pos tags. Both use different tags for
the two functions of her. The name of the tags are different, though.
 """
she_pron = 0
he_pron = 0
her_pron = 0
him_pron = 0

her_det = 0
his_det = 0
hers_det = 0

tagged_corpus = nltk.corpus.brown.tagged_words(tagset="universal")
for item in tagged_corpus:
    if item[0].lower() == "she" and item[1] =="PRON":
        she_pron += 1
    elif item[0].lower() == "he" and item[1] =="PRON":
        he_pron += 1
    elif item[0].lower() == "her" and item[1] =="PRON":
        her_pron += 1
    elif item[0].lower() == "him" and item[1] =="PRON":
        him_pron += 1
    elif item[0].lower() == "her" and item[1] =="DET":
        her_det += 1
    elif item[0].lower() == "his" and item[1] =="DET":
        his_det += 1
    elif item[0].lower() == "hers" and item[1] =="DET":
        hers_det += 1

pron_total = she_pron + he_pron + her_pron + him_pron
det_total = her_det + his_det
total_pronouns = det_total + pron_total + she_pron + he_pron + her_pron + her_det + him_pron + his_det + her_det

col_names = ["pronouns", "personal pronouns", "possessive pronouns", "total"]
data = [["she", she_pron, "0", she_pron],
        ["he", he_pron, "0", he_pron],
        ["her", her_pron, her_det, her_pron + her_det],
        ["him", him_pron, "0", him_pron],
        ["his", "0", his_det, his_det],
        ["hers", "0", hers_det, hers_det],
        ["total", pron_total, det_total, total_pronouns]]


# print(tabulate(data, headers=col_names))
# Table as it is produced by the code above
# pronouns      personal pronouns    possessive pronouns    total
# ----------  -------------------  ---------------------  -------
# she                        2860                      0     2860
# he                         9546                      0     9546
# her                        1107                   1929     3036
# him                        2619                      0     2619
# his                           0                   6957     6957
# hers                          0                      0        0
# total                     16132                   8886    51965

""" 1e """
""" We can now correct the numbers from point (b) above. How large percentage of the
feminine personal pronoun occurs in nominative form and in objective form? What are the
comparable percentages for the masculine personal pronoun? """

rel_freq_her_corrected = her_pron /(her_pron + she_pron)
rel_freq_she = she_pron / (her_pron + she_pron)
rel_freq_him_corrected = him_pron / (him_pron + he_pron)
rel_freq_he = he_pron / (him_pron + he_pron)


# print("\n","Relative Frequency \'Her\'", rel_freq_her_corrected) #0.2790521804890345
# print("\n","Relative Frequency \'She\'", rel_freq_she) #0.7209478195109654
# print("\n", "Relative Frequency \'Him\'", rel_freq_him_corrected) #0.21528976572133168
# print("\n","Relative Frequency \'He\'", rel_freq_he) #0.7847102342786683


""" 1f """
""" Illustrate the numbers from (d) with a bar chart. """

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

pronouns = ("she_pers", "he_pers", "her_pers", "him_pers", "her_poss", "his_poss", "hers_poss")
y_pos = np.arange(len(pronouns))
use = [she_pron, he_pron, her_pron, him_pron, her_det, his_det, hers_det]

plt.bar(y_pos, use, align="center", alpha=0.5)
plt.xticks(y_pos, pronouns)
plt.ylabel("Usage")
plt.title("Usage of personal and possessive pronouns in Brown Corpus")

# plt.show()


""" 1g """
""" Exercise 1f can be found in the PDF document. """