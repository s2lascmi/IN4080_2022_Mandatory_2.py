import numpy
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
import spacy



sent0 = "For example, this isn't a well-formed sentence."

sent1 = "Maybe motel-keeping isn't the nation's biggest industry."
#print(word_tokenize(sent1))

brown_sentences = [s for s in brown.sents()]
#print("Brown sentence: ",brown_sentences[1750])


nlp = spacy.load("en_core_web_sm")
doc0 = nlp(sent0)
#print(doc0)

tok00 = doc0[4]
#print(tok00)

# for token in doc0:
#     print(token.text)

doc0 = nlp(sent0)
doc1 = nlp(sent1)
#print("sent0 with spaCy: ", doc0)


# for token in doc0:
#     print(token.text)
# print("sent0 with nltk: ", word_tokenize(sent0))



#print("sent1 with SpaCy: ", doc1)
#for token in doc1:
    #print(token.text)
#print("sent0 with nltk: ", word_tokenize(sent1))

# Spacy: well - formed are three tokens, is+n't and nation+'s as two separate tokens, punctuation as token
# Brown: "isn't" and "nation's" as one token
# NLTK: well-formed as one token, is+n't and nation+'s as two separate tokens, punctuation as token

sent2 = "It listed his wife's age as 74 and place of birth as Opelika , Ala."


# doc3 = nlp(sent2)
# print("sent2 with spaCy: ", doc3)
# for token in doc3:
#     print(token.text)
#
# print("NLTK tokenization: ", word_tokenize(sent2))
#
# brown_sentences = [s for s in brown.sents()]
# print("Brown tokenization: ", brown_sentences[36])


sent3 = ("He didn't know what was so tough about Vivian's world, " +
"slopping around Nassau with what's-his-name.")


doc4 = nlp(sent3)
print("sent2 with spaCy: ", doc4)
for token in doc4:
    print(token.text)

print("NLTK tokenization: ", word_tokenize(sent3))

brown_sentences = [s for s in brown.sents()]
print("Brown tokenization: ", brown_sentences[55310])