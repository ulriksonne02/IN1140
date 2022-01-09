import nltk
import numpy as np
from collections import Counter
from nltk import bigrams,trigrams
from nltk.corpus import gutenberg
from collections import defaultdict


#Oppgave 1
#1
#(<s>,Per),(Per,synger),(synger,ikke),(synger,<\s>)
#(<s>,Kari),(Kari,synger),(synger,<\s>)
#(<s>,Ola),(Ola,synger),(synger,ikke),(ikke,<\s>)
#2

#For å regne ut sannsynligheten deler vi forekomster av bigrammet på forekomster av det første ordet i bigrammet i korpuset

#3
#P(<s> Kari synger ikke <\s>)
#(1/3)*(1/1)*(2/3)*(2/2)
#P=0.222

#oppgave 2
#Jeg:PO
#spiser: VB
#sushi: NN
#med: PR
#pinner: NN


#oppgave 3

kjv_words = gutenberg.words("bible-kjv.txt")

#del 1
print("tokens i 'bible-kjv.txt",len(kjv_words))

#print(kjv_words)
#kjv_tokens = nltk.word_tokenize(kjv_words)
#nltk.pos_tag(kjv_tokens)

#del 2
distinct_words = []
for w in kjv_words:
    distinct_words.append(w.lower())
total_distinct_words = set(distinct_words)
print("Distinkte ord: ",len(total_distinct_words))

#del 3
fd_kjv_words = Counter(kjv_words)
print("Most common words: ", fd_kjv_words.most_common(20)) #Denne fungerer // ikke slett

probabilities = {}
for word, count in fd_kjv_words.items():
    probabilities[word] = count/len(kjv_words)
print(sum(probabilities.values()))

#del 4
print("frekvens av 'heaven':",probabilities["heaven"],"antall:",kjv_words.count("heaven"))
print("frekvens av 'death':",probabilities["death"],"antall:",kjv_words.count("death"))
print("frekvens av 'life':",probabilities["life"],"antall:",kjv_words.count("life"))

kjv_sents = gutenberg.sents("bible-kjv.txt")

#del 5
print("bigrammer i setning 4:\n",list(bigrams(kjv_sents[3])),"\n")
#del 6
print("trigrammer i setning 5:\n",list(trigrams(kjv_sents[6])),"\n")


#Oppgave 4
