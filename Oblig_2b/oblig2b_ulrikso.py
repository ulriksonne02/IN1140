#Oppg 1
#1) Setningen er strukturelt flertydig fordi den kan bety at enten per har boken eller at ola har boken
#2) NP -> NP PP fordi da "ola"(NP) og "med  boka"(PP) slås sammen til "ola med boka"(NP). Etter det blir det "slår"(V) og "ola med boka"(NP) vil bli til en VP, herfra ser treet likt ut som første tolkning.
#3) Ja det er rekursjon i VP -> VP PP fordi VP kan inneholde uendelig PP. Et eksempel: "Per slår ola med boka med boka med boka med boka" fordi VP er rekursiv.


#Oppg 3
#2)
    #Jeg brukte den manuelle metoden og fikk tallene under.
    #fordi jeg gikk tom for desimaler gikk jeg over til å bruke fractions via wolfram alpha, fordi wolfram alpha er bra.
    #
    #POS=3/7*(2*1*2*2*1*2)/47^6=1.99937*10^-10
    #NEG=4/7*(1*1*2*1*2*2)/50^6=2.92571*10^-10
    #setningen "førsteklasses artist men dårlig og kjedelig album" er NEG
    #
    #POS=3/7*(2*2*2*2)/47^4=1.40524*10^-6
    #NEG=4/7*(1*2*1*2)/50^4=3.65714*10^-7
    #setningen "fortreffelig orkester og flott album" er POS
#3)
    #Jeg er enig i klassiferingen
#Oppg 4

#oppgave 4.1

#Blant topp 10 ord er flere av dem logiske, men noen er rare. at gåseøyne er regnet som positivt er litt rart.


#Blant topp 20 ord: at "som", "er" og "i" er negativt virker rart.

#For å fikse disse problemene
#Jeg tror ikke at problemet er et for lite datasett.
#Gitt kunnskap fra neste deloppgave kan å bruke en mer avansert klassifiseringsmetode være nyttig.

#oppgave 4.2

#under er accuracy som målt med accuracy(dev_set) for forskjellige varianter av BOW klassifisering.

#BOW uten stoppord og tegnsetting:0.65
#BOW + bigrams:0.72
#BOW + bigrams + trigrams:0.73


#under er accuracy med test_set

#BOW uten stoppord og tegnsetting:0.61
#BOW + bigrams:0.63
#BOW + bigrams + trigrams:0.64


#beste accuracy fikk jeg med BOW + bigrams + trigrams


#analyse av mest informative trekk:
#BOW uten stoppord og tegnsetting:
#De fleste positive gir mening, med unntak som "russisk" og "island" som man ikke ville forventet at var sterkt positive.
#"intrigen" er også negativ som er kanskje litt rart.

#BOW + bigrams:

#Et bigram er inkludert i de 10 mest informative trekkene ("typen","«"). og er 9 ganger mer negativt enn positivt.
#Ellers er resten det samme.

#BOW + bigrams + trigrams:

#Listen er identisk for de topp 10 mest informative trekkene som den med bare bigrams, men accuracy gikk opp. Kommer det trigrams i listen som forbedrer nøyaktigheten


#spørsmål 1 kode:

import nltk
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.corpus import CategorizedPlaintextCorpusReader

corpus_root = 'NoReC/'
reviews = CategorizedPlaintextCorpusReader(corpus_root, '.*\.txt',cat_pattern=r'(neg|pos)/.*')

all_words = nltk.FreqDist(w.lower() for w in reviews.words())
#print(list(all_words)[0:100])
print(type(reviews.words()))



word_features = list(all_words)[:1000]
#print(reviews.sents(categories='pos'))
# Funksjon for å hente ut trekk til vår Naive Bayes.
def document_features(document): #tar inn et dokument
    document_words = set(document) # bruker sett for å enkelt kunne hente ut alle unike ord fra dokumentet
    features = {}
    for word in word_features: #sjekke om hvert ord i word_features finnes i dokumentet
        features['contains({})'.format(word)] = (word in document_words)
    return features

print(len(word_features))
documents = []

for category in reviews.categories():
    for fileid in reviews.fileids(category):
        documents.append((reviews.words(fileid), category)) #lage en liste av (text, klasse) par som skal brukes ved trening og testing
#print(len(rei))
featuresets = [(document_features(d), c) for (d,c) in documents] #bruke forrige funksjon til å generere trekk

train_set, test_set = featuresets[182:], featuresets[:122] #dele dataen i train og test sets
print(len(train_set),len(test_set))


classifier = nltk.NaiveBayesClassifier.train(train_set)

accuracy = classify.accuracy(classifier, test_set) #Regne ut accuracy på vår Naive Bayes
print (accuracy)

print(classifier.show_most_informative_features(30)) #Vise de fem mest relevante trekk for klassifiseringen
