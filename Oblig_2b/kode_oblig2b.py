import nltk
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.corpus import stopwords
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import word_tokenize

# Vi må bruke ``PlaintextCorpusReader'' fra nltk.corpus for å bruke bruke samme metoder som vist på forelesningen på vårt egen korpus.
# Din kode bør ligge i samme mappe som NoReC korpuset.
def load_corpus():
    corpus_root = 'NoReC/'
    reviews = PlaintextCorpusReader(corpus_root, '.*\.txt')
    return reviews

# Hente ut alle positive og negative ord som finnes i movie_reviews
def get_word_reviews(reviews, pos_reviews, neg_reviews):
    for fileid in reviews.fileids():
        if fileid.startswith('pos'):
            words = reviews.words(fileid)
            pos_reviews.append([x.lower() for x in words])
        elif fileid.startswith('neg'):
            words = reviews.words(fileid)
            neg_reviews.append([x.lower() for x in words])
    return pos_reviews, neg_reviews

# Bruk denne for å fordele dataen din i train, dev, og test set.
def splitdata(pos_reviews_feat, neg_reviews_feat):
    test_set = pos_reviews_feat[:122] + neg_reviews_feat[:122]
    dev_set = pos_reviews_feat[122:182] + neg_reviews_feat[122:182]
    train_set = pos_reviews_feat[182:] + neg_reviews_feat[182:]

    return test_set, dev_set, train_set

def document_features(document,word_features): #tar inn et dokument
    document_words = set(document) # bruker sett for å enkelt kunne hente ut alle unike ord fra dokumentet
    features = {}
    for word in word_features: #sjekke om hvert ord i word_features finnes i dokumentet
        features['contains({})'.format(word)] = (word in document_words)
    return features

def main():
    pos_reviews = []
    neg_reviews = []
    reviews = load_corpus()
    pos_reviews, neg_reviews = get_word_reviews(reviews, pos_reviews, neg_reviews)

    poslist = []
    neglist = []
    alllist = []
    for lists in pos_reviews:
        poslist = poslist + lists

    for lists in neg_reviews:
        neglist = neglist + lists
    for lists in neg_reviews+pos_reviews:
        alllist = alllist + lists

    all_words = nltk.FreqDist(alllist[:1000])
    pos_words = nltk.FreqDist(poslist[:1000])
    neg_words = nltk.FreqDist(neglist[:1000])
    #print(all_words)


    #pos_features = document_features(pos_words, all_words)
    #neg_features = document_features(neg_words, all_words)
    other_words = []
    pos_features = {}
    neg_features = {}
    word_counter = 0
    for word in pos_words:
        if word_counter>700:
            pos_features[word]=True
        else:
            other_words.append(word)
        word_counter += 1
    word_counter = 0
    for word in neg_words:
        if word_counter>700:
            neg_features[word]=True
        else:
            other_words.append(word)
        word_counter += 1
    #print(pos_features)

    train_list = []
    train_list.append((pos_features,"pos"))
    train_list.append((neg_features,"neg"))
    #test_set,dev_set,train_set = splitdata(pos_features, neg_features)
    #print(test_set)

    classifier = nltk.NaiveBayesClassifier.train(train_list)

    accuracy = classify.accuracy(classifier, other_words)
    #print(accuracy)

    #print(type(data))
    #print(test_set[0])
    #all_words = nltk.FreqDist(pos_reviews)
    # Her skal du skrive koden din.

    # Du skal bruke fordelingen av dataen gitt i denne koden. Bruk derfor funksjonen splitdata()

    # Du kan bruke listen av stoppord som finnes i NLTK, for å bruke den norske listen kan du bare skrive:
    # stopwords_no = stopwords.words('norwegian')



if __name__ == "__main__":
    main()
