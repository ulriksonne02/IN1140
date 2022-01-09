import nltk
from nltk import NaiveBayesClassifier
from nltk import classify
from nltk.corpus import stopwords
from nltk.corpus import PlaintextCorpusReader
from nltk import bigrams,trigrams
import string

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
    print(len(pos_reviews),len(neg_reviews))
    return pos_reviews, neg_reviews

# Bruk denne for å fordele dataen din i train, dev, og test set.
def splitdata(pos_reviews_feat, neg_reviews_feat):
    test_set = pos_reviews_feat[:122] + neg_reviews_feat[:122]
    dev_set = pos_reviews_feat[122:182] + neg_reviews_feat[122:182]
    train_set = pos_reviews_feat[182:] + neg_reviews_feat[182:]

    return test_set, dev_set, train_set

def bag_of_words(words):
    words_cleaned = []

    for word in words:
        word = word.lower()
        words_cleaned.append(word)

    words_dict = dict([word, True] for word in words_cleaned)
    # Å bruke en dictionary her hjelper oss med å slette alle duplikater.

    return words_dict

stopwords_norwegian = stopwords.words('norwegian')

def bag_of_words_no_stopwords_punct(words):
    words_cleaned = []

    for word in words:
        word = word.lower()
        if word not in stopwords_norwegian and word not in string.punctuation:
            words_cleaned.append(word)
    words_dict = dict([word, True] for word in words_cleaned)

    return words_dict

def bag_of_ngrams_no_stopwords_punct(words):
    words_cleaned = []

    lowered = [w.lower() for w in words if w not in stopwords_norwegian and w not in string.punctuation]
    for bi in bigrams(lowered):
        words_cleaned.append(str(bi))

    for word in words:
        word = word.lower()
        if word not in stopwords_norwegian and word not in string.punctuation:
            words_cleaned.append(word)
    words_dict = dict([word, True] for word in words_cleaned)

    return words_dict

def bag_of_trigrams_no_stopwords_punct(words):
    words_cleaned = []

    lowered = [w.lower() for w in words if w not in stopwords_norwegian and w not in string.punctuation]
    for bi in bigrams(lowered):
        words_cleaned.append(str(bi))
    for tri in trigrams(lowered):
        words_cleaned.append(str(tri))
    for word in words:
        word = word.lower()
        if word not in stopwords_norwegian and word not in string.punctuation:
            words_cleaned.append(word)
    words_dict = dict([word, True] for word in words_cleaned)

    return words_dict
print("\"")
def main():
    print("********************************************")
    print("BOW")
    print("********************************************")
    pos_reviews = []
    neg_reviews = []
    reviews = load_corpus()
    all_words = nltk.FreqDist(w.lower() for w in reviews.words())
    word_features = list(all_words)[:1000]
    pos_reviews, neg_reviews = get_word_reviews(reviews, pos_reviews, neg_reviews)
    pos_reviews_feat = []
    for words in pos_reviews:
        pos_reviews_feat.append((bag_of_words(words), 'pos'))
    neg_reviews_feat = []
    for words in neg_reviews:
        neg_reviews_feat.append((bag_of_words(words), 'neg'))
    test_set,dev_set,train_set = splitdata(pos_reviews_feat,neg_reviews_feat)

    print(len(test_set),  len(dev_set), len(train_set))
    classifier = NaiveBayesClassifier.train(train_set)
    accuracy = classify.accuracy(classifier, test_set)
    print("Accuracy on dev_set: %0.2f" % accuracy)
    print (classifier.show_most_informative_features(10))

    print("********************************************")
    print("BOW: uten stoppord og tegnsetting")
    print("********************************************")
    pos_reviews_feat = []
    for words in pos_reviews:
        pos_reviews_feat.append((bag_of_words_no_stopwords_punct(words), 'pos'))

    # lage trekk for negative anmeldelser
    neg_reviews_feat = []
    for words in neg_reviews:
        neg_reviews_feat.append((bag_of_words_no_stopwords_punct(words), 'neg'))
    test_set,dev_set,train_set = splitdata(pos_reviews_feat,neg_reviews_feat)
    print(len(test_set),  len(dev_set), len(train_set))
    classifier = NaiveBayesClassifier.train(train_set)
    accuracy = classify.accuracy(classifier, test_set)
    print("Accuracy on dev_set: %0.2f" % accuracy)
    print (classifier.show_most_informative_features(10))

    print("********************************************")
    print("BOW: + bigrams")
    print("********************************************")

    pos_reviews_feat = []
    for words in pos_reviews:
        pos_reviews_feat.append((bag_of_ngrams_no_stopwords_punct(words), 'pos'))

    # lage trekk for negative anmeldelser
    neg_reviews_feat = []
    for words in neg_reviews:
        neg_reviews_feat.append((bag_of_ngrams_no_stopwords_punct(words), 'neg'))

    test_set,dev_set,train_set = splitdata(pos_reviews_feat,neg_reviews_feat)

    print(len(test_set),  len(dev_set), len(train_set))
    classifier = NaiveBayesClassifier.train(train_set)
    accuracy = classify.accuracy(classifier, test_set)
    print("Accuracy on dev_set: %0.2f" % accuracy)
    print (classifier.show_most_informative_features(10))

    print("********************************************")
    print("BOW: + bigrams+trigrams")
    print("********************************************")

    pos_reviews_feat = []
    for words in pos_reviews:
        pos_reviews_feat.append((bag_of_trigrams_no_stopwords_punct(words), 'pos'))

    # lage trekk for negative anmeldelser
    neg_reviews_feat = []
    for words in neg_reviews:
        neg_reviews_feat.append((bag_of_trigrams_no_stopwords_punct(words), 'neg'))

    test_set,dev_set,train_set = splitdata(pos_reviews_feat,neg_reviews_feat)

    print(len(test_set),  len(dev_set), len(train_set))
    classifier = NaiveBayesClassifier.train(train_set)
    accuracy = classify.accuracy(classifier, test_set)
    print("Accuracy on dev_set: %0.2f" % accuracy)
    print (classifier.show_most_informative_features(10))

    # Her skal du skrive koden din.

    # Du skal bruke fordelingen av dataen gitt i denne koden. Bruk derfor funksjonen splitdata()

    # Du kan bruke listen av stoppord som finnes i NLTK, for å bruke den norske listen kan du bare skrive:
    # stopwords_no = stopwords.words('norwegian')


if __name__ == "__main__":
    main()
