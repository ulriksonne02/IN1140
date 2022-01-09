import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
text = word_tokenize("And now for something completely different")
print(nltk.pos_tag(text))
