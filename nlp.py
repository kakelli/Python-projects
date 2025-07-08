import nltk
import numpy
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk import ne_chunk
from nltk import ngrams
from nltk import FreqDist
from nltk.classify import NaiveBayesClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.corpus import PlaintextCorpusReader
from nltk import CFG
from nltk.parse.generate import generate
'''nltk.download('all') #Downloading the models and corpora'''
#Tokenization- breaking of a sentence or a para into small tokens
'''text = 'This is the NLP python program! It is important'
print(word_tokenize(text)) #Separates word by word
print(sent_tokenize(text)) #Separates sentence by sentance'''

#Stopwords removal
'''sw = set(stopwords.words('english'))
txt = word_tokenize('This is a hola soy.')
filtere = [i for i in txt if i.lower() not in sw]
print(filtere)'''

#Stemming- reducing word to root form
'''stemmer = PorterStemmer() #creating an instance
print(stemmer.stem('running'))
print(stemmer.stem('walking'))'''

#Lemmatization  - smarter cousin of Stemming
'''lemmatizer = WordNetLemmatizer() #Creating an instance
print(lemmatizer.lemmatize('running', pos = 'v')) #Part of speech is verb
print(lemmatizer.lemmatize('better', pos = 'a'))'''

#POS Tagging (Part of Speech)
'''tokens = word_tokenize('NLTK is a great and fun tool to use')
print(pos_tag(tokens))'''

#Named Entity Recognition- (NER)
'''txt = 'Alexander is the top King of Greece'
token = word_tokenize(txt)
tags = pos_tag(token)
tree = ne_chunk(tags)
print(tree)'''

#n-grams- finding sequences
'''text = 'This is a NLP python project!'
tokens = word_tokenize(text)
bigram = list(ngrams(tokens, 2))
print(bigram)'''

#Frequency distribution - How often does a word occur
'''word = word_tokenize('I never should have called you jinx, pow pow')
fdist = FreqDist(word)
print(fdist['pow']) #Prints the number of occurence of pow'''

#Chunking- grouping the words into phrases
'''grammer = 'NP:{<DT>?<JJ>*<NN>}'
parser = nltk.RegexpParser(grammer)
sentence = [('The', 'DT'), ('small', 'JJ'), ('wee', 'NN')]
tree = parser.parse(sentence)
tree.draw() # type: ignore'''

#Text Classifier- Sentiment Analysis using Naive Bayes Classifier
'''train_data = [
    ({'text': 'I love this movie'}, 'pos'),
    ({'text': 'I hate this movie'}, 'neg')
]
def word_count(text):
    return {word: True for word in word_tokenize(text)} 
train_set = [(word_count(text['text']), label) for text, label in train_data]
classifier = NaiveBayesClassifier.train(train_set)
print(classifier.classify(word_count('I love interstellar')))'''

#Collocations - Common word pairs
'''text = word_tokenize('I love Dogs. Dogs love me too. I love love.')
finder = BigramCollocationFinder.from_words(text)
print(finder.nbest(BigramAssocMeasures.likelihood_ratio, 3))'''

#Custom Corpuse Loading
'''corpus = PlaintextCorpusReader('/home/johnbright/Documents/IMPD', '.*\\.txt') 
print(corpus.words()) '''

#Parsing tree using COntext free grammer (CFG)
grammer = CFG.fromstring("""
S -> NP VP
NP -> DT NN
VP -> V NP   
DT -> 'the'
NN -> 'dog' | 'cat'
V -> 'chased' | 'saw'                      
""")
for sentence in generate(grammer, n=5):
    print(' '.join(sentence))