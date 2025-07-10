from textblob import TextBlob
from textblob import Word
from textblob.classifiers import NaiveBayesClassifier



#Creating a textblob object
'''blob = TextBlob("Textblob is easy and fun to use")'''
'''print(blob) #Successfully created object'''

#Tokenization - splitting of words or sentences
'''print(blob.words) #Splitting of words
print(blob.sentences) #Splitting of sentences'''

#Part of SPeech (POS) tagging
'''print(blob.tags) #POS given'''

#Noun Phase Extraction
'''print(blob.noun_phrases) '''

#Sentiment Analysis
'''print(blob.sentiment) #Gives polarity and subjectivity '''

#Correction of Spelling
'''blob = TextBlob("I thik my speling is good")
corrected = blob.correct()
print(corrected)'''

#Word Inflection and Lemmatization
'''word = Word("studies")
print(word.lemmatize()) #The output is study'''

#Singular and plural
'''word = Word('apple')
print(word.pluralize()) #apples

word = Word('cacti')
print(word.singularize()) #Cactus'''

#Text Classification (NaiveBayesAnalyzer)
'''blob = TextBlob("I love the movie Interstellar", analyzer=NaiveBayesAnalyzer())
print(blob.sentiment)'''

#Creating Textblob from Files or URLs
'''with open('example.txt', 'r') as f:
    blob = TextBlob(f.read())
    print(blob.sentiment)'''

#Training a custom classifier
'''train = [
    ("I love this", 'pos'),
    ("I hate this", "neg")
]
cl = NaiveBayesClassifier(train)
blob = TextBlob("I hate this traffic", classifier = cl)
print(blob.classify())'''