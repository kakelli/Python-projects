from math import ulp
import spacy
from spacy import displacy
import random
from spacy.training.example import Example
from spacy.pipeline.textcat import single_label_bow_config, Config

#Load model
'''nlp = spacy.load("en_core_web_sm")'''
'''doc = nlp("Microsoft is oughting to buy $1 Billion company at Norway.")'''
'''for i in doc:
    print(i.text, i.pos_, i.dep_)'''

#Tokenization - Splitting of words
'''for i in doc:
    print(i.text)'''

#Lemmatization - Root form of word
'''for i in doc:
    print(i.text, '->', i.lemma_)'''

#POS Tagging - Grammatical role
'''for i in doc:
    print(i.text, i.pos_)'''

#Named Entity Recognition- Realize the name of real world entities
'''for i in doc.ents:
    print(i.text, i.label_)'''

#Depndancy Parsing - SHowing how words r related in a sentance
'''for i in doc:
    print(i.text, i.dep_, i.head.text)'''

#Visualization of parsing
'''displacy.serve(doc, style='dep')'''

#Sentence detection
'''for i in doc.sents:
    print(i.text)'''

#Custom pipeline componenet
'''@spacy.language.Language.component("my_component") #Adding custom pipeline
def my_component(doc):
    print("Custom pipeline executed!")
    return doc
nlp.add_pipe("my_component", first=True)
doc = nlp("Hello America!")'''

#Training Named Entity Recognition (NER) model
'''nlp = spacy.blank('en') #Building from scratch
ner = nlp.add_pipe('ner') #Adds a component of ner
ner.add_label('SOFTWARE')
train_data = [
    ('I use Chrome daily!', {"entities": [(7,12, 'SOFTWARE')]}),
    ('He like Visual Studio code to work with', {"entities":[(9,26)]})
]

optimizer = nlp.initialize()
for i in range(10):
    random.shuffle(train_data)
    for text, annots in train_data:
        example = Example.from_dict(nlp.make_doc(text), annots)
        nlp.update([example], sgd = optimizer) #Stochastic Gradient Descent'''

#Text Classification (Binary /Multi Label)
'''nlp = spacy.blank('en')'''
'''config = Config().from_str(single_label_bow_config)
text_cat = nlp.add_pipe("textcat", config=config)

text_cat.add_label("POSITIVE")
text_cat.add_label("NEGATIVE")'''

#Saving / Loading model
'''nlp.to_disk("my_model")  #Saving the model'''
'''nlp2 = spacy.load('my_model')''' #Loading the model

#Using Spacy with Transformers (BERT)
'''nlp = spacy.load('en_core_web_trf')
doc = nlp("Apple is a trillion-dollar company.")'''