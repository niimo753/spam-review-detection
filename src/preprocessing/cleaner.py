import time
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords, words, wordnet
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords", quiet=True)
nltk.download("words", quiet=True)
nltk.download("wordnet", quiet=True)

from unidecode import unidecode
import re
import contractions
import string
from langdetect import detect

class BasicTextCleaning:
    def __init__(self):
        # define some necessary elements
        self.stopwords = set(stopwords.words('english'))
        self.words_corpus = set(words.words())
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

        # dictionary of methods can be used
        self.methods = {'lowercase': str.lower,
                        'accent_removal': self.accent_removal,
                        'strip': str.strip,
                        'nice_display': self.nice_display,
                        'tokenization': self.tokenization,
                        'stemming': self.stemming,
                        'lemmatization': self.lemmatization,
                        'punctuation_removal': self.punctuation_removal,
                        'stopwords_removal': self.stopwords_removal,
                        'contractions_expand': self.contractions_expand,
                        'nonsense_removal': self.nonsense_removal,
                        'number_removal': self.number_removal}
        
        self.punctuations = '[%s]' % re.escape(string.punctuation)

    def text_cleaning(self, texts, methods=None):
        if not methods:
            methods = ['accent_removal', 'lowercase','nice_display', 'punctuation_removal',
                       'stopwords_removal', 'lemmatization', 'stemming']
        if isinstance(texts, str):
            texts = [texts]
        cleaned_texts = []
        for text in texts:
            for method in methods:
                if method not in self.methods.keys():
                    raise Warning('Invalid method "{}". Basic text cleaning methods available: {}'.format(method, ", ".join(self.methods.keys())))
                text = self.methods[method](text)
            cleaned_texts.append(text)
        return cleaned_texts

    def strip_text(self, text):
        return text.strip()
    
    def lowercase(self, text):
        return text.lower()

    def contractions_expand(self, text):
        return contractions.fix(text)
    
    def number_removal(self, text):
        text = re.sub(r"[^a-zA-Z]", " ", text)
        text = re.sub(r'\s+', ' ', text)
        return text

    def nice_display(self, text):
        text = re.sub(r"([^\w\s([{\'])(\w)", r"\1 \2", text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def accent_removal(self, text):
        text = unidecode(text)
        return text

    def punctuation_removal(self, text):
        text = re.sub(self.punctuations, ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def stopwords_removal(self, text):
        return " ".join([word for word in text.split() if word not in self.stopwords])

    def stemming(self, text):
        return " ".join([self.stemmer.stem(word) for word in text.split()])

    def lemmatization(self, text):
        return " ".join([self.lemmatizer.lemmatize(word) for word in text.split()])

    def tokenization(self, text):
        if isinstance(text, str):
            return nltk.word_tokenize(text)
        return []

    def nonsense_removal(self, text):
        return " ".join([word for word in text.split() if wordnet.synsets(word)])

if __name__ == '__main__':
    None