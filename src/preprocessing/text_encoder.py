from nltk import ngrams
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import scipy

class AvgWord2Vec:
    def __init__(self, vector_size=300, min_count=1, sg=1, ngram_range=(1, 1), window=5, seed=42):
        self.w2v = Word2Vec(vector_size=vector_size, min_count=min_count, sg=sg,
                            window=window, workers=5, seed=seed)
        self.min_count = min_count
        self.sg = sg
        self.window = window
        self.seed = seed
        self.vsize = vector_size

        self.ngrams = np.arange(ngram_range[0], ngram_range[1]+1, 1)
        self.raw = None
        self.corpus = None
        self.vocabulary_ = None

    def _create_ngrams(self, n, X):
        phrases = [list(ngrams(sent.split(), n)) for sent in X]
        for i in range(len(phrases)):
            phrases[i] = [" ".join(word) for word in phrases[i]]
        return phrases
    
    def _create_corpus(self, X, update_train=False):
        ngrams_phrases = {}
        for n in self.ngrams:
            phrases = self._create_ngrams(n, X)
            ngrams_phrases[f"{n}"] = phrases
        data = []
        corpus = []
        for n in ngrams_phrases.values():
            if len(data)==0:
                data = n
            data = [data[i] + n[i] for i in range(len(data))]
            corpus.extend(n)
        if update_train:
            self.corpus = corpus
        return data
    
    def _avg_sentence(self, data):
        avg_sentences = []
        for sent in data:
            if len(sent)!=0:
                avg_sentence = np.mean([self.w2v.wv.get_vector(word) for word in sent
                                        if word in self.w2v.wv.index_to_key], axis=0)
            else:
                avg_sentence = np.zeros(self.vsize)
            avg_sentences.append(avg_sentence)
        return np.array(avg_sentences)

    def fit(self, X):
        self.w2v = Word2Vec(vector_size=self.vsize, min_count=self.min_count, sg=self.sg,
                            window=self.window, workers=5, seed=self.seed)
        self.raw = list(X)
        self._create_corpus(update_train=True, X=X)
        self.w2v.build_vocab(self.corpus)
        self.w2v.train(self.corpus, total_examples=self.w2v.corpus_count, epochs=self.w2v.epochs)
        self.vocabulary_ = self.w2v.wv.key_to_index

    def fit_transform(self, X):
        self.w2v = Word2Vec(vector_size=self.vsize, min_count=self.min_count, sg=self.sg,
                            window=self.window, workers=5, seed=self.seed)
        self.raw = list(X)
        data = self._create_corpus(update_train=True, X=X)
        self.w2v.build_vocab(self.corpus)
        self.w2v.train(self.corpus, total_examples=self.w2v.corpus_count, epochs=self.w2v.epochs)
        self.vocabulary_ = self.w2v.wv.key_to_index
        return scipy.sparse.csr_matrix(self._avg_sentence(data))
        
    def transform(self, X):
        data = self._create_corpus(update_train=False, X=X)
        return scipy.sparse.csr_matrix(self._avg_sentence(data))
    
    def get_feature_names_out(self):
        columns = np.array([f'component_{i+1}' for i in range(self.vsize)])
    
if __name__ == "__main__":
    texts = ["I love eating chocolate icecream and strawberry cake",
            "Dogs are obedient"]
    w2v = AvgWord2Vec(ngram_range=(1, 2))
    data = w2v.fit_transform(texts)
    data_df = pd.DataFrame(data)
    print(data_df)