from nltk import ngrams
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
import scipy

from nltk import ngrams

class AvgWord2Vec:
    def __init__(self, vector_size=300, min_count=1, sg=1, ngram_range=(1, 1), window=5, epochs=5, seed=42,
                 quiet=True):
        self.w2v = Word2Vec(vector_size=vector_size, min_count=min_count, sg=sg,
                            window=window, workers=4, seed=seed, epochs=epochs)
        self.min_count = min_count
        self.sg = sg
        self.window = window
        self.seed = seed
        self.vsize = vector_size
        self.epochs = epochs

        self.ngrams = np.arange(ngram_range[0], ngram_range[1]+1, 1)
        self.raw = None
        self.corpus = None
        self.vocabulary_ = None
        self.quiet = quiet

    def _create_ngrams(self, n, X):
        phrases = []
        for sent in X:
            words = sent.split()
            if len(words) >= n:  # Check if words list is not empty
                ngram_list = list(ngrams(words, n))
                phrases.append([" ".join(word) for word in ngram_list])
            else:
                phrases.append([])
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
        return data, corpus
    
    # def _avg_sentence_old(self, data):
    #     avg_sentences = []
    #     for sent in data:
    #         if len(sent)!=0:
    #             avg_sentence = np.mean([self.w2v.wv.get_vector(word) for word in sent
    #                                     if word in self.w2v.wv.index_to_key], axis=0)
    #         else:
    #             avg_sentence = np.zeros(self.vsize)
    #         avg_sentences.append(avg_sentence)
    #     return np.array(avg_sentences)
    
    def _avg_sentence(self, sentences):
        w2v_model = self.w2v
        avg_sentences = []
        for sentence in sentences:
            if len(sentence)!=0:
                avg_sentence = np.mean([w2v_model.wv.get_vector(word) for word in sentence
                                        if word in w2v_model.wv.key_to_index], axis=0)
            else:
                avg_sentence = np.zeros(w2v_model.vector_size)
            avg_sentences.append(avg_sentence)
        return np.array(avg_sentences)

    def fit(self, X):
        self.w2v = Word2Vec(vector_size=self.vsize, min_count=self.min_count, sg=self.sg,
                            window=self.window, workers=4, seed=self.seed, epochs=self.epochs)
        self.raw = list(X)
        
        start = time.time()
        corpus = self._create_corpus(update_train=True, X=X)[1]
        durations = time.time() - start
        if not self.quiet:
            print(f'Create corpus: Done in {int(durations//60)}m{int(durations%60)}s')
        
        start = time.time()
        self.w2v.build_vocab(corpus)
        durations = time.time() - start
        if not self.quiet:
            print(f'Build vocab: Done in {int(durations//60)}m{int(durations%60)}s')

        start = time.time()
        self.w2v.train(corpus, total_examples=self.w2v.corpus_count, epochs=self.w2v.epochs)
        durations = time.time() - start
        if not self.quiet:
            print(f'Training : Done in {int(durations//60)}m{int(durations%60)}s')

        self.vocabulary_ = self.w2v.wv.key_to_index

    def fit_transform(self, X):
        self.w2v = Word2Vec(vector_size=self.vsize, min_count=self.min_count, sg=self.sg,
                            window=self.window, workers=4, seed=self.seed, epochs=self.epochs)
        self.raw = list(X)

        start = time.time()
        data, corpus = self._create_corpus(update_train=True, X=X)
        durations = time.time() - start
        if not self.quiet:
            print(f'Create corpus: Done in {int(durations//60)}m{int(durations%60)}s')

        start = time.time()
        self.w2v.build_vocab(corpus)
        durations = time.time() - start
        if not self.quiet:
            print(f'Build vocab: Done in {int(durations//60)}m{int(durations%60)}s')

        start = time.time()
        self.w2v.train(corpus, total_examples=self.w2v.corpus_count, epochs=self.w2v.epochs)
        durations = time.time() - start
        if not self.quiet:
            print(f'Training : Done in {int(durations//60)}m{int(durations%60)}s')

        start = time.time()
        avg_sents = self._avg_sentence(data)
        durations = time.time() - start
        if not self.quiet:
            print(f'Average : Done in {int(durations//60)}m{int(durations%60)}s')

        start = time.time()
        avg_sents_sprs = scipy.sparse.csr_matrix(avg_sents)
        durations = time.time() - start
        if not self.quiet:
            print(f'Sparse : Done in {int(durations//60)}m{int(durations%60)}s')

        return avg_sents_sprs
        
    def transform(self, X):
        start = time.time()
        data = self._create_corpus(update_train=False, X=X)[0]
        durations = time.time() - start
        if not self.quiet:
            print(f'Create corpus: Done in {int(durations//60)}m{int(durations%60)}s')

        start = time.time()
        avg_sents = self._avg_sentence(data)
        durations = time.time() - start
        if not self.quiet:
            print(f'Average : Done in {int(durations//60)}m{int(durations%60)}s')

        start = time.time()
        avg_sents_sprs = scipy.sparse.csr_matrix(avg_sents)
        durations = time.time() - start
        if not self.quiet:
            print(f'Sparse : Done in {int(durations//60)}m{int(durations%60)}s')

        return avg_sents_sprs
    
    def get_feature_names_out(self):
        columns = np.array([f'component_{i+1}' for i in range(self.vsize)])
        return columns
    
if __name__ == "__main__":
    texts = ["I love eating chocolate icecream and strawberry cake",
            "Dogs are obedient"]
    w2v = AvgWord2Vec(ngram_range=(1, 2))
    data = w2v.fit_transform(texts)
    data_df = pd.DataFrame(data)
    print(data_df)