import numpy as np
import pandas as pd
import gensim
import re

from numpy.random import multinomial as multinom_draw
from gensim.parsing.preprocessing import STOPWORDS as stopwords
from nltk.stem import WordNetLemmatizer


class LocalLDA:
    def __init__(self, docs, alpha, beta, K,
                 localLDA=True, lemma=True, stem=False):
        self.a = alpha
        self.b = beta

        if localLDA:
            sentences = []
            for doc in docs:
                s = splitdocs(doc)
                sentences.extend(s)
            docs = sentences

        # Preprocess the documents, create word2id mapping & map words to IDs
        prepped_corp = prep_docs(docs, stem=stem, lemma=lemma)
        self.word2id = gensim.corpora.dictionary.Dictionary(prepped_corp)
        self.doc_tups = [self.word2id.doc2bow(doc) for doc in prepped_corp]
        self.doc_tups = [doc for doc in self.doc_tups if len(doc) > 1]

        # Gather some general LDA parameters
        self.V = len(self.word2id)
        self.K = K
        self.D = len(self.doc_tups)

        self.w_to_v = self.word2id.token2id
        self.v_to_w = self.word2id

        self.z_dn = []
        self.n_zk = np.zeros(self.K, dtype=int)
        self.n_d_k = np.zeros((self.D, self.K), dtype=int)
        self.n_k_v = np.zeros((self.K, self.V), dtype=int)

        self.docs = []
        self.freqs = []
        for d, doctup in enumerate(self.doc_tups):
            ids, freqs = zip(*doctup)
            self.docs.append(list(ids))
            self.freqs.append(list(freqs))

            zets = np.random.choice(self.K, self.K)
            self.z_dn.append(zets)
            for v, z, freq in zip(ids, zets, freqs):
                self.n_zk[z] += freq
                self.n_d_k[d, z] += freq
                self.n_k_v[z, v] += freq

        self.th_hat = None   # will be filled during training
        self.ph_hat = None   # will be filled during training

    def training_iteration(self):
        docs = self.docs
        freqs = self.freqs

        zdn = self.z_dn
        for d, (doc, freq, zet) in enumerate(zip(docs, freqs, zdn)):
            doc_n_d_k = self.n_d_k[d]
            for n, (v, f, z) in enumerate(zip(doc, freq, zet)):
                self.n_k_v[z, v] -= f
                doc_n_d_k[z] -= f
                self.n_zk[z] -= f

                a = doc_n_d_k + self.a
                num_b = self.n_k_v[:, v] + self.b
                den_b = self.n_zk + self.V * self.b

                prob = a * (num_b / den_b)
                prob /= np.sum(prob)
                z_new = multinom_draw(1, prob).argmax()

                self.z_dn[d][n] = z_new

                self.n_k_v[z_new, v] += f
                doc_n_d_k[z_new] += f
                self.n_zk[z_new] += f

    def run_training(self, iters, thinning):
        for n in range(iters):
            self.training_iteration()
            print('Running iteration # %d ' % (n + 1))
            if (n + 1) % thinning == 0:
                cur_ph = self.get_phi()
                cur_th = self.get_theta()

                s = (n + 1) / thinning
                if s == 1:
                    self.ph_hat = cur_ph
                    self.th_hat = cur_th
                elif s > 1:
                    factor = (s - 1) / s
                    self.ph_hat = factor * self.ph_hat + (1 / s * cur_ph)
                    self.th_hat = factor * self.th_hat + (1 / s * cur_th)
                if np.any(self.ph_hat < 0):
                    raise ValueError('A negative value occurred in self.ph_hat'
                                     'while saving iteration %d ' % n)
                if np.any([np.isnan(x) for x in self.ph_hat]):
                    raise ValueError('A nan has creeped into ph_hat')
                wordload = self.ph_hat.sum(axis=0)
                if np.any([x == 0 for x in wordload]):
                    raise ValueError('A word in dictionary has no z-value')

    def get_phi(self):
        num = self.n_k_v + self.b
        den = self.n_zk[:, np.newaxis] + self.V * self.b
        return num / den

    def get_theta(self):
        num = self.n_d_k + self.a
        den = num.sum(axis=1)[:, np.newaxis]
        return num / den

    def print_topwords(self, n=10):
        ph = self.get_phi()
        topiclist = []
        for k in range(self.K):
            v_ind = np.argsort(-ph[k, :])[:n]
            top_n = [self.v_to_w[x] for x in v_ind]
            top_n.insert(0, str(k))
            topiclist += [top_n]
        print(topiclist)
        pass


def prep_docs(docs, stem=False, lemma=True):
    return [prep_doc(doc, stem=stem, lemma=lemma) for doc in docs]


def prep_doc(doc, stem=False, lemma=True):
    doc = doc.lower()
    doc = re.sub('[^\w\s]', '', doc)
    doc = doc.split()
    # remove stopwords and short words
    doc = [word for word in doc if word not in stopwords and len(word) > 2]

    if stem:
        p = gensim.parsing.PorterStemmer()
        return [p.stem(word) for word in doc]
    elif lemma:
        lm = WordNetLemmatizer()
        return [lm.lemmatize(word, pos='v') for word in doc]
    else:
        return doc


def splitdocs(doc):
    sentences = re.split('!|\.|\?|,|-|', doc)
    return sentences
