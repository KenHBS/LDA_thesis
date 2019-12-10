import gensim

import pandas as pd
import numpy as np

from numpy.random import multinomial as multinom_draw

from typing import List, Dict, Tuple, Optional


class BaseLDA:
    def __init__(self,
                 *,
                 doc_tuples: List[List[Tuple[int]]],
                 vocabulary: gensim.corpora.dictionary.Dictionary,
                 alpha: float,
                 beta: float,
                 labs: Optional[List[List[str]]]=None,
                 K: Optional[int]=None):
        self.vocabulary = vocabulary
        self.doc_tups = doc_tuples

        self.alpha = alpha
        self.beta = beta

        self.D = len(self.doc_tups)
        self.V = len(self.vocabulary)

        if labs:
            self.label_mapping = self.generate_label_mapping(labs)
            self.labs = np.array([self.dummify_label(lab) for lab in labs])
            self.K = len(self.label_mapping)
        else:
            print('No labels provided, running unsupervised LDA..')
            self.label_mapping = dict(zip(range(K), range(K)))
            self.K = K
            self.labs = np.ones((self.D, self.K), dtype=int)

        # Initiate theta, phi, z, w and helper counters:
        self.ph_hat = np.zeros((self.K, self.V), dtype=float)
        self.th_hat = np.zeros((self.D, self.K), dtype=float)
        self.cur_perplexity = []  # Currently unused

        self.z_dn = []
        self.n_zk = np.zeros(self.K, dtype=int)
        self.n_d_k = np.zeros((self.D, self.K), dtype=int)
        self.n_k_v = np.zeros((self.K, self.V), dtype=int)

        # Initialise word to topic assignments
        self.docs = []
        self.freqs = []
        for d, (doc, lab) in enumerate(zip(self.doc_tups, self.labs)):

            ids, freqs = zip(*doc)
            self.docs.append(list(ids))
            self.freqs.append(list(freqs))

            ld = len(doc)
            prob = lab / lab.sum()
            zets = np.random.choice(self.K, size=ld, p=prob)
            self.z_dn.append(zets)
            for v, z, freq in zip(ids, zets, freqs):
                self.n_zk[z] += freq
                self.n_d_k[d, z] += freq
                self.n_k_v[z, v] += freq

        pass

    @staticmethod
    def generate_label_mapping(labels: List[List[str]]) -> Dict[str, int]:
        """ Return label-to-integer mapping dictionary """
        label_space = {lab for labs in labels for lab in labs if lab != ''}
        n = len(label_space) + 1

        mapping = dict(zip(label_space, range(1, n)))
        mapping['root'] = 0
        return mapping

    def dummify_label(self, label: List[str]) -> np.ndarray:
        """ Create a K-length dummy vector """
        vec = np.zeros(len(self.label_mapping))
        vec[0] = 1.0
        for x in label:
            vec[self.label_mapping[x]] = 1.0
        return vec

    def training_iteration(self) -> None:
        """ Re-assign word assignments in training iteration """
        docs = self.docs
        freqs = self.freqs
        zdn = self.z_dn
        labs = self.labs

        for d, (doc, freq, zet, lab) in enumerate(zip(docs, freqs, zdn, labs)):
            doc_n_d_k = self.n_d_k[d]
            for n, (v, f, z) in enumerate(zip(doc, freq, zet)):
                self.n_k_v[z, v] -= f
                doc_n_d_k[z] -= f
                self.n_zk[z] -= f

                a = doc_n_d_k + self.alpha
                num_b = self.n_k_v[:, v] + self.beta
                den_b = self.n_zk + self.V * self.beta

                prob = lab * a * (num_b / den_b)
                prob /= np.sum(prob)
                z_new = multinom_draw(1, prob).argmax()

                self.z_dn[d][n] = z_new

                self.n_k_v[z_new, v] += f
                doc_n_d_k[z_new] += f
                self.n_zk[z_new] += f
        pass

    def run_training(self, iters: int, thinning: int) -> None:
        """ Run training iterations with intermediate savings """
        for n in range(iters):
            self.training_iteration()
            print("Running iteration # {}".format(n + 1))

            if (n + 1) % thinning == 0:
                self._save_training_iteration(n, thinning)

                self._validate_current_state(n)
        pass

    def _save_training_iteration(self, n: int, thinning: int) -> None:
        """ Update ph_hat and th_hat with the current state """
        cur_ph = self.get_phi()
        cur_th = self.get_theta()
        
        save_count = (n + 1) / thinning
        if save_count == 1:
            self.ph_hat = cur_ph
            self.th_hat = cur_th

        elif save_count > 1:
            factor = (save_count - 1) / save_count
            fraction = 1 / save_count
            self.ph_hat = factor * self.ph_hat + (fraction * cur_ph)
            self.th_hat = factor * self.th_hat + (fraction * cur_th)

        pass

    def _validate_current_state(self, i: int) -> None:
        """ Validate that all matrices are well-filled """ 
        std_msg = f"while saving iteration # {i}"
        any_negative_phi = (self.ph_hat < 0).any()
        if any_negative_phi:
            msg = f"A negative value occured in ph_hat {std_msg}"
            raise ValueError(msg)

        any_missing_phi = np.isnan(self.ph_hat).any()
        if any_missing_phi:
            msg = f"A missing value has creeped into ph_hat {std_msg}"
            raise ValueError(msg)

        wordloads = self.ph_hat.sum(axis=0)
        any_empty_topics = (wordloads == 0).any()
        if any_empty_topics:
            msg = f"A word in dictionary has no single z-assignment {std_msg}"
            raise ValueError(msg)

    def get_phi(self) -> np.array:
        """ Return new phi matrix (Gibbs sampling step?) """
        num = self.n_k_v + self.beta
        den = self.n_zk[:, np.newaxis] + self.V * self.beta

        return num / den

    def get_theta(self) -> np.array:
        """ Return new theta matrix (Gibbs sampling step?) """
        num = self.n_d_k + self.labs * self.alpha
        den = num.sum(axis=1)[:, np.newaxis]

        return num / den

    def topwords_per_topic(self, n: int = 10) -> List[List[List[str]]]:
        """ For all topics, return the words with highest loading """
        possible_labels = list(self.label_mapping.keys())
        #v_to_w = {k:v for k, v in self.vocabulary.items()}

        ph = self.get_phi()
        topic_list = []
        for k in range(self.K):
            v_inds = np.argsort(-ph[k, :])[:n]
            top_n = [self.vocabulary[x] for x in v_inds]

            topic_name = possible_labels[k]
            top_n.insert(0, topic_name)

            topic_list += [top_n]
    
        return topic_list
