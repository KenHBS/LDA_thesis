from gensim.corpora import dictionary
from gensim.parsing import preprocessing

from helpers import get_pruned_g_dict

import numpy as np
from numpy.random import multinomial as multinom_draw

from typing import List, Dict, Union, Tuple


class LabeledLDA:
    def __init__(
        self,
        *,
        docs: List[str],
        labels: List[List[str]],
        alpha: float,
        beta: float,
        preprocessed: bool = False
    ):
        if not preprocessed:
            docs = preprocessing.preprocess_documents(docs)

        self.g_dict = get_pruned_g_dict(docs)

        label_space = {lab for labs in labels for lab in labs if lab != ""}
        n_labs = label_space

        self.label_mapping = dict(zip(label_space, range(1, n_labs)))
        self.label_mapping["root"] = 0

        self.K = len(self.label_mapping)

        self.alpha = alpha
        self.beta = beta

        self.vocab = list(self.g_dict.values())
        self.w_to_v = self.g_dict.token2id
        self.v_to_w = self.g_dict.id2token

        self.labs = np.array([self.dummify_label(lab) for lab in labs])

        self.D = len(docs)
        self.V = len(self.vocab)

        self.ph_hat = np.zeros((self.K, self.V), dtype=float)
        self.th_hat = np.zeros((self.D, self.K), dtype=float)
        self.cur_perplexity = []

        self.z_dn = []
        self.n_zk = np.zeros(self.K, dtype=int)
        self.n_d_k = np.zeros((self.D, self.K), dtype=int)
        self.n_k_v = np.zeros((self.K, self.V), dtype=int)

        self.doc_tups = [self.g_dict.doc2bow(doc) for doc in docs]

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
                cur_ph = self.get_phi()
                cur_th = self.get_theta()

                current_perplexity = self.perplexity()
                self.cur_perplexity.append(current_perplexity)

                save_count = (n + 1) / thinning
                if save_count == 1:
                    self.ph_hat = cur_ph
                    self.th_hat = cur_th

                elif save_count > 1:
                    factor = (save_count - 1) / save_count
                    fraction = 1 / save_count
                    self.ph_hat = factor * self.ph_hat + (fraction * cur_ph)
                    self.th_hat = factor * self.th_hat + (fraction * cur_th)

                any_negative_phi = (self.ph_hat < 0).any()
                if any_negative_phi:
                    msg = "A negative value occured in self.ph_hat while \
                    saving iteration # {}".format(
                        n
                    )
                    raise ValueError(msg)

                any_missing_phi = np.isnan(self.ph_hat).any()
                if any_missing_phi:
                    msg = "A missing value has creeped into ph_hat"
                    raise ValueError(msg)

                wordloads = self.ph_hat.sum(axis=0)
                if (wordloads == 0).any():
                    msg = "A word in dictionary has no z-value"
                    raise ValueError(msg)

    def initiate_new_document(
        self, doc: List[str], preprocessed: bool = True
    ) -> Tuple[List[int], List[int], List[int], np.array]:
        """ Prep document and initiate topic assignments for unseen doc """
        if not preprocessed:
            doc = preprocessing.preprocess_document(doc)

        doc_tups = self.g_dict.doc2bow(doc)
        doc, freqs = zip(*doc_tups)

        z_dn = []
        n_dk = np.zeros(self.K, dtype=int)

        probs = self.ph_hat[:, doc]
        with np.errstate(divide="raise", invalid="raise"):
            try:
                probs /= probs.sum(axis=0)
            except FloatingPointError:
                probs = 1 / self.K * np.ones_like(probs)
        for n, f in enumerate(freqs):
            prob = probs[:, n]
            while prob.sum() > 1:
                prob /= 1.0000000005
            new_z = multinom_draw(1, prob).argmax()

            z_dn.append(new_z)
            n_dk[new_z] += f
        start_state = (doc, freqs, z_dn, n_dk)
        return start_state

    def run_test(self, newdocs: List[List[str]], it: int, thinning: int) -> np.array:
        """ Fit theta on unseen documents """
        nr = len(newdocs)
        th_hat = np.zeros((nr, self.K), dtype=float)

        for d, newdoc in enumerate(newdocs):
            doc, freqs, z_dn, n_dk = self.initiate_new_document(newdoc)

            for i in range(it):
                for n, (v, f, z) in enumerate(zip(doc, freqs, z_dn)):
                    n_dk[z] -= f

                    num_a = n_dk + self.alpha
                    b = self.ph_hat[:, v]

                    prob = num_a * b
                    prob /= prob.sum()
                    while prob.sum() > 1:
                        prob /= 1.0000005

                    new_z = multinom_draw(1, prob).argmax()

                    z_dn[n] = new_z
                    n_dk[new_z] += f

                # Save the current state in MC chain and calc. average state:
                # Only the document-topic distribution estimate theta is saved
                if (n + 1) % thinning == 0:
                    save_count = (n + 1) / thinning

                    this_state = n_dk / n_dk.sum()
                    if save_count == 1:
                        avg_state = this_state
                    elif save_count > 1:
                        factor = (save_count - 1) / save_count
                        fraction = 1 / save_count

                        old = factor * avg_state
                        new = fraction * this_state
                        avg_state = old + new

                    th_hat[d, :] = avg_state
        return th_hat

    def get_prediction(
        self, single_th: np.array, n: int = 5
    ) -> List[Tuple[Union[str, float]]]:
        """ Return the top n topics from a fitted theta vector """
        possible_labels = np.array(list(self.label_mapping.keys()))

        top_n_inds = np.argsort(-single_th)[:n]
        top_n_loadings = np.flip(np.sort(single_th), axis=0)[:n]

        top_n_topics = possible_labels[top_n_inds]
        return list(zip(top_n_topics, top_n_loadings))

    def get_predictions(
        self, all_th: np.array, n: int = 5
    ) -> List[List[Tuple[Union[str, float]]]]:
        """ Return the top n topic with loadings from fitted theta matrix """
        predictions = []
        nr = all_th.shape[0]
        for d in range(nr):
            one_th = all_th[d, :]
            prediction = self.get_prediction(one_th, n)

            predictions.append(prediction)
        return predictions

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

        ph = self.get_phi()
        topic_list = []
        for k in range(self.K):
            v_inds = np.argsort(-ph[k, :])[:n]
            top_n = [self.v_to_w[x] for x in v_inds]

            topic_name = possible_labels[k]
            top_n.insert(0, topic_name)

            topic_list += [top_n]
        return topic_list

    def perplexity(self) -> float:
        """ Return perplexity of current state """
        phis = self.get_phi()
        thetas = self.get_theta()

        log_per = 0
        l = 0
        for doc, th in zip(self.docs, thetas):
            for w in doc:
                log_per -= np.log(np.inner(phis[:, w], th))
            l += len(doc)
        return np.exp(log_per / l)


def split_data(f, d=2):
    a, b, c = load_corpus(f, d)

    zipped = list(zip(a, b))
    np.random.shuffle(zipped)
    a, b = zip(*zipped)

    split = int(len(a) * 0.9)
    train_data = (a[:split], b[:split], c)
    test_data = (a[split:], b[split:], c)
    return train_data, test_data


def prune_dict(docs, lower=0.1, upper=0.9):
    dicti = dictionary.Dictionary(docs)
    lower *= len(docs)
    dicti.filter_extremes(no_above=upper, no_below=lower)
    return dicti


def train_it(traindata, it=30, s=3, al=0.001, be=0.001, l=0.05, u=0.95):
    a, b, c = traindata
    dicti = prune_dict(a, lower=l, upper=u)
    llda = LabeledLDA(a, b, c, dicti, al, be)
    llda.run_training(it, s)
    return llda


def test_it(model, testdata, it=500, thinning=25, n=5):
    testdocs = testdata[0]
    testdocs = [[x for x in doc if x in model.vocab] for doc in testdocs]
    th_hat = model.run_test(testdocs, it, thinning)
    preds = model.get_predictions(th_hat, n)
    th_hat = [[round(x, 4) for x in single_th] for single_th in th_hat]
    return th_hat, preds
