import gensim.parsing.preprocessing as gensimm
import numpy as np
from optparse import OptionParser
from numpy.random import multinomial as multinom_draw

def load_corpus(filename, d):
    import csv
    docs = []
    labs = []
    labelmap = dict()
    f = open(filename, 'r')
    reader = csv.reader(f)
    for row in reader:
        doc = row[1]
        lab = row[2]
        if len(lab) > 1:
            lab = lab.split(" ")
            [x[:d] for x in lab]
            for x in lab:
                labelmap[x] = 1
        else:
            lab = lab[1:d]
        docs.append(doc)
        labs.append(lab)
    docs = gensimm.preprocess_documents(docs)
    f.close()
    return docs, labs, list(labelmap.keys())


class LabeledLDA(object):
    def __init__(self, docs, labs, labelset, alpha, beta):
        labelset.insert(0, 'root')
        self.labelmap = dict(zip(labelset, range(len(labelset))))
        self.K = len(self.labelmap)

        self.alpha = alpha
        self.beta = beta

        self.vocab = []
        self.w_to_v = dict()
        self.labs = np.array([self.set_label(lab) for lab in labs])
        self.docs = [[self.term_to_id(term) for term in doc] for doc in docs]
        self.v_to_w = {v:k for k, v in self.w_to_v.items()}

        self.D = len(docs)
        self.V = len(self.vocab)

        self.z_dn = []
        self.n_d_k = np.zeros((self.D, self.K), dtype=int)
        self.n_k_v = np.zeros((self.K, self.V), dtype=int)
        self.n_zk = np.zeros(self.K, dtype=int)

        self.ph_hat = np.zeros((self.K, self.V), dtype=float)
        self.th_hat = np.zeros((self.D, self.K), dtype=float)
        self.perplx = []

        for d, doc, lab in zip(range(self.D), self.docs, self.labs):
            len_d = len(doc)
            zets = [np.random.multinomial(1, lab/lab.sum()).argmax() for x in range(len_d)]
            self.z_dn.append(zets)
            for v, z in zip(doc, zets):
                self.n_d_k[d, z] += 1
                self.n_k_v[z, v] += 1
                self.n_zk[z] += 1

    def set_label(self, label):
        vec = np.zeros(len(self.labelmap))
        vec[0] = 1.0
        for x in label:
            vec[self.labelmap[x]] = 1.0
        return vec

    def term_to_id(self, term):
        if term not in self.w_to_v:
            voca_id = len(self.vocab)
            self.w_to_v[term] = voca_id
            self.vocab.append(term)
        else:
            voca_id = self.w_to_v[term]
        return voca_id

    def training_iteration(self):
        for d, doc, lab in zip(range(self.D), self.docs, self.labs):
            len_d = len(doc)
            for n in range(len_d):
                v = doc[n]
                z = self.z_dn[d][n]
                self.n_d_k[d, z] -= 1
                self.n_k_v[z, v] -= 1
                self.n_zk[z] -= 1

                numer_a = self.n_d_k[d] + self.alpha
                denom_a = len_d - 1 + self.K * self.alpha
                numer_b = self.n_k_v[:, v] + self.beta
                denom_b = self.n_zk + self.V * self.beta

                prob = lab * (numer_a/denom_a) * (numer_b/denom_b)
                prob /= np.sum(prob)
                z_new = multinom_draw(1, prob).argmax()

                self.z_dn[d][n] = z_new
                self.n_d_k[d, z_new] += 1
                self.n_k_v[z_new, v] += 1
                self.n_zk[z_new] += 1

    def run_training(self, iters, thinning):
        for n in range(iters):
            self.training_iteration()
            print('Running iteration # %d ' % (n+1))
            if (n+1) % thinning == 0:
                cur_phi = self.get_phi()
                cur_th = self.get_theta()
                cur_perp = self.perplexity()
                self.perplx.append(cur_perp)
                s = n/thinning
                if s == 1:
                    self.ph_hat = cur_phi
                    self.th_hat = cur_th
                else:
                    factor = (s-1)/s
                    self.ph_hat = factor*self.ph_hat + (1/s * cur_phi)
                    self.th_hat = factor*self.th_hat + (1/s * cur_th)
                if np.any(self.ph_hat < 0):
                    raise ValueError('A negative value occurred in self.ph_hat'
                                     'while saving iteration %d ' % n)

    def test_init(self, doc):
        doc = [self.w_to_v[term] for term in doc]
        len_d = len(doc)
        z_dn = []
        n_zk = np.zeros(self.K, dtype=int)
        probs = self.ph_hat[:, doc]
        probs /= probs.sum(axis=0)
        for n in range(len_d):
            prob = probs[:, n]
            while prob.sum() > 1:
                prob /= 1.0000000005
            new_z = multinom_draw(1, prob).argmax()

            z_dn.append(new_z)
            n_zk[new_z] += 1
        start_state = (doc, z_dn, n_zk)
        return start_state

    def run_test(self, testdocs, iters, thinning):
        nrdocs = len(testdocs)
        theta_hat = np.zeros((nrdocs, self.K), dtype=float)
        for d, testdoc in zip(range(nrdocs), testdocs):
            doc, z_dn, n_zk = self.test_init(testdoc)
            len_d = len(z_dn)
            for i in range(iters):
                for n in range(len_d):
                    v = doc[n]
                    z = z_dn[n]
                    n_zk[z] -= 1

                    numer_a = n_zk + self.alpha
                    denom_a = len_d - 1 + self.K * self.alpha
                    b = self.ph_hat[:, v]
                    prob = (numer_a / denom_a) * b
                    prob /= prob.sum()
                    while prob.sum() > 1:
                        prob /= 1.0000005
                    new_z = multinom_draw(1, prob).argmax()

                    z_dn[n] = new_z
                    n_zk[new_z] += 1

                # Save the current state in MC chain and calc. average state:
                # Only the document-topic distribution estimate theta is saved
                s = (i + 1) / thinning
                s2 = int(s)
                if s == s2:
                    this_state = n_zk / n_zk.sum()
                    if s2 == 1:
                        avg_state = this_state
                    else:
                        old = (s2 - 1) / s2 * avg_state
                        new = (1 / s2) * this_state
                        avg_state = old + new
            theta_hat[d, :] = avg_state
        return theta_hat

    def get_prediction(self, single_th, n=5):
        labels = np.array(list(self.labelmap.keys()))
        best_topic_inds = np.argsort(-single_th)[:n]
        best_topic_load = np.flip(np.sort(single_th), axis=0)[:n]

        top_topics = labels[best_topic_inds]
        return list(zip(top_topics, best_topic_load))

    def get_predictions(self, all_th, n=5):
        predictions = []
        ndocs = all_th.shape[0]
        for d in range(ndocs):
            single_th = all_th[d, :]
            pred = self.get_prediction(single_th, n)
            predictions.append(pred)
        return predictions

    def get_phi(self):
        numer = self.n_k_v + self.beta
        denom = self.n_zk[:, np.newaxis] + self.V * self.beta
        return numer / denom

    def get_theta(self):
        numer = self.n_d_k + self.labs * self.alpha
        denom = numer.sum(axis=1)[:, np.newaxis]
        return numer / denom

    def topwords_per_topic(self, topwords=10):
        n = topwords
        ph = self.get_phi()
        topiclist = []
        label_list = list(self.labelmap.keys())
        for k in range(self.K):
            v_inds = np.argsort(-ph[k, :])[:n]
            top_n = [self.v_to_w[x] for x in v_inds]

            topic_name = label_list[k]
            top_n.insert(0, topic_name)

            topiclist += [top_n]
        return topiclist

    def perplexity(self):
        phis = self.get_phi()
        thetas = self.get_theta()

        log_per = N = 0
        for doc, th in zip(self.docs, thetas):
            for w in doc:
                log_per -= np.log(np.inner(phis[:, w], th))
            N += len(doc)
        return np.exp(log_per / N)

def run_it(f="thesis_data.csv", d=3, it=30, thinning=3, al=0.001, be=0.001):
    a, b, c = load_corpus(f, d)
    llda = LabeledLDA(a, b, c, al, be)
    llda.run_training(it, thinning)
    return llda

def main():
    parser = OptionParser()
    parser.add_option("-f", dest="filename", help="csv-data filename")
    parser.add_option("--alpha", dest="alpha", type="float",
                      help="hyperprior alpha", default=0.001)
    parser.add_option("--beta", dest="beta", type="float",
                      help="hyperprior beta", default=0.001)
    parser.add_option("--iters", dest="iterations", type="int",
                      help=" # of training iterations", default=250)
    parser.add_option("-n", dest="topwords", type="int",
                      help="topwords per topic", default=10)
    parser.add_option("-d", dest="depth", type="int",
                      help="depth of label", default=3)
    (options, args) = parser.parse_args()
    if not options.filename:
        parser.error("need to supply csv-data file location (-f)")

    a, b, c = load_corpus(options.filename, d=options.depth)
    llda = LabeledLDA(docs=a, labs=b, labelset=c,
                      alpha=options.alpha, beta=options.beta)

    for i in range(options.iterations):
        print("Iteration #: %d " % (i + 1))
        llda.training_iteration()

if __name__ == "__main__":
    main()