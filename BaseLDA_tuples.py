import gensim.parsing.preprocessing as gensimm
from gensim.corpora import dictionary
import numpy as np
from optparse import OptionParser
from numpy.random import multinomial as multinom_draw


def load_corpus(filename, d):
    import csv, sys, re

    # Increase max line length for csv.reader:
    max_int = sys.maxsize
    decrement = True
    while decrement:
        decrement = False
        try:
            csv.field_size_limit(max_int)
        except OverflowError:
            max_int = int(max_int/10)
            decrement = True

    docs = []
    labs = []
    labelmap = dict()
    n = 0
    pat = re.compile("[A-Z]\d{2}")
    f = open(filename, 'r')
    reader = csv.reader(f)
    for row in reader:
        doc = row[1]
        lab = row[2]
        if len(lab) > 3:
            lab = lab.split(" ")
            lab = list(filter(lambda i: pat.search(i), lab))
            lab = [x[:d] for x in lab]
            for x in lab:
                labelmap[x] = 1
        else:
            lab = lab[:d]
            labelmap[lab] = 1
            lab = [lab]
        lab = list(set(lab))
        docs.append(doc)
        labs.append(lab)
        n += 1
        print(n)
    f.close()
    print("Stemming documents ....")
    docs = gensimm.preprocess_documents(docs)
    return docs, labs, list(labelmap.keys())


class LabeledLDA(object):
    def __init__(self, docs, labs, labelset, dicti, alpha, beta):
        labelset.insert(0, 'root')
        self.labelmap = dict(zip(labelset, range(len(labelset))))
        self.K = len(self.labelmap)
        self.dicti = dicti

        self.alpha = alpha
        self.beta = beta

        self.vocab = list(dicti.values())
        self.w_to_v = dicti.token2id
        self.v_to_w = dicti.id2token

        self.labs = np.array([self.set_label(lab) for lab in labs])
        self.doc_tups = [dicti.doc2bow(x) for x in docs]

        self.D = len(docs)
        self.V = len(self.vocab)

        self.ph_hat = np.zeros((self.K, self.V), dtype=float)
        self.th_hat = np.zeros((self.D, self.K), dtype=float)
        self.perplx = []

        self.z_dn = []
        self.n_zk = np.zeros(self.K, dtype=int)
        self.n_d_k = np.zeros((self.D, self.K), dtype=int)
        self.n_k_v = np.zeros((self.K, self.V), dtype=int)

        self.docs = []
        self.freqs = []
        for d, (doc, lab) in enumerate(zip(self.doc_tups, self.labs)):
            ids, freqs = zip(*doc)
            self.docs.append(list(ids))
            self.freqs.append(list(freqs))

            ld = len(doc)
            prob = lab/lab.sum()
            zets = np.random.choice(self.K, size=ld, p=prob)
            self.z_dn.append(zets)
            for v, z, freq in zip(ids, zets, freqs):
                self.n_zk[z] += freq
                self.n_d_k[d, z] += freq
                self.n_k_v[z, v] += freq

    def set_label(self, label):
        vec = np.zeros(len(self.labelmap))
        vec[0] = 1.0
        for x in label:
            vec[self.labelmap[x]] = 1.0
        return vec

    # May be irrelevant
    def term_to_id(self, term):
        if term not in self.w_to_v:
            voca_id = len(self.vocab)
            self.w_to_v[term] = voca_id
            self.vocab.append(term)
        else:
            voca_id = self.w_to_v[term]
        return voca_id

    def training_iteration(self):
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

                prob = lab * a * (num_b/den_b)
                prob /= np.sum(prob)
                z_new = multinom_draw(1, prob).argmax()

                self.z_dn[d][n] = z_new

                self.n_k_v[z_new, v] += f
                doc_n_d_k[z_new] += f
                self.n_zk[z_new] += f

    def run_training(self, iters, thinning):
        for n in range(iters):
            self.training_iteration()
            print('Running iteration # %d ' % (n+1))
            if (n+1) % thinning == 0:
                cur_ph = self.get_phi()
                cur_th = self.get_theta()
                cur_perp = self.perplexity()
                self.perplx.append(cur_perp)
                s = n/thinning
                if s == 1:
                    self.ph_hat = cur_ph
                    self.th_hat = cur_th
                else:
                    factor = (s-1)/s
                    self.ph_hat = factor*self.ph_hat + (1/s * cur_ph)
                    self.th_hat = factor*self.th_hat + (1/s * cur_th)
                if np.any(self.ph_hat < 0):
                    raise ValueError('A negative value occurred in self.ph_hat'
                                     'while saving iteration %d ' % n)

    def prep4test(self, doc):
        doc_tups = self.dicti.doc2bow(doc)
        doc, freqs = zip(*doc_tups)

        z_dn = []
        n_dk = np.zeros(self.K, dtype=int)

        probs = self.ph_hat[:, doc]
        probs /= probs.sum(axis=0)
        for n, f in enumerate(freqs):
            prob = probs[:, n]
            while prob.sum() > 1:
                prob /= 1.0000000005
            new_z = multinom_draw(1, prob).argmax()

            z_dn.append(new_z)
            n_dk[new_z] += f
        start_state = (doc, freqs, z_dn, n_dk)
        return start_state

    def run_test(self, newdocs, it, thinning):
        nr = len(newdocs)
        th_hat = np.zeros((nr, self.K), dtype=float)
        for d, newdoc in enumerate(newdocs):
            doc, freqs, z_dn, n_dk = self.prep4test(newdoc)
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
                s = (i + 1) / thinning
                s2 = int(s)
                if s == s2:
                    this_state = n_dk / n_dk.sum()
                    if s2 == 1:
                        avg_state = this_state
                    else:
                        old = (s2 - 1) / s2 * avg_state
                        new = (1 / s2) * this_state
                        avg_state = old + new
            th_hat[d, :] = avg_state
        return th_hat

    def get_pred(self, single_th, n=5):
        labs = np.array(list(self.labelmap.keys()))
        top_tops = np.argsort(-single_th)[:n]
        top_load = np.flip(np.sort(single_th), axis=0)[:n]

        top_tops = labs[top_tops]
        return list(zip(top_tops, top_load))

    def get_preds(self, all_th, n=5):
        preds = []
        nr = all_th.shape[0]
        for d in range(nr):
            one_th = all_th[d, :]
            pred = self.get_pred(one_th, n)
            preds.append(pred)
        return preds

    def get_phi(self):
        num = self.n_k_v + self.beta
        den = self.n_zk[:, np.newaxis] + self.V * self.beta
        return num / den

    def get_theta(self):
        num = self.n_d_k + self.labs * self.alpha
        den = num.sum(axis=1)[:, np.newaxis]
        return num / den

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

        log_per = l = 0
        for doc, th in zip(self.docs, thetas):
            for w in doc:
                log_per -= np.log(np.inner(phis[:, w], th))
            l += len(doc)
        return np.exp(log_per / l)


def split_data(f="clean_fulldocs.csv", d=2):
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


def train_it(traindata, it=30, s=3, al=0.001, be=0.001):
    a, b, c = traindata
    dicti = prune_dict(a, lower=0.1, upper=0.9)
    llda = LabeledLDA(a, b, c, dicti, al, be)
    llda.run_training(it, s)
    return llda


def test_it(model, testdata, it=500, thinning=25, n=5):
    testdocs = testdata[0]
    testdocs = [[x for x in doc if x in model.vocab] for doc in testdocs]
    th_hat = model.run_test(testdocs, it, thinning)
    preds = model.get_preds(th_hat, n)
    th_hat = [[round(x, 4) for x in single_th] for single_th in th_hat]
    return th_hat, preds


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
    parser.add_option("-l", dest="low", type="float",
                      help="low limit for pruning corpus dictionary",
                      default=0.1)
    parser.add_option("-u", dest="high", type="float",
                      help="up limit for pruning corpus dictionary",
                      default=0.9)
    (options, args) = parser.parse_args()
    if not options.filename:
        parser.error("need to supply csv-data file location (-f)")

    a, b, c = load_corpus(options.filename, d=options.depth)
    dicti = prune_dict(a, lower=options.low, upper=options.high)
    llda = LabeledLDA(docs=a, labs=b, labelset=c, dicti=dicti,
                      alpha=options.alpha, beta=options.beta)

    for i in range(options.iterations):
        print("Iteration #: %d " % (i + 1))
        llda.training_iteration()

if __name__ == "__main__":
    main()
