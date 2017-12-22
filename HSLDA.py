import gensim.parsing.preprocessing as gensimm
import numpy as np
from scipy.stats import truncnorm
from scipy.special import erf
multinom_draw = np.random.multinomial
rvs = truncnorm.rvs


def partition_label(lab, d):
    return [lab[:i+1] for i in range(d)]


def phi(x):
    return 1/2 * (1 + erf(x/np.sqrt(2)))


def load_corpus(filename, d=3):
    import csv
    docs = []
    labs = []
    labelmap = dict()
    f = open(filename, 'r')
    reader = csv.reader(f)
    for row in reader:
        doc = row[1]
        lab = row[2]
        if len(lab) > 3:
            lab = lab.split(" ")
            lab = [partition_label(x, d) for x in lab]
            lab = [item for sublist in lab for item in sublist]
            lab = list(set(lab))
            for x in lab:
                labelmap[x] = 1
        else:
            lab = partition_label(lab, d)
            for x in lab:
                labelmap[x] = 1
        docs.append(doc)
        labs.append(lab)
    docs = gensimm.preprocess_documents(docs)
    f.close()
    return docs, labs, list(labelmap.keys())


class HSLDA(object):
    def __init__(self, docs, labs, labelset, K=15,
                 alpha_prime=1, alpha=1, gamma=1, mu=0, sigma=1, xi=0):
        labelset.insert(0, '')
        self.labelmap = dict(zip(labelset, range(len(labelset))))

        self.aprime = alpha_prime
        self.alpha = alpha
        self.gamma = gamma
        self.mu = mu
        self.sigma = sigma
        self.xi = xi
        self.K = K

        self.vocab = []
        self.w_to_v = dict()
        self.labs = np.array([self.set_label(lab) for lab in labs])
        self.docs = [[self.term_to_id(term) for term in doc] for doc in docs]
        self.v_to_w = {v:k for k, v in self.w_to_v.items()}

        self.D = len(docs)
        self.L = len(self.labelmap)
        self.V = len(self.vocab)

        k_ones = np.repeat(1, self.K)
        v_ones = np.repeat(1, self.V)
        mu_par = self.mu * k_ones
        self.eta = np.random.normal(mu_par, 1, size=(self.L, self.K))
        self.beta = np.random.dirichlet(self.aprime * k_ones)
        self.ph = np.random.dirichlet(self.gamma * v_ones, size=self.K)
        self.th = np.random.dirichlet(self.beta * self.alpha, size=self.D)

        self.z_dn = []
        self.n_d_k = np.zeros((self.D, self.K), dtype=int)
        self.n_k_v = np.zeros((self.K, self.V), dtype=int)
        self.n_zk = np.zeros(self.K, dtype=int)

        for d, doc in enumerate(self.docs):
            len_d = len(doc)
            prob = self.th[d, :]
            zets = np.random.choice(self.K, size=len_d, p=prob)
            self.z_dn.append(zets)
            for v, z in zip(doc, zets):
                self.n_d_k[d, z] += 1
                self.n_k_v[z, v] += 1
                self.n_zk[z] += 1

        self.zbar = self.get_zbar()
        self.mean_a = np.dot(self.zbar, self.eta.T)

        border_left = np.where(self.labs == 1, -self.mean_a, -np.inf)
        border_right = np.where(self.labs == 1, np.inf, -self.mean_a)
        self.a = rvs(border_left, border_right, self.mean_a)

        parents = [x[:-1] for x in labelset]
        parents = [self.labelmap[x] for x in parents]
        own = [self.labelmap[x] for x in labelset]
        self.child_to_parent = dict(zip(own, parents))

    def get_zbar(self):
        return self.n_d_k / self.n_d_k.sum(axis=1, keepdims=True)

    def get_ph(self):
        return self.n_k_v / self.n_k_v.sum(axis=1, keepdims=True)

    def set_label(self, label):
        l = len(self.labelmap)
        vec = np.zeros(l, dtype=int)
        vec[0] = 1
        for x in label:
            vec[self.labelmap[x]] = 1
        return vec

    def term_to_id(self, term):
        if term not in self.w_to_v:
            voca_id = len(self.vocab)
            self.w_to_v[term] = voca_id
            self.vocab.append(term)
        else:
            voca_id = self.w_to_v[term]
        return voca_id

    def sample_z(self, opt=1):
        """
        Draws new values for all word-topic assignments in the corpus, based on
        Eq. (1) in Perotte '11 HSLDA paper. Two variations have been added
        for mathematical and theoretical precision and comparison
        (see :param opt below).
        This function contains two loops: the outer loop collects doc-level
        data from the HSLDA-object to avoid lengthy and superfluous computation
        The inner loop uses those subsets to first deduct the current token's
        topic assignment in all relevant subsets, then calculate probabilities
        for k = 1, 2, ... K and then draw a random values, based on those probs
        opt=1 stands for Eq. (1) as presented in the paper.

        val_a:  L' x 1 np.array(floats):
                        The values of the running variable a. Only the
                        relevant values for document d are used here
        mean_a: L' x 1 np.array(floats):
                        The mean of the running variable a. That is,
                        np.dot(zbar.T, eta).
        dif_mean: L' x K np.array(floats):
                        This is the reduction in mean_a, due to new topic
                        assignment z_{d,n}. This implicitly affects zbar, then
                        np.dot(zbar, eta), which is mean_a. Every column
                        represents the hypothetical change in mean_a caused
                        by a reassignment of topic k.

        labs:   L x 1 np.array(binary):
                        An L-dimensional vector with zeros and ones,
                        indicating whether label l is part of document d's
                        labelset, or not
        relevant_labs: L' x 1 np.array(int):
                        Vector containing the label ID of the labels in
                        document d's labelset


        :param opt: 1 calculates p(a_{l,d} = x) for l positive labels only
                    2 calculates p(a_{l,d} > 0) for l positive labels only
                    3 calculates p(a_{l',d} > 0) for all l' positive label and
                                 p(a_{l'', d} < 0) for all l'' negative label
        :return: K-dimensional probability vector
        """
        for d, doc in enumerate(self.docs):

            # Identify the labelset of document doc:
            labs = self.labs[d]
            if opt in [1, 2]:
                relevant_labs = np.where(labs == 1)[0]
            elif opt == 3:
                relevant_labs = range(self.L)

            # Select relevant data subsets in outer loop
            z_dn = self.z_dn[d]
            n_d_k = self.n_d_k[d, :]
            eta = self.eta[relevant_labs, :]
            val_a = self.a[d, relevant_labs, np.newaxis]
            mean_a = self.mean_a[d, relevant_labs, np.newaxis]

            # Calculate the implicit update of a's mean.
            n_d = len(doc)
            dif_mean = eta / n_d
            means_a = mean_a + dif_mean
            for n, v in enumerate(doc):
                # Find and deduct the word-topic assignment:
                old_z = z_dn[n]
                means_a[:, old_z] -= dif_mean[:, old_z]
                n_d_k[old_z] -= 1
                self.n_k_v[old_z, v] -= 1
                self.n_zk[old_z] -= 1

                # Calculate probability of first part of Eq. (1)
                l = n_d_k + self.alpha * self.beta
                r_num = self.n_k_v[:, v] + self.gamma
                r_den = self.n_zk + self.V * self.gamma
                p1 = l * r_num / r_den

                # Calculate probability of second part of Eq. (1)
                if opt == 1:
                    p2 = np.exp((means_a - val_a) ** 2 * (-1 / 2))
                elif opt in [2, 3]:
                    labcheck = labs[relevant_labs]
                    labcheck = labcheck[:, np.newaxis]
                    means_a -= self.xi
                    signed_mean = np.where(labcheck == 1, means_a, -means_a)
                    p2 = phi(signed_mean)
                p2 *= 2
                p2 = p2.prod(axis=0)

                # Combine two parts and draw new word-topic assignment z_{d,n}
                prob = p1 * p2
                prob /= prob.sum()
                new_z = multinom_draw(1, prob).argmax()

                # Add back z_new to all relevant containers:
                z_dn[n] = new_z
                means_a[:, new_z] += dif_mean[:, new_z]
                n_d_k[new_z] += 1
                self.n_k_v[new_z, v] += 1
                self.n_zk[new_z] += 1
            self.n_d_k[d, :] = n_d_k
            self.z_dn[d] = z_dn
            self.zbar[d, :] = n_d_k / n_d
        self.mean_a = np.dot(self.zbar, self.eta.T)

    def sample_eta(self):
        sig_prior = np.identity(self.K) / self.sigma
        sig_data = np.dot(self.zbar.T, self.zbar)
        sigma_hat = scipy.linalg.inv(sig_prior + sig_data)

        mu_prior = self.mu / self.sigma
        mu_data = np.dot(self.zbar.T, self.a)
        raw_mean = mu_prior + mu_data
        mu_hat = np.dot(sigma_hat, raw_mean)

        for l in range(self.L):
            mu = mu_hat[:, l]
            eta_l = np.random.multivariate_normal(mu, sigma_hat)
            self.eta[l, :] = eta_l

    def sample_a(self):
        border_left = np.where(self.labs > 0, -self.mean_a, -np.inf)
        border_right = np.where(self.labs > 0, np.inf, -self.mean_a)
        self.a = rvs(border_left, border_right, self.mean_a)

    def train_it(self, it=25, thinning=5):
        for i in range(it):
            self.sample_z1(opt=2)
            self.sample_eta()
            self.sample_a()
            s = ((i+1) / thinning)
            if s == int(s):
                print("Training iteration #", i)
                p = i / it * 100
                print("Progress is %.2f %%" % p)
                print("-----")
                cur_ph = self.get_ph()
                cur_th = self.get_zbar()
                if s > 1:
                    m = (s-1)/s
                    self.ph = m * self.ph + (1-m) * cur_ph
                    self.th = m * self.th + (1-m) * cur_th
                else:
                    self.ph = cur_ph
                    self.th = cur_th


def run_it(f="thesis_data.csv", d=3):
    a, b, c = load_corpus(f, d)
    hslda = HSLDA(a, b, c)
    return hslda


def train_it(f="thesis_data.csv", d=3, it=5, s=2):
    hslda = run_it(f=f, d=d)
    hslda.train_it(it=it, thinning=s)
    return hslda

