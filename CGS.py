import numpy as np
from gensim.parsing import preprocessing
from gensim import corpora, matutils
from copy import copy


# Static methods:
def dir_draw(array_in, axis=0):
    return np.apply_along_axis(np.random.dirichlet, axis=axis, arr=array_in)


class GibbsSampling:
    def __init__(self, documents, alpha_strength=50):
        # if not isinstance(documents, DocDump):
        #    raise TypeError("Gibbs only takes DocDump instances as input")

        self.docs = preprocessing.preprocess_documents(documents.docs)
        self.dict = corpora.Dictionary(self.docs)
        self.corpus = [self.dict.doc2bow(doc) for doc in self.docs]

        self.labs = documents.prepped_labels
        self.ldict = corpora.Dictionary(self.labs)
        self.ordered_labs = np.sort([x for x in self.ldict.token2id.keys()])

        self.V = len(self.dict)
        self.K = len(self.ldict)
        self.D = len(self.docs)

        self.alpha_m = alpha_strength

        _ = [self.ldict.doc2bow(label) for label in self.labs]
        self.alpha = matutils.corpus2dense(_, self.K)
        # Rearrange so that col1 is label A, col2 is label B, etc.

        _ = np.argsort([x for x in self.ldict.token2id.keys()])
        self.alpha = self.alpha[_, :]

        _ = 50/np.sum(self.alpha, axis=0)
        self.alpha = self.alpha * _

        self.beta_c = 200/self.V
        self.beta = self.beta_c * np.ones((self.V, self.K))

        # Set up containers for assignment counts
        self.lenD = [len(doc) for doc in self.docs]
        self.zet = [np.repeat(0, x) for x in self.lenD]
        self.n_zxd = np.zeros((self.K, self.D))  # count of topic k assignments in doc d
        self.n_wxz = np.zeros((self.V, self.K))  # count of word v assignments in topic k
        self.n_z = np.zeros(self.K) # total number of assignment counts in topic k (colsum n_wxz)

        for d, doc in enumerate(self.docs):
            z_n = self.zet[d]
            for w, word in enumerate(doc):
                p_z = self.alpha[:, d] / np.sum(self.alpha[:, d])
                while sum(p_z) > 1:
                    p_z /= 1.0005
                z = np.random.multinomial(1, p_z).argmax()
                z_n[w] = z
                v = self.dict.token2id[word]
                self.n_zxd[z, d] += 1
                self.n_wxz[v, z] += 1
                self.n_z[z] += 1
            #self.zet[d] = z_n  is implicit by z_n = self.zet[d]

    def get_theta(self, ):
        th = np.zeros((self.D, self.K))
        for d in range(self.D):
            for z in range(self.K):
                frac_a = self.n_zxd[z][d] + self.alpha[z][d]
                frac_b = self.lenD[d] + self.alpha_m
                th[d][z] = frac_a / frac_b
        return th

    def get_phi(self):
        ph = np.zeros((self.K, self.V))
        for z in range(self.K):
            for w in range(self.V):
                frac_a = self.n_wxz[w][z] + self.beta_c
                frac_b = self.n_z[z] + self.beta_c*self.V
                ph[z][w] = frac_a / frac_b
        return ph

    def sample_z(self, d, word, pos):
        v = self.dict.token2id[word]
        z = self.zet[d][pos]
        self.n_wxz[v, z] -= 1
        self.n_zxd[z, d] -= 1
        self.n_z[z] -= 1
        self.lenD[d] -= 1

        left_num = self.n_wxz[v, :] + self.beta_c    # K dimensional
        left_den = self.n_z + self.beta_c * self.V   # K dimensional
        right_num = self.n_zxd[:, d] + self.alpha[:, d] # K dimensional
        right_den = self.lenD[d] + self.alpha_m         # K dimensional

        prob = (left_num / left_den) * (right_num / right_den)
        prob /= sum(prob)
        new_z = np.random.multinomial(1, prob).argmax()

        self.zet[d][pos] = new_z
        self.n_wxz[v, new_z] += 1
        self.n_zxd[new_z, d] += 1
        self.n_z[new_z] += 1
        self.lenD[d] += 1

    def run(self, nsamples, burnin=0):
        if nsamples <= burnin:
            raise Exception('Burn-in point exceeds number of samples')

        for s in range(nsamples):
            for d, doc in enumerate(self.docs):
                if(d % 250 == 0):
                    print("Working on document %d in sample number %d "%(d, s+1))
                for pos, word in enumerate(self.docs[d]):
                    self.sample_z(d, word, pos)
            # th = self.get_theta()
            # ph = self.get_phi()

    def get_topiclist(self, n=10):
        self.phi = self.get_phi()
        topiclist = []
        for k in range(self.K):
            inds = np.argsort(-self.phi[k, :])[:n]
            topwords = [self.dict[x] for x in inds]
            topwords.insert(0, self.ordered_labs[k])
            topiclist += [topwords]
        return topiclist

    def init_newdoc(self, new_doc, sym=False):
        if sym:
            alpha = np.repeat(50/self.K, self.K)
        else:
            alpha = copy(self.n_z)
        _ = np.random.dirichlet(alpha)
        zet = np.random.multinomial(1, _, len(new_doc)).argmax(axis=1)

        z_counts = np.zeros(self.K)
        for it, zn in enumerate(zet):
            z_counts[zet[it]] += 1
        assert sum(z_counts)==len(new_doc), print('z_counts %d is not same as\
         nr of words %d' % (sum(z_counts), len(new_doc)))
        return zet, z_counts

    def sample_for_posterior(self, new_doc, sym=False, n_iter=250):
        zet, zcounts = self.init_newdoc(new_doc, sym=sym)
        if "phi" not in self.__dir__():
            self.phi = self.get_phi()
        for i in range(n_iter):
            for pos, word in enumerate(new_doc):
                v = self.dict.token2id[word]
                z = zet[pos]
                zcounts[z] -= 1

                prob = self.phi[:, v] * (zcounts/sum(zcounts))
                prob /= sum(prob)
                new_z = np.random.multinomial(1, prob).argmax()

                zet[pos] = new_z
                zcounts[new_z] += 1
        return zet, zcounts

    def posterior(self, new_docs, sym=False):
        theta_container = []

        for d, doc in enumerate(new_docs):
            zet, zcount = self.sample_for_posterior(doc, sym)
            single_theta = list(zcount/sum(zcount))
            theta_container.append(single_theta)
            if d % 5 == 0:
                print("Unseen document number %d" % d)
        return theta_container

    def theta_output(self, th):
        th = np.array(th)
        inds = np.where(th > 0)
        labs = self.ordered_labs[inds]
        return list(zip(labs, th[inds]))

    def post_theta(self, new_docs, sym=False):
        thetas = self.posterior(new_docs, sym=sym)
        return [self.theta_output(theta) for theta in thetas]
# End of GibbsSampling Class


class Variational_Inf:
    def __init__(self):
        pass

# End of Variational_

# Static methods for preparing unseen data:
# Check whether I provide 1 or multiple


# test = [sorted(x) for x in rawdata.prepped_labels