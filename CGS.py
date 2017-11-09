import numpy as np
from scipy import stats as stats
from gensim.parsing import preprocessing
from gensim import corpora, matutils
from copy import copy


# Static methods:
def dir_draw(array_in, axis=0):
    return np.apply_along_axis(np.random.dirichlet, axis=axis, arr=array_in)

class Gibbs:
    def __init__(self, documents, K="flex", alpha_strength=50):
        # base setup
        self.docs = preprocessing.preprocess_documents(documents.docs)
        self.dict = corpora.Dictionary(self.docs)
        self.corpus = [self.dict.doc2bow(doc) for doc in self.docs]

        self.labs = documents.prepped_labels
        self.ldict = corpora.Dictionary(self.labs)
        self.ordered_labs = np.sort([x for x in self.ldict.token2id.keys()])

        self.V = len(self.dict)
        self.D = len(self.docs)

        # This if-else is to distinguish between LDA and HSLDA!
        # If: Sets up the Ramage09 type of LDA
        if K=="flex":
            # Determine K in data based LDA:
            self.K = len(self.ldict)

            # Determine alpha in LDA (a la Ramage 09)
            self.alpha = self._get_label_matrix_indic(nr_labels=self.K)
            _ = 50 / np.sum(self.alpha, axis=0)
            self.alpha = self.alpha * _

        # Elif: Sets up the HSLDA type of LDA
        elif isinstance(K, int):
            self.K = K
            self.alpha = self._hslda_alpha_naive()
            self.nr_of_labs = len(self.ordered_labs)
            #self.Y = self._get_label_matrix_indic(self.nr_of_labs)
            #self.Y[self.Y == 0] = -1

            # Reorder rows Y according to ldict (Illogical for humans!)
            # _ = np.argsort(list(self.ldict.token2id.keys()))
            # self.Y = self.Y[_, :]
        else:
            raise ValueError('%s is not a valid value. K must either be "fixed" or an integer') % K

        self.alpha_m = alpha_strength

        self.beta_c = 200 / self.V
        self.beta = self.beta_c * np.ones((self.V, self.K))

        # Count-containers for word assignments
        self.lenD = [len(doc) for doc in self.docs]
        self.zet = [np.repeat(0, x) for x in self.lenD]
        self.n_zxd = np.zeros((self.K, self.D))  # count of topic k assignments in doc d
        self.n_wxz = np.zeros((self.V, self.K))  # count of word v assignments in topic k
        self.n_z = np.zeros(self.K)  # total number of assignment counts in topic k (colsum n_wxz)

        # Fill the count-containers (initialize word assignments z)
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

    def _get_label_matrix_indic(self, nr_labels):
        # Get tuples indicating counts per label per document:
        _ = [self.ldict.doc2bow(label) for label in self.labs]
        label_indic = matutils.corpus2dense(_, nr_labels, self.D)
        # Rearrange so that row1 is label A, row2 is label B, etc.
        _ = np.argsort([x for x in self.ldict.token2id.keys()])
        return label_indic[_, :]

    def _hslda_alpha_naive(self, alpha_p2=1):
        alpha_p1 = 50/self.K
        beta = np.random.dirichlet(np.repeat(alpha_p1, self.K), 1)
        beta = np.tile(beta.transpose(), (1, self.D))
        return alpha_p2*beta

    def get_theta(self):
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

    def get_topiclist(self, n=10):
        self.phi = self.get_phi()
        topiclist = []
        for k in range(self.K):
            inds = np.argsort(-self.phi[k, :])[:n]
            topwords = [self.dict[x] for x in inds]
            topwords.insert(0, self.ordered_labs[k])
            topiclist += [topwords]
        return topiclist

    def _common_sample_z(self, d, v, z):
        self.n_wxz[v, z] -= 1
        self.n_zxd[z, d] -= 1
        self.n_z[z] -= 1
        self.lenD[d] -= 1

        left_num = self.n_wxz[v, :] + self.beta_c  # K dimensional
        left_den = self.n_z + self.beta_c * self.V  # K dimensional
        right_num = self.n_zxd[:, d] + self.alpha[:, d]  # K dimensional
        right_den = self.lenD[d] + self.alpha_m  # K dimensional
        return (left_num / left_den) * (right_num / right_den)

    def sample_z(self, d, word, pos):
        v = self.dict.token2id[word]
        z = self.zet[d][pos]
        prob = self._common_sample_z(d, v, z)
        prob /= sum(prob)
        new_z = np.random.multinomial(1, prob).argmax()

        self.zet[d][pos] = new_z
        self.n_wxz[v, new_z] += 1
        self.n_zxd[new_z, d] += 1
        self.n_z[new_z] += 1
        self.lenD[d] += 1

class GibbsSampling(Gibbs):
    def __init__(self, documents, alpha_strength=50):
        super(GibbsSampling, self).__init__(documents, K="flex")

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


class HSLDA_Gibbs(Gibbs):
    def __init__(self, documents,K=15):
        super(HSLDA_Gibbs, self).__init__(documents, K=K)
        self.eta = self._hslda_eta_naive()
        self.zbar = self.get_theta()

        _ = [self.ldict.doc2bow(label) for label in self.labs]
        self.Y = matutils.corpus2dense(_, self.nr_of_labs, self.D)
        self.Y[self.Y == 0] = -1

        # Create parent label mapping:
        _ = list(self.ldict.token2id.keys())
        self.parent_map = dict(zip(_, [lab[:-1] for lab in _]))

        # For my understanding, get ldict in ordered way:
        # self.ldict2 = dict(zip(range(len(self.ordered_labs)), self.ordered_labs))

        # Initiate a_ld running variables
        self.a_ld = np.empty(shape=(self.nr_of_labs, self.D))
        self.zbarT_etaL = copy(self.a_ld)

        for l_id in range(self.nr_of_labs):
            for d in range(self.D):
                self.zbarT_etaL[l_id, d] = np.dot(self.zbar[d, :], self.eta[:, l_id])
                a_mu =self.zbarT_etaL[l_id, d]
                parent_lab = self.parent_map[self.ldict.__getitem__(l_id)]
                parent_id = self.ldict.token2id[parent_lab]

                if self.Y[parent_id, d] == 1:
                    if self.Y[l_id, d] == 1:
                        self.a_ld[l_id, d] = a_mu + stats.truncnorm.rvs(-a_mu, 10, size=1)
                    elif self.Y[l_id, d] == -1:
                        self.a_ld[l_id, d] = a_mu + stats.truncnorm.rvs(-10, -a_mu, size=1)
                    else:
                        raise ValueError("The label %s in document %s is nor -1 or 1" % l_id, d)
                elif self.Y[parent_id, d] == -1:
                    self.a_ld[l_id, d] = a_mu + stats.truncnorm.rvs(-10, -a_mu, size=1)
            #print("Just finished label %s")%(l_id)

    def _hslda_eta_naive(self, mu=-1, sigma=1):
        eta_l = np.random.normal(mu, sigma, self.K*self.nr_of_labs)
        return eta_l.reshape(self.K, self.nr_of_labs)


    # TODO: Initiate a_ld  (check)
    # TODO: Check out the flexible alpha/beta by teh et al
    # TODO: Write update functions
        # P(z | .. ) =
        # P(eta_l | ...) =
            # Sigma, Mu
        # P(a_ld | ... ) =
        #
    # TODO: Organise GibbsSampling & HSLDA_Gibbs properly (superclass?) (check)
    # TODO: optimize get_theta(). It now involves DxK every time it's called
# End of HSLDA_Gibbs

# Static methods for preparing unseen data:
# Check whether I provide 1 or multiple


