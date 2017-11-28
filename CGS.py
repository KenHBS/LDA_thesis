import numpy as np
# from scipy import stats as stats
# from scipy.special import erf
from gensim.parsing import preprocessing
from gensim import corpora, matutils
from copy import copy
import rtnorm as rt
from antoniak import rand_antoniak

# Static methods:
def dir_draw(array_in, axis=0):
    return np.apply_along_axis(np.random.dirichlet, axis=axis, arr=array_in)


class Gibbs:
    """
    The most basic Gibbs sampling class.

    This class roughly serves four purposes:
        1) Organize a text corpus into usable chunks: bags-of-words,
            word dictionaries, label dictionaries.
        2) Identify and set up LDA specific parameters and priors
        3) Setup and fill word-assignment containers with initial information
        4) Define Gibbs methods that are shared among all specialized children


    Attributes:
        --- Processing of training data ---
        docs   (list): D sublists with stemmed bag-of-words
        dict   (gensim dict): Dictionary mapping all words in corpus to IDs
        corpus (list): dense word id representation of docs.
        labs   (list): contains D sublists with the labels per document
        ldict  (gensim dictionary): Dict mapping all labels to label IDs

        --- LDA (HSLDA) specific parameters ---
        K      (int): Number of topics
        V      (int): Number of words in corpus
        D      (int): Number of documents in corpus
        alpha (float): Hyperprior for document-topic distribution (theta)
        beta  (float): Hyperprior for topic-word distribution (phi)
        lenD  (int): D sublists with the number of words in document d
        phi   (...): Empty container for later use.

        --- Containers for word-assignment information ---
        zet   (list): document with every word assigned to one topic k
        n_zxd (list): per topic k count of word-assignent in each document d
        n_wxz (list): length K sublists with word-assignment counts
        n_z   (list): total count of word-assignments per document (K counts)

    Methods:
        _get_label_matrix_indic: Orders labels per document
        _hslda_alpha_naive     : Calculates the alpha hyperprior for HSLDA
        get_theta              : Calculate empirical doc-topic distr. (theta)
        get_phi                ; Calculate empirical topic-word distr. (phi)
        get_topiclist          : Retrieve top N words in every word-distr.
        _common_sample         : Prepare for basic LDA draw for zet
        sample_z               : Basic LDA draw for zet
        addback_zet            : Close basic LDA draw for zet


    """

    def __init__(self, documents, K="flex", alpha_strength=50):
        # 1) Processing of training data:
        self.docs = preprocessing.preprocess_documents(documents.docs)
        self.dict = corpora.Dictionary(self.docs)
        self.corpus = [self.dict.doc2bow(doc) for doc in self.docs]

        self.labs = documents.prepped_labels
        self.ldict = corpora.Dictionary(self.labs)
        self.ordered_labs = np.sort([x for x in self.ldict.token2id.keys()])

        # 2) LDA (HSLDA) specific parameters:
        self.V = len(self.dict)
        self.D = len(self.docs)

        # This if-else is to distinguish between LDA and HSLDA!
        if K == "flex":   # Sets up the Ramage09 type of LDA
            self.K = len(self.ldict)

            # Determine alpha in LDA (a la Ramage 09)
            self.alpha = self._get_label_matrix_indic(nr_labels=self.K)
            _ = 50 / np.sum(self.alpha, axis=0)
            self.alpha = self.alpha * _
        elif isinstance(K, int):  # Sets up the HSLDA type of LDA
            self.K = K
            self.alpha = self._hslda_alpha_naive()
            self.nr_of_labs = len(self.ordered_labs)

        else:
            raise ValueError('%s is not a valid value. K must either \
            be "fixed" or an integer' % K)

        self.alpha_m = alpha_strength

        self.beta_c = 100 / self.V
        self.beta = self.beta_c * np.ones((self.V, self.K))
        self.phi = None

        # 3) Count-containers for word assignments
        self.lenD = [len(doc) for doc in self.docs]
        self.zet = [np.repeat(0, x) for x in self.lenD]
        self.n_zxd = np.zeros((self.K, self.D))  # count topic k in doc d
        self.n_wxz = np.zeros((self.V, self.K))  # count word v in topic k
        self.n_z = np.zeros(self.K)  # ttl assignments in topic k (colsum n_wxz)

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

    def _get_label_matrix_indic(self, nr_labels):
        """Orders labels in interpretable way for humans"""

        # Get tuples indicating counts per label per document:
        _ = [self.ldict.doc2bow(label) for label in self.labs]
        label_indic = matutils.corpus2dense(_, nr_labels, self.D)
        # Rearrange so that row1 is label A, row2 is label B, etc.
        _ = np.argsort([x for x in self.ldict.token2id.keys()])
        return label_indic[_, :]

    def _hslda_alpha_naive(self, alpha_p2=1):
        """Compute naive alpha hyperparameter (for theta) in HSLDA setting"""
        alpha_p1 = 50/self.K
        beta = np.random.dirichlet(np.repeat(alpha_p1, self.K), 1)
        beta = np.tile(beta.transpose(), (1, self.D))
        return alpha_p2*beta

    def get_theta(self):
        """ Average word-assignment counts per document: empirical theta """
        return (self.n_zxd / self.lenD).T

    def get_theta2(self):
        """ Inefficient version of get_theta(). Serves as sanity check """
        th = np.zeros((self.D, self.K))
        for d in range(self.D):
            for z in range(self.K):
                frac_a = self.n_zxd[z][d] + self.alpha[z][d]
                frac_b = self.lenD[d] + self.alpha_m
                th[d][z] = frac_a / frac_b
        return th

    def get_phi(self):
        """ Average of word-token count per topic: empirical phi """
        return (self.n_wxz / self.n_z).T

    def get_phi2(self):
        """ Inefficient version of get_phi(). Serves as sanity check """
        ph = np.zeros((self.K, self.V))
        for z in range(self.K):
            for w in range(self.V):
                frac_a = self.n_wxz[w][z] + self.beta_c
                frac_b = self.n_z[z] + self.beta_c*self.V
                ph[z][w] = frac_a / frac_b
        return ph

    def get_topiclist(self, n=10, hslda=False):
        """ Lists top n words in every topic-word distr. (phi overview)"""

        # self.phi = self.get_phi()
        topiclist = []
        for k in range(self.K):
            inds = np.argsort(-self.phi_hat[k, :])[:n]
            topwords = [self.dict[x] for x in inds]
            if hslda:
                topwords.insert(0, "Topic " + str(k))
            else:
                topwords.insert(0, self.ordered_labs[k])
            topiclist += [topwords]
        return topiclist

    def _deduct_z(self, d, v, z):
        """ Reduce single word-assignment count by one to compute posterior """
        self.n_wxz[v, z] -= 1
        self.n_zxd[z, d] -= 1
        self.n_z[z] -= 1
        self.lenD[d] -= 1

    def _common_sample_z(self, d, v, z):
        self._deduct_z(d, v, z)

        left_num = self.n_wxz[v, :] + self.beta_c  # K dimensional
        left_den = self.n_z + self.beta_c * self.V  # K dimensional
        right_num = self.n_zxd[:, d] + self.alpha[:, d]  # K dimensional
        right_den = self.lenD[d] + self.alpha_m  # K dimensional
        return (left_num / left_den) * (right_num / right_den)

    def sample_z(self, d, word, pos):
        """ Pick word -> reduce count -> compute posterior -> resample z """
        v = self.dict.token2id[word]
        z = self.zet[d][pos]
        prob = self._common_sample_z(d, v, z)
        prob /= sum(prob)
        new_z = np.random.multinomial(1, prob).argmax()
        self.addback_zet(d, pos, new_z, v)

    def addback_zet(self, d, pos, new_z, v):
        """ Add the newly sampled zet back to the count containers """
        self.zet[d][pos] = new_z
        self.n_wxz[v, new_z] += 1
        self.n_zxd[new_z, d] += 1
        self.n_z[new_z] += 1
        self.lenD[d] += 1


class GibbsSampling(Gibbs):
    """
    Child of Gibbs. Enables Gibbs sampling for LDA, following Ramage '09.

    Attributes:
         Same as Gibbs

    Methods:
        run:                  Perform z-sampling for all words in all docs.
        init_new_doc:         Create zet containers for unseen document
        sample_for_posterior: Samples word-assignment for single unseen doc
        posterior:            Loop all unseen docs, retain posterior info
        theta_output:         Overview of single document-topic distr. theta
        post_theta:           Overview of all unseen doc-topic distributions

    TODO: Incorporate 'thinning' logic is Ramage's LDA
    """

    def __init__(self, documents):
        super(GibbsSampling, self).__init__(documents, K="flex")

    def run(self, nsamples, burnin=0):
        """
        Run iterations for all documents and all words
        and reassign the word-assignment in every iteration.

        :param nsamples:(int) Number of iterations over all docs and words
        :param burnin: (int)  Nr of iterations before sample states are recorded
        :return:              Propagate from one state to next state ´
        """
        if nsamples <= burnin:
            raise Exception('Burn-in point exceeds number of samples')

        for s in range(nsamples):
            for d, doc in enumerate(self.docs):
                if(d % 250 == 0):
                    print("Working on doc %d in sample number %d " % d, s+1)
                for pos, word in enumerate(self.docs[d]):
                    self.sample_z(d, word, pos)

    def init_newdoc(self, new_doc, sym=False):
        """
        Prepare unseen document for posterior calculation. Attach initial state
        of word-topic assignments to every word in the document

        :param new_doc: (list) Unseen bag-of-words (stemmed and tokenized)
        :param sym: (boolean)  Should informative or uninformative (symmetric)
                                    alpha prior be used?
        :return:               Containers with word-assignment counts
        """
        if sym:
            alpha = np.repeat(50/self.K, self.K)
        else:
            alpha = copy(self.n_z)
        _ = np.random.dirichlet(alpha)
        zet = np.random.multinomial(1, _, len(new_doc)).argmax(axis=1)

        z_counts = np.zeros(self.K)
        for it, zn in enumerate(zet):
            z_counts[zet[it]] += 1
        assert sum(z_counts) == len(new_doc), print('z_counts %d is not same \
         as nr of words %d' % (sum(z_counts), len(new_doc)))
        return zet, z_counts

    def sample_for_posterior(self, new_doc, sym=False, n_iter=250):
        """
        Move from sampling state to next sampling state. Resampling word-topic
        assignments in the transition.

        :param new_doc: (list) Unseen bag-of-words (stemmed and tokenized)
        :param sym: (boolean)  Should informative or uninformative (symmetric)
                                    alpha prior be used?
        :param n_iter: (int)   Number of iterations/transitions
        :return:               word assignment count containers for new_doc
        """
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

    def posterior(self, new_docs, sym=False, n=250):
        """
        Takes multiple unseen documents as input and resamples to return a
        document-topic distribution (theta) for every document

        :param new_docs: (list) Unseen bags-of-words (stemmed and tokenized)
        :param sym: (boolean)   Should informative or uninformative (symmetric)
            alpha prior be used?
        :param n: (int)         Number of iterations/transitions
        :return: (list)         With the document-topic distributions (theta)
            for every document
        """
        theta_container = []

        for d, doc in enumerate(new_docs):
            zet, zcount = self.sample_for_posterior(doc, sym, n_iter=n)
            single_theta = list(zcount/sum(zcount))
            theta_container.append(single_theta)
            if d % 5 == 0:
                print("Unseen document number %d" % d)
        return theta_container

    def theta_output(self, th):
        """ Overview of a single document's document-topic distribution """
        th = np.array(th)
        inds = np.where(th > 0)
        labs = self.ordered_labs[inds]
        return list(zip(labs, th[inds]))

    def post_theta(self, new_docs, sym=False):
        """ Overview of all unseen documents' doc-topic distribution """
        thetas = self.posterior(new_docs, sym=sym)
        return [self.theta_output(theta) for theta in thetas]
# End of GibbsSampling Class


class VariationalInf:
    def __init__(self):
        pass

# End of Variational_


class HSLDA_Gibbs(Gibbs):
    """
    Child of Gibbs. Enables Gibbs sampling for HSLDA, following Perotte '11.
    The child-parent hierarchy in the labels (like D12, D1, D) is used to
    leverage the classification results. This is important, because neigbouring
    leaf labels like D12 and D13 are nearly identical. Exploiting the label
    structure may allow for discriminating between nearly identical labels
    near the leafs of the label-tree.

    Attributes:
        --- Inherited attributes ---
        See Gibbs for attributes. Additionally, HSLDA_Gibbs contains:

        --- HSLDA-specific variables and parameters ---
        eta        (np.ndarray): K x L array where every column resembles
                                    the relation between label l and topic k.
                                    The elements are probit regression
                                    coefficients.
        zbar       (np.ndarray): D x K array where every row is an empirical
                                    estimate of the document's mixture of
                                    topics
        mu              (float): Prior for the probit regression coefficients
        sigma           (float): Prior for the probit regression coefficients
        Y          (np.ndarray): L x D array with indicators (-1 or 1) whether
                                     label l is part of document d's label set.
        a_ld       (np.ndarray): L x D array with continuous running variables
                                     for Y
        zbarT_etaL (np.ndarray): L x D array that serves as a container with
                                    intermediate results to avoid numerous
                                        matrix multiplications.

        --- Dictionaries managing the parent-child structure in labels ---
        parent_map  (dict): Maps children to parents, e.g. A12 is mapped to A1
        parent_dict (dict): Immediately maps child's label ID to parent's
                                label ID.

        --- Intermediate result containers. 'thinning'-related containers ---
        phi_hat    (np.ndarray): KxV array. Empirical topic-word distributions
        theta_hat  (np.ndarray): DxK array. Empirical doc-topic distributions
        eta_hat    (np.ndarray): KxL array. Empirical probit coefficients

        --- Containers for the Hierarchical Dirichlet prior b:
        m_aux (np.ndarray): DxK array - auxiliary vars for sampling b
        m_dot (np.ndarray): 1xK array - column sum of m_aux
        b     (np.ndarray): 1xK array - Hierarch. Dirichlet prior

    Methods:
        sample_a            : Resamples values a for all labels l amd docs d
        _hslda_eta_naive    : Draw initial values for eta. Based on mu, sigma
        sample_to_next_state: Transition from one state to next.
        save_this_state     : Saves states in regular intervals.

    TO DO:
        - Incorporate hierarchical Dirichlet prior for doc-topic
        - Incorporate Antoniak random draws
        - Incorporate new sampling for doc-topic prior beta
        - Incorporate new prior updates in z sampling
        - Check logic of alpha, alpha', beta and gamma
    """
    def __init__(self, documents, K=15, mu=-1, sigma=1, a_prime=1,
                 alpha=0.5):
        super(HSLDA_Gibbs, self).__init__(documents, K=K)
        self.a_prime = a_prime
        self.alpha = alpha

        self.eta = self._hslda_eta_naive(mu=mu, sigma=sigma)
        self.zbar = self.get_theta()
        self.mu = mu
        self.sigma = sigma

        _ = [self.ldict.doc2bow(label) for label in self.labs]
        self.Y = matutils.corpus2dense(_, self.nr_of_labs, self.D)
        self.Y[self.Y == 0] = -1

        # Create parent label mapping:
        _ = list(self.ldict.token2id.keys())
        self.parent_map = dict(zip(_, [lab[:-1] for lab in _]))

        # Straight from label id, to parent's label id
        _ = [self.parent_map[x] for x in _]
        dict_vals = [self.ldict.token2id[x] for x in _]
        dict_keys = range(self.nr_of_labs)
        self.parent_dict = dict(zip(dict_keys, dict_vals))

        # Initiate a_ld running variables
        self.a_ld = np.empty(shape=(self.nr_of_labs, self.D))
        self.zbarT_etaL = np.dot(self.zbar, self.eta).T
        self.sample_a()

        # Empty containers for saving sampling states:
        self.phi_hat = None
        self.theta_hat = None
        self.eta_hat = None

        # Hierarchical Dirichlet Prior (auxiliary) vars and outcomes
        self.m_aux = np.empty((self.D, self.K))[0]
        self.m_dot = np.empty((1, self.K))[0]
        self.b = np.random.dirichlet( (self.a_prime)*np.ones(self.K), size=1 )

    def sample_a(self):
        """
        Resample all running variables a_{l,d}, by drawing from a truncated
        normal distribution, that's specified by the correspond y_{l,d} value
        and the value of its parent label.

        :return: A new value for a_{l,d}
        """
        for index, value in np.ndenumerate(self.zbarT_etaL):
            parent_id = self.parent_dict[index[0]]

            d_parent = (self.Y[parent_id, index[1]] == 1)
            d_own = (self.Y[index] == 1)

            if d_parent is False:
                self.a_ld[index] = rt.rtnorm(a=-float('inf'), b=0, mu=value)
            elif d_own is False:
                self.a_ld[index] = rt.rtnorm(a=-float('inf'), b=0, mu=value)
            else:
                self.a_ld[index] = rt.rtnorm(a=0, b=float('inf'), mu=value)

    def sample_m_dot(self):
        """
        Take the word-topic assignments z and the doc-topic priors alpha and b
        to sample values for m_{d,k}, based on the Antoniak distribution.
        See Teh et al (2006) and Wallach (2009) for more details

        :return: (np.ndarray): DxK array - updated version of self.m_aux:
        """
        for k in range(self.K):
            ab = self.alpha * self.b[0, k]
            sub = self.n_zxd[k, :]
            for d in range(self.D):
                n_dk = sub[d]
                self.m_aux[d, k] = rand_antoniak(ab, int(n_dk))
        self.m_dot = np.apply_along_axis(sum, 0, self.m_aux)

    def sample_b(self):
        """
        Update the base measure of the topic-doc prior based on the current
        data. See Teh et al (2006) and Wallach (2009) for more details

        :return: (np.ndarray): 1xK array - updated version of self.b
        """
        post_b = self.m_dot + self.a_prime
        self.b = np.random.dirichlet(post_b, size=1)

    def sample_z_one(self, d, v, z):
        """
        Update the word-topic assignments z_{d,n} with Hierarchical Dirichlet
        priors for the document-topic distribution.

        :return: (np.ndarray): 1xK array - first part cond. posterior z_{d,n}
        """
        l = self.n_zxd[:, d] + self.alpha*self.b
        r_num = self.n_wxz[v, :] + self.beta_c
        r_den = self.n_z + self.beta_c*self.V
        return(l * (r_num/r_den))

    def sample_z_two(self, l, z, invd, z_eta, sub_a):
        """
        Second half of the cond. posterior of z_{d,n}. Only manipulates cells
        affected by the change in index, to avoid calculating K large matrix
        multiplications for every D x N word token

        :param l: (list) label IDs of document d's label set
        :param z: (int) Old topic assignment of word w_{d,n}. Between 1 and K
        :param invD: (float) 1 divided by document length of document d
        :param z_eta: (np.ndarray): The elements in z*eta that are affected
        :param sub_a: (np.ndarray): The elements a_{l,d} in a_ld from labels l
        :return: (np.ndarray): 1xK array - second part cond. posterior z_{d,n}
        """
        diff_z_k = self.eta[:, l] - self.eta[z, l]
        z_eta_new = z_eta + (invd * diff_z_k)
        kernel = ((z_eta_new - sub_a) ** 2) / (-2)
        part2 = [np.exp(np.sum(x)) for x in kernel]
        return(np.array(part2))

    def _hslda_eta_naive(self, mu=-1, sigma=1):
        """ Use uninformative priors to initialize eta """
        eta_l = np.random.normal(mu, sigma, self.K*self.nr_of_labs)
        return eta_l.reshape(self.K, self.nr_of_labs)

    def sample_to_next_state(self, nsamples, burnin=0, thinning=10):
        """ The sampling procedure consists of four conditional posteriors:
            1) word-assignments z_{d,n} : all words and docs are reassigned
            2) probit-coefficients eta  : all labels and topics are reassigned
            3) running variables a_{l,d}: all labels and docs are reassigned
            4) update doc-topic hierarchical dirichlet prior b_k

        :param nsamples: (int) Nr of iterations
        :param burnin: (int)   Nr of iterations before the first state is saved
        :param thinning: (int) Length of interval between saved states
        :return:               New state of the model.
        """
        if nsamples <= burnin:
            raise Exception('Burn-in point exceeds number of samples')
        for s in range(nsamples):
            intersave = (s+1)/thinning
            if intersave == int(intersave):
                self.save_this_state(N=int(intersave))

            # 1) Sample new word-assignments z_{d,n}
            for d in range(self.D):
                # Find the labels that are part of document d's label set:
                lab_d = np.where(self.Y[:, d] == 1)[0]
                invD = 1/self.lenD[d]
                # Only focus on document d and labels in d's label set:
                z_eta = self.zbarT_etaL[lab_d, d]
                sub_a = self.a_ld[lab_d, d]
                if d % 1000 == 0:
                    print("Working on doc %d in sample number %d "%(d, s+1))
                    for pos, word in enumerate(self.docs[d]):
                        v = self.dict.token2id[word]
                        z = self.zet[d][pos]
                        self._deduct_z(d, v, z)

                        # Get probability and draw new z-value
                        part1 = self.sample_z_one(d, v, z)
                        part2 = self.sample_z_two(lab_d, z, invD, z_eta, sub_a)
                        prob = part1*part2
                        prob /= np.sum(prob)

                        new_z = np.random.multinomial(1, prob[0]).argmax()

                        # Replace old z value with new one
                        self.addback_zet(d, pos, new_z, v)
                        self.zbar[d, z] -= invD
                        self.zbar[d, new_z] += invD
                        delta = invD*(self.eta[new_z, :]-self.eta[z, :])

                        self.zbarT_etaL[:, d] += delta

            # 2) Drawing new eta_l samples
            zT_z = np.dot(self.zbar.T, self.zbar)
            sig_hat_inv = (np.identity(self.K) * 1/self.sigma)+zT_z
            sig_hat = np.linalg.inv(sig_hat_inv)
            musigma = np.ones(self.K) * (self.mu/self.sigma)
            print("Start updating eta for all labels")
            for l in range(self.nr_of_labs):
                part2 = musigma + np.dot(self.zbar.T, self.a_ld[l, :].T)
                mu_hat = np.dot(sig_hat, part2)
                new_draw = np.random.multivariate_normal(mu_hat, sig_hat)
                self.eta[:, l] = new_draw
            # Recalculate objects that need updating due to new eta:
            self.zbarT_etaL = np.dot(self.eta.T, self.zbar.T)

            # 3) Drawing new a_{l,d} samples:
            print("Start updating a_ld for all docs and labels")
            self.sample_a()

            # 4) Update doc-topic dirichlet prior b_k with new data:
            self.sample_m_dot()
            self.sample_b()


    def save_this_state(self, N):
        """
        Update the mean theta, phi and eta with the current state's results
        """
        ph = self.get_phi()
        th = self.get_theta()
        if N > 1:
            self.phi_hat = (N-1)/(N) * self.phi_hat + 1/N * ph
            self.theta_hat = (N-1)/(N) * self.theta_hat + 1/N * th
            self.eta_hat = (N-1)/(N) * self.eta_hat + 1/N * self.eta
        else:
            self.phi_hat = ph
            self.theta_hat = th
            self.eta_hat = self.eta

    # TODO: Check out the flexible alpha/beta by teh et al
    # TODO: Repair get_theta() to sum to 1 instead of about 0.5
    # TODO: optimize get_theta(). It now involves DxK every time it's called
    # TODO: Check why z doesn't separate topics
# End of HSLDA_Gibbs



