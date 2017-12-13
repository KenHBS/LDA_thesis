import numpy as np
# from scipy import stats as stats
# from scipy.special import erf
from gensim.parsing import preprocessing, preprocess_documents
from gensim import corpora, matutils
from copy import copy
from itertools import compress
import re
from rtnorm import rtnorm as rt
from antoniak import *
from numpy.random import normal
from scipy.stats import truncnorm, norm
rvs = truncnorm.rvs
from doc_prepare import new_docs_prep, open_txt_doc

# TODO Check why theta_hat has so many equal values..
# TODO Check whether results improve by removing generic words:
#       Paper, estimate,
# TODO Improve doc_prepare

# Static methods:
def dir_draw(array_in, axis=0):
    return np.apply_along_axis(np.random.dirichlet, axis=axis, arr=array_in)

def check_equal(checklist):
    return len(set(checklist)) <= 1

# Generate the table for Stirling numbers of the first kind:
def get_stirling_nrs(N):
    stir = np.identity(int(N))
    stir[1,0] = 0
    stir[2,1] = 1

    for n in range(3, N):
        for k in range(1,n):
            stir[n, k] = (stir[n-1, k-1] + (n-1)*stir[n-1, k])
    stir = list(map(np.divide, stir, [max(x) for x in stir]))
    return stir

def pick_z(prob):
    return np.random.multinomial(1, prob).argmax()

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

    def __init__(self, documents, K="flex", alpha_strength=50,
                 rm_generic=False, ramage_mix=False, LN=1, perf_alpha=False,
                 cascade=False, major_dict=None):
        # 1) Processing of training data:
        if rm_generic:
            rm_ =  ['model', 'market', 'economy', 'economic', 'policy',
                    'paper', 'result', 'increase','polici', 'effect', 'effects']
            preprocessing.STOPWORDS = preprocessing.STOPWORDS.union(set(rm_))

        #if cascade:
        #    labs = documents.prepped_labels
        #    labs = [[y for y in doclab if len(y)==level] for doclab in labs]
        #    documents.prepped_labels = labs
        self.docs = preprocessing.preprocess_documents(documents.docs)

        if cascade:
            self.dict = major_dict
        else:
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
            if ramage_mix:
                # Set self.K equal to leaf nodes only
                flat_list = []
                for sublist in self.ldict.values():
                    flat_list.append(sublist)
                LN_labels = list(set([x for x in flat_list if len(x) == LN]))
                self.K = len(LN_labels)

                labsets = [x.split(" ") for x in documents.lab]
                if LN in [1,2,3]:
                    labsets_LN = [[x[0:LN] for x in d_lab] for d_lab in labsets]
                else:
                    raise ValueError("LN must be either 1,2 or 3")
                LN_dict = corpora.Dictionary(labsets_LN)
                _ = [LN_dict.doc2bow(label) for label in labsets_LN]
                label_indic = matutils.corpus2dense(_, self.K, self.D)
                _ = np.argsort([x for x in LN_dict.token2id.keys()])

                self.alpha = label_indic[_, :] + (0.5 / self.K)*perf_alpha
                self.nr_of_labs = len(self.ordered_labs)
            else:
                self.K = len(self.ldict)
                # Determine alpha in LDA (a la Ramage 09)
                self.alpha = self._get_label_matrix_indic(nr_labels=self.K)
                _ = 50 / np.sum(self.alpha, axis=0)
                self.alpha *= _
                self.nr_of_labs = len(self.ordered_labs)
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
        self.phi_hat = None
        self.theta_hat = None
        self.eta = None
        self.eta_hat = None

        # 3) Count-containers for word assignments
        self.lenD = [len(doc) for doc in self.docs]
        self.zet = [np.repeat(0, x) for x in self.lenD]
        self.n_zxd = np.zeros((self.K, self.D))  # count topic k in doc d
        self.n_wxz = np.zeros((self.V, self.K))  # count word v in topic k
        self.n_z = np.zeros(self.K)  # tot assignments in top k (colsum n_wxz)

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

    def get_phi(self):
        """ Average of word-token count per topic: empirical phi """
        return (self.n_wxz / self.n_z).T

    def save_this_state(self, N, hslda=False):
        """
        Update the mean theta, phi and eta with the current state's results
        """
        ph = self.get_phi()
        th = self.get_theta()
        if N > 1:
            self.phi_hat = (N-1)/(N) * self.phi_hat + 1/N * ph
            self.theta_hat = (N-1)/(N) * self.theta_hat + 1/N * th
            if hslda:
                self.eta_hat = ((N-1)/(N)) * self.eta_hat + (1/N) * self.eta
        else:
            self.phi_hat = ph
            self.theta_hat = th
            if hslda:
                self.eta_hat = self.eta

    def clean_new_docs(self, docs):
        bows = preprocess_documents(docs)
        vocab = list(self.dict.token2id.keys())
        return [[x for x in bow if x in vocab] for bow in bows]

    def get_topiclist(self, n=10, hslda=False, cascade=False):
        """ Lists top n words in every topic-word distr. (phi overview)"""

        # self.phi = self.get_phi()
        topiclist = []
        for k in range(self.K):
            inds = np.argsort(-self.phi_hat[k, :])[:n]
            topwords = [self.dict[x] for x in inds]
            if hslda:
                topwords.insert(0, "Topic " + str(k))
            elif cascade:
                all_labs = list(self.ldict.values())
                l3_labs = np.sort([x for x in all_labs if len(x)==3])
                topwords.insert(0, l3_labs[k])
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

    def __init__(self, documents, majordict=None, cascade=False):
        super(GibbsSampling, self).__init__(documents, K="flex",
                                            major_dict=majordict,
                                            cascade=cascade)

    def run(self, nsamples, thinning=None, burnin=0):
        """
        Run iterations for all documents and all words
        and reassign the word-assignment in every iteration.

        :param nsamples:(int) Number of iterations over all docs and words
        :param burnin: (int)  Nr of iter. before sample states are recorded
        :return:              Propagate from one state to next state ´
        """
        if nsamples <= burnin:
            raise Exception('Burn-in point exceeds number of samples')
        if thinning is None:
            thinning = int(nsamples/10)
        for s in range(nsamples):
            intersave = (s+1)/thinning
            if intersave == int(intersave):
                self.save_this_state(N=int(intersave), hslda=False)
            for d, doc in enumerate(self.docs):
                if(d % 250 == 0):
                    print("Working on doc %d in sample number %d " % (d, s+1))
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
        elif not sym:
            alpha = copy(self.n_z)
        alpha *= 0.5   # "Concentration measure" for Dirichlet

        _ = np.random.dirichlet(alpha)
        zet = np.random.multinomial(1, _, len(new_doc)).argmax(axis=1)

        z_counts = np.zeros(self.K, dtype=int)
        for it, zn in enumerate(zet):
            z_counts[zn] += 1
        assert sum(z_counts) == len(new_doc), print('z_counts %d is not same \
         as nr of words %d' % (sum(z_counts), len(new_doc)))
        return zet, z_counts

    def init_newdoc2(self, new_doc):
        zet = np.zeros(len(new_doc), dtype=int)
        zcounts = np.zeros(self.K, dtype=int)
        for pos, word in enumerate(new_doc):
            v = self.dict.token2id[word]
            phi_column = self.phi[:, v]
            prob = phi_column/sum(phi_column)

            z_new = np.random.multinomial(1, prob).argmax()
            zet[pos] = z_new
            zcounts[z_new] += 1
        return zet, zcounts


    def sample_for_posterior(self, new_doc, sym=False, n_iter=250,
                             thinning=25):
        """
        Move from sampling state to next sampling state. Resampling word-topic
        assignments in the transition.

        :param new_doc: (list) Unseen bag-of-words (stemmed and tokenized)
        :param sym: (boolean)  Should informative or uninformative (symmetric)
                                    alpha prior be used?
        :param n_iter: (int)   Number of iterations/transitions
        :return:               word assignment count containers for new_doc
        """
        #zet, zcounts = self.init_newdoc2(new_doc, sym=sym)
        self.th_newdoc = []
        if "phi_hat" in self.__dir__():
            self.phi = self.phi_hat
        elif "phi" not in self.__dir__():
            self.phi = self.get_phi()
        zet, zcounts = self.init_newdoc2(new_doc)
        for i in range(n_iter):
            for pos, word in enumerate(new_doc):
                v = self.dict.token2id[word]
                z = zet[pos]
                zcounts[int(z)] -= 1

                prob = self.phi[:, v] * ((zcounts/sum(zcounts))+(0.5/self.K)) # + hyperpriors
                prob /= sum(prob)
                new_z = np.random.multinomial(1, prob).argmax()

                zet[pos] = new_z
                zcounts[new_z] += 1
            intersave = (i+1)/thinning
            if intersave == int(intersave):
                self.save_state_test(N=int(intersave), zcounts=zcounts)
        return self.th_newdoc

    def save_state_test(self, N, zcounts):
        new_d_th = zcounts/sum(zcounts)
        if N > 1:
            self.th_newdoc = (N-1)/(N) * self.th_newdoc + 1/N * new_d_th
        else:
            self.th_newdoc = new_d_th

    def posterior(self, test_docs, sym=False, n=250):
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
        new_docs = self.clean_new_docs(test_docs)
        for d, doc in enumerate(new_docs):
            self.th_newdoc = self.sample_for_posterior(doc, sym, n_iter=n)
            theta_container.append(self.th_newdoc)
            if d % 5 == 0:
                print("Unseen document number %d" % d)
        return theta_container

    def theta_output(self, th, cascade=False):
        """ Overview of a single document's document-topic distribution """
        th = np.array(th)
        inds = np.where(th > 0)
        if cascade:
            all_labs = list(self.ldict.values())
            labs = np.sort([x for x in all_labs if len(x) == 3])
            labs = labs[inds]
        else:
            labs = self.ordered_labs[inds]
        return list(zip(labs, th[inds]))

    def post_theta(self, test_docs, sym=False, cascade=False, n=100):
        """ Overview of all unseen documents' doc-topic distribution """
        thetas = self.posterior(test_docs, sym=sym, n=n)
        return [self.theta_output(theta, cascade=cascade) for theta in thetas]
# End of GibbsSampling Class


class VariationalInf:
    def __init__(self):
        pass

# End of Variational_


class CascadeLDA(GibbsSampling):
    def __init__(self, documents):
        super(CascadeLDA, self).__init__(documents)
        self.major_raw = documents
        self.major_dict = self.dict
        self.major_V = self.V
        self.major_phi = np.zeros((1, self.V))
        self.labset = self.major_raw.prepped_labels
        flat_list = [item for sublist in self.labset for item in sublist]
        flat_list = [x for x in flat_list if len(x)==3]
        self.K = len(set(flat_list))

    def run_sub_lda(self, subdocuments, n, thinning):
        sublda = GibbsSampling(documents=subdocuments, cascade=True,
                               majordict=self.major_dict)
        sublda.run(nsamples=n, thinning=thinning)
        return sublda.phi_hat

    def save_sub_phi(self, subphi):
        self.major_phi = np.vstack((self.major_phi, subphi))

    def subset_corpus(self, labsub):
        keepthese = [labsub in doclab for doclab in self.labset]
        subdocs = list(compress(self.major_raw.docs, keepthese))
        sublabs = list(compress(self.major_raw.lab, keepthese))
        cut_prepped = [self.get_labs(labsub, x) for x in self.major_raw.lab]
        cut_prepped = [x for x in cut_prepped if len(x) != 0]

        subraw = copy(self.major_raw)
        subraw.docs = subdocs
        subraw.lab = sublabs
        subraw.prepped_labels = cut_prepped
        return subraw

    def get_all_l2(self):
        l2_labs = [re.findall('[A-Z][0-9]', x) for x in self.major_raw.lab]
        l2_labs = [item for sublist in l2_labs for item in sublist]
        return list(np.sort(list(set(l2_labs))))

    def check_phi_shape(self, ph_hat, raw):
        l3_sub = [x for sublist in raw.prepped_labels for x in sublist]
        labs = len(set(l3_sub))
        dim = ph_hat.shape
        assert dim[0] == labs, "Not all labels are incorporated in this LDA" \
                               " model. %s" % labs
        assert dim[1] == self.major_V, "Not all words in the corpus are" \
                                       "incorporated in this LDA model" \
                                       "%s" % labs

    def run_cascade(self, n, thinning):
        for label_sub in self.get_all_l2():
            sub_raw = self.subset_corpus(label_sub)
            phi_hat_sub = self.run_sub_lda(subdocuments=sub_raw,
                                           n=n, thinning=thinning)
            self.check_phi_shape(ph_hat=phi_hat_sub, raw=sub_raw)

            self.save_sub_phi(phi_hat_sub)
            print("Just finished LDA for the labels ", label_sub)
        self.phi_hat = self.major_phi[1:]

    def get_regex(self, label):
        return label + "[0-9]{1}"

    def get_labs(self, label, lablist):
        reg = self.get_regex(label)
        try:
            return re.findall(reg, lablist)
        except AttributeError:
            pass


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
        - Check logic of alpha, alpha', beta and gamma
    """
    def __init__(self, documents, K=15, mu=-1, sigma=1, a_prime=1,
                 alpha=0.5, rm_=False, ramage_mix=False, LN=1, ksi=0,
                 perf_alpha=False):
        super(HSLDA_Gibbs, self).__init__(documents, K=K, rm_generic=rm_,
                                          ramage_mix=ramage_mix, LN=LN,
                                          perf_alpha=perf_alpha)
        self.a_prime = a_prime
        self.ksi = ksi     # Running variable: y_{l.d} = 1 iff a_{l,d} > ksi
        self.ramage_mix = ramage_mix
        if self.ramage_mix:
            pass
        else:
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
        self.m_aux = np.empty((self.D, self.K))
        self.m_dot = np.empty((1, self.K))
        self.b = np.random.dirichlet( (self.a_prime)*np.ones(self.K), size=1 )

    def sample_a_old(self):
        """
        Resample all running variables a_{l,d}, by drawing from a truncated
        normal distribution, that's specified by the correspond y_{l,d} value
        and the value of its parent label
        :return: A new value for a_{l,d}
        """
        parents = self.Y[list(self.parent_dict.values()), :]
        child = self.Y[list(self.parent_dict.keys()), :]
        func_ind = (child == (parents==1)*1)

        a = np.where(func_ind, self.zbarT_etaL, -np.inf)
        b = np.where(func_ind, np.inf, -self.zbarT_etaL)

        self.a_ld = rvs(a, b, self.zbarT_etaL)

    def sample_a(self):
        """
        Resample a_{l,d} only based on its own values y_{l,d} ignoring parent
        labels (during training phase).
        :return: new draws for a_{l,d}
        """
        a = np.where(self.Y > 0, -self.zbarT_etaL, -np.inf)
        b = np.where(self.Y > 0, np.inf, -self.zbarT_etaL)

        self.a_ld = rvs(a, b, self.zbarT_etaL)

    def sample_m_dot(self, func):
        """
        Take the word-topic assignments z and the doc-topic priors alpha and b
        to sample values for m_{d,k}, based on the Antoniak distribution.
        See Teh et al (2006) and Wallach (2009) for more details

        :return: (np.ndarray): DxK array - updated version of self.m_aux:
        """
        stirl_it_up = get_stirling_nrs(100)
        for k in range(self.K):
            ab = self.alpha * self.b[0, k]
            sub = self.n_zxd[k, :]
            for d in range(self.D):
                # print("Ran topic %s, doc %s"% (k, d))
                n_dk = sub[d]
                if n_dk > 99:
                    stirl_it_up = get_stirling_nrs(n_dk+1)
                if int(n_dk) == 0:
                    self.m_aux[d, k] = 0
                if int(n_dk):
                    self.m_aux[d, k] = rand_antoniak(param=ab,
                                                mm=n_dk,
                                                stirling_matrix=stirl_it_up)
        self.m_dot = np.apply_along_axis(func, 0, self.m_aux)

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
        if self.ramage_mix:
            l = self.n_zxd[:, d] + self.alpha[:, d]
        else:
            l = self.n_zxd[:, d] + (self.alpha*self.b)[0]
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
                self.save_this_state(N=int(intersave), hslda=True)

            # 1) Sample new word-assignments z_{d,n}
            for d in range(self.D):
                # Find the labels that are part of document d's label set:
                lab_d = np.where(self.Y[:, d] == 1)[0]
                invD = 1/self.lenD[d]
                # Only focus on document d and labels in d's label set:
                z_eta = self.zbarT_etaL[lab_d, d]
                sub_a = self.a_ld[lab_d, d]
                assert all(sub_a>self.ksi), "Something went wrong. a_ld must" \
                                              " be positive if y_ld equal one!"
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

                        new_z = np.random.multinomial(1, prob).argmax()

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
            #print("Updating the Hierarchical Dirichlet prior")
            #self.sample_m_dot(func=np.mean)
            #self.sample_b()
            #self.get_perplexity()

    #def get_perplexity(self, testdocs):
    #    th = get_theta()
    #    ph = get_phi()


# End of HSLDA_Gibbs


class UnseenPosterior(HSLDA_Gibbs):
    """
    All intermed_a may be unnecessary!
    """
    def __init__(self, hslda_trained):
        if not isinstance(hslda_trained, HSLDA_Gibbs):
            raise TypeError("hslda_trained must be an instance of HSLDA_Gibbs")
        self.newdocs = None
        self.post_z = None
        self.z_prob = None
        self.z_probs = None

        self.hs = hslda_trained
        self.intermed_a = None
        self.test_a = None

        lab_levels = [len(x) for x in list(self.ldict.values())]
        self.L0_labs = np.where(np.array(lab_levels) == 0)[0]
        self.L1_labs = np.where(np.array(lab_levels) == 1)[0]
        self.L2_labs = np.where(np.array(lab_levels) == 2)[0]
        self.L3_labs = np.where(np.array(lab_levels) == 3)[0]
        self.a_hat = None

    def __getattr__(self, name):
        try:
            return getattr(self.hs, name)
        except AttributeError:
            print("Child' object has no attribute %s" % name)

    def get_prob_z(self, newdoc):
        glob_prior_p_z = (self.n_z / np.sum(self.n_z))[:, np.newaxis]
        word_nrs = [self.dict.token2id[word] for word in newdoc]

        z_p = self.phi_hat[:, word_nrs] * glob_prior_p_z
        z_p = np.apply_along_axis(lambda x: x / np.sum(x), 0, z_p)
        z_p = np.apply_along_axis(np.sum, 1, z_p)
        z_p /= np.sum(z_p)
        return z_p

    def get_probs_z(self, newdocs):
        self.clean_new_docs(newdocs)
        self.post_z = np.array(list(map(self.get_prob_z, self.newdocs)))
        # Fill a container for upcoming sampling a_{l,d}
        self.intermed_a = self.set_up_a_container(newdocs)
        self.test_a = self.set_up_a_container(newdocs)

    def set_up_a_container(self, newdocs):
        a_cont = np.zeros((self.nr_of_labs, len(newdocs)))
        a_cont[0, :] = 1
        return a_cont

    def samp_a_per_level(self, par_level, own_level, all_means):
        inter_a = self.intermed_a
        level_means = all_means[:, own_level]
        a = -np.inf

        par_inds = np.array(list(self.parent_dict.values()))[own_level]
        b = np.where(inter_a[par_inds, :] > 0, np.inf, -level_means)
        inter_a[own_level, :] = rvs(a, b, level_means)

    def posterior_a_sampling(self, samples=200):
        all_means = np.dot(self.post_z, self.eta_hat).T
        L1 = self.L1_labs
        L2 = self.L2_labs
        L3 = self.L3_labs
        self.a_hat = copy(self.test_a)
        for n in range(samples):
            self.a_all_down(L1, all_means)
            self.a_all_down(L2, all_means)
            self.a_all_down(L3, all_means)
            if n > 1:
                self.a_hat *= (n-1)/n
                self.a_hat += (1/n)*self.test_a
                print("Worked on posterior sample ", n)

    def a_all_down(self, own_inds, all_means):
        # Get the mean and realizations of a_ld
        all_a = self.test_a
        # Identify the label ID of the parents to
        par_inds = np.array(list(self.parent_dict.values()))[own_inds]
        parents_a = all_a[par_inds, :]
        # Focus only on the mean for a_ld, for l in this level's hierarchy
        own_means = all_means[own_inds, :]
        # Find the truncation borders for every a_ld in this hierarchy
        a = -np.inf
        b = np.where(parents_a > 0, np.inf, -own_means)
        own_a = rvs(a, b, own_means)
        # Check for every label in higher hierarchy, if at least on descendant
        # has a positive value. Every final label is leaf node in label tree
        parent_set = list(set(par_inds))
        label_level = [len(self.ldict[x]) for x in parent_set]
        assert(check_equal(label_level)), "The labels in the hierarchy should" \
                                          " not be in the same hierarchy.."
        for parent in parent_set:
            children = np.where(par_inds == parent)[0]
            pos_parents = np.where(all_a[parent, :] > 0)[0]
            trouble_docs = [1]
            while (len(trouble_docs) > 0):
                pos_desc_per_doc = np.sum(own_a[children, :] > 0, axis=0)

                no_descents = np.where(pos_desc_per_doc == 0)[0]
                if len(no_descents) == 0:
                    all_a[own_inds, :] = own_a
                    break
                trouble_docs = np.array(
                    list(set(no_descents) & set(pos_parents)))
                new_a = normal(own_means[np.ix_(children, trouble_docs)])
                own_a[np.ix_(children, trouble_docs)] = new_a
        self.test_a[own_inds, :] = own_a

    def one_label_pred(self, doc, threshold=0):
        lab_doc_d = np.where(self.a_hat[:, doc] > threshold)[0]
        ldict_lst = list(self.ldict.values())
        return [ldict_lst[x] for x in lab_doc_d]

    def all_lab_preds(self):
        allpreds = [self.one_label_pred(d) for d in range(len(self.newdocs))]
        return  [[x for x in docpred] for docpred in allpreds]

    def prob_y_hat(self, threshold):
        all_means = np.dot(self.post_z, self.eta_hat)
        return norm.cdf(threshold, -all_means)