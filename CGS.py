import numpy as np
from gensim.parsing import preprocessing
from gensim import corpora, matutils
import collections

class DocDump:
    # TODO make this class more elaborate, also available for non-conform data
    def __init__(self, textract):
        self.docs = [x[1] for x in textract]
        self.id = [x[0] for x in textract]
        self.lab = [x[2] for x in textract]
        self.prepped_labels = None

    def prep_labels(self, level = 1):
        """
        :param doc_labels: One list with labels per documents. All lists gathered
        in one big list. Like [["A12 B23 M35"], ["D12 E59"], ["A10"], ... ]
        :return: Well-prepared list for ramage_labels() with proper label depth
        """
        assert level in [1, 2, 3], 'The level of the labels must be 1, 2 or 3'

        labels = [label.split(" ") for label in self.lab]
        labels = [[label[0:level] for label in doclab] for doclab in labels]
        self.prepped_labels = [list(set(label)) for label in labels]
# endclass DocDump

class GibbsStart:
    """
    First try to get a class that will run a collapsed Gibbs sampler.
    Challenges:
        Need to initialize:
            corpus
            dictionary
            label assignments
            label distr. over words


        Optional are:
            asymmetric prior on label frequency
            prior knowledge on labels (Ramage ')


    """
    def __init__(self, documents, K = 20, hyper_alpha = 1):
        # take from database_fetch
        if not isinstance(documents, DocDump):
            raise TypeError("Gibbs only takes DocDump instances as input")

        self.docs = preprocessing.preprocess_documents(documents.docs)
        self.dict = corpora.Dictionary(self.docs)
        self.corpus = [self.dict.doc2bow(doc) for doc in self.docs]

        D = len(self.docs)
        V = len(self.dict)


        # Hyperpriors 1: Alpha
        if(documents.prepped_labels is not None):
            self.labs = documents.prepped_labels
            self.ldict = corpora.Dictionary(self.labs)

            pre_alpha = [self.ldict.doc2bow(label) for label in self.labs]
            self.alpha = matutils.corpus2dense(pre_alpha, len(self.ldict))

            # Rearrange so that col1 is label A, col2 is label B, etc.
            inds = np.argsort([x for x in self.ldict.token2id.keys()])
            self.alpha = self.alpha[inds, :]
            # todo Create a 1x4635 np array with
            _ = 50/np.sum(self.alpha, axis = 0)
            self.alpha = self.alpha * _


        else:
            self.alpha = 50/K * np.ones((K, D))

        self.beta = 200/V * np.ones((V, K))

        self.phi = self.dir_draw(array_in = self.beta, axis = 0)
        self.theta = self.dir_draw(array_in = self.alpha, axis = 0)

        self.zet = self.draw_z()

    def dir_draw(self, array_in, axis = 0):
        return np.apply_along_axis(np.random.dirichlet,
                                   axis = axis, arr = array_in)
    def multin_draw(self, param, size):
        return np.random.multinomial(1, param, size = size)

    def z_size(self):
        lengths = [len(doc) for doc in self.docs]
        return [np.repeat(0, nrwords) for nrwords in lengths]

    def draw_z(self):
        z = self.z_size()
        for d, doc in enumerate(self.docs):
            bin_doc = self.multin_draw(param = self.theta[:, d], size = len(doc))
            for w, word in enumerate(bin_doc):
                z[d][w] = np.flatnonzero(word)
        return z
# endclass GibbsStart

class GibbsSampling:
    def __init__(self, sstate, alpha_strength = 50):
        if not isinstance(sstate, GibbsStart):
            raise TypeError("GibbsSampling requires input of class GibbsStart")
        self.dict = sstate.dict
        self.D = len(sstate.docs)
        self.K = len(sstate.ldict)
        self.V = len(sstate.dict)
        self.alpha = sstate.alpha
        self.multiply_a = alpha_strength
        self.beta = sstate.beta
        self.beta_const = sstate.beta[1,1]
        self.theta = sstate.theta
        self.phi = sstate.phi
        self.docs = sstate.docs
        self.labs = sstate.labs
        self.lenD = [len(doc) for doc in self.docs]
        self.zet = sstate.zet
        _ = sorted(np.concatenate(self.zet).ravel())
        _ = collections.Counter(_)
        self.cntT = [k for k in _.values()]

        self.TWcnts = np.zeros((self.K, self.V))
        for d, doc in enumerate(self.docs):
            for pos, word in enumerate(doc):
                w = self.dict.token2id[word]
                z = self.zet[d][pos]
                self.TWcnts[z, w] += 1

        self.DTcnts = np.zeros((self.D, self.K))
        for d, d_zet in enumerate(self.zet):
            for z in d_zet:
                self.DTcnts[d, z] += 1

    def get_theta(self):
        th = np.zeros((self.D, self.K))
        for d in range(self.D):
            for z in range(self.K):
                th[d][z] = (self.DTcnts[d][z] + self.alpha[z][d]) / ( self.lenD[d] + self.multiply_a)

    def sample_z(self, d, word, pos):
        w = self.dict.token2id[word]
        z = self.zet[d][pos]
        self.TWcnts[z, w] -= 1
        self.DTcnts[d, z] -= 1
        self.cntT[z] -= 1
        self.lenD[d] -= 1

        left_num = ( self.TWcnts[:, w] + self.beta_const )    # one column  of (20 x 9341) matrix. (20 values)
        left_den = ( self.cntT + self.beta_const * self.V )   # list of length 20
        right_num = ( self.DTcnts[d] + self.alpha[:, d] )
        right_den = ( self.lenD[d] + self.multiply_a)  # self.multiply_a because every doc has alpha sum for every doc of self.multiply_a.

        prob = (left_num / left_den) * (right_num / right_den)
        prob = prob / np.sum(prob)

        new_z = np.random.multinomial(1, prob).argmax()
        self.zet[d][pos] = new_z
        self.TWcnts[new_z, w] += 1
        self.DTcnts[d, new_z] += 1
        self.cntT[new_z] += 1
        self.lenD[d] += 1

    def run(self, nsamples, burnin = 0):
        if(nsamples <= burnin):
            raise Exception('Burn-in point exceeds number of samples')
        for s in range(nsamples):
            for d, doc in enumerate(self.docs):
                if(d % 250 == 0):
                    print("Working on document %d in sample number %d " % (d, s+1))
                for pos, word in enumerate(self.docs[d]):
                    self.sample_z(d, word, pos)

