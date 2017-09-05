import numpy as np
from gensim.parsing import preprocessing
from gensim import corpora, matutils
import collections


# Static methods:
def dir_draw(array_in, axis=0):
    return np.apply_along_axis(np.random.dirichlet, axis=axis, arr=array_in)

def multin_draw(param, size):
    return np.random.multinomial(1, param, size=size)


# Three classes:
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
    # TODO remove standard LDA options without prior knowledge
    """
    First try to get a class that will run a collapsed Gibbs sampler.
        Optional are:
            asymmetric prior on label frequency
            prior knowledge on labels (Ramage ')
    """
    def __init__(self, documents):
        if not isinstance(documents, DocDump):
            raise TypeError("Gibbs only takes DocDump instances as input")

        self.docs = preprocessing.preprocess_documents(documents.docs)
        self.dict = corpora.Dictionary(self.docs)
        self.corpus = [self.dict.doc2bow(doc) for doc in self.docs]

        self.labs = documents.prepped_labels
        self.ldict = corpora.Dictionary(self.labs)

        self.V = len(self.dict)
        self.K = len(self.ldict)

        _ = [self.ldict.doc2bow(label) for label in self.labs]
        self.alpha = matutils.corpus2dense(_, self.K)
        # Rearrange so that col1 is label A, col2 is label B, etc.
        _ = np.argsort([x for x in self.ldict.token2id.keys()])
        self.alpha = self.alpha[_, :]
        _ = 50/np.sum(self.alpha, axis=0)
        self.alpha = self.alpha * _
        self.beta = 200/self.V * np.ones((self.V, self.K))

        self.phi = dir_draw(array_in=self.beta, axis=0)
        self.theta = dir_draw(array_in=self.alpha, axis=0)

        self.zet = self.draw_z()

    def z_size(self):
        lengths = [len(doc) for doc in self.docs]
        return [np.repeat(0, nrwords) for nrwords in lengths]

    def draw_z(self):
        z = self.z_size()
        for d, doc in enumerate(self.docs):
            bin_doc = multin_draw(param=self.theta[:, d], size=len(doc))
            for w, word in enumerate(bin_doc):
                z[d][w] = np.flatnonzero(word)
        return z
# endclass GibbsStart


class GibbsSampling(GibbsStart):
    def __init__(self, documents, alpha_strength=50):
        GibbsStart.__init__(self, documents)

        # Classic LDA variables: nr of docs, nr of topics,
        self.D = len(self.docs)

        self.multiply_a = alpha_strength
        self.beta_const = self.beta[1, 1]
        self.lenD = [len(doc) for doc in self.docs]

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
                th[d][z] = (self.DTcnts[d][z] + self.alpha[z][d]) / (self.lenD[d] + self.multiply_a)
        return th

    def get_phi(self):
        ph = np.zeros((self.K, self.V))
        for z in range(self.K):
            for w in range(self.V):
                ph[z][w] = (self.TWcnts[z][w] + self.beta_const) / (self.cntT[z] + self.beta_const*self.V)
        return ph

    def sample_z(self, d, word, pos):
        w = self.dict.token2id[word]
        z = self.zet[d][pos]
        self.TWcnts[z, w] -= 1
        self.DTcnts[d, z] -= 1
        self.cntT[z] -= 1
        self.lenD[d] -= 1

        left_num = self.TWcnts[:, w] + self.beta_const
        left_den = self.cntT + self.beta_const * self.V
        right_num = self.DTcnts[d] + self.alpha[:, d]
        right_den = self.lenD[d] + self.multiply_a

        prob = (left_num / left_den) * (right_num / right_den)
        prob = prob / np.sum(prob)

        new_z = np.random.multinomial(1, prob).argmax()
        self.zet[d][pos] = new_z
        self.TWcnts[new_z, w] += 1
        self.DTcnts[d, new_z] += 1
        self.cntT[new_z] += 1
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
