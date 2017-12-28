import gensim.parsing.preprocessing as gensimm
import numpy as np
multinom_draw = np.random.multinomial


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


def partition_label(lab, d):
    return [lab[:i+1] for i in range(d)]


class CascadeLDA(object):
    def __init__(self, docs, labs, labelset, alpha=0.001, beta=0.001):
        labelset.insert(0, 'root')
        self.labelmap = dict(zip(labelset, range(len(labelset))))
        self.K = len(self.labelmap)
        self.lablist = labelset

        self.alpha = alpha
        self.beta = beta

        self.vocab = []
        self.w_to_v = dict()
        self.labs = np.array([self.set_label(lab) for lab in labs])
        self.docs = [[self.term_to_id(term) for term in doc] for doc in docs]
        self.v_to_w = {v:k for k, v in self.w_to_v.items()}

        self.D = len(docs)
        self.V = len(self.vocab)

        self.ph = np.zeros((self.K, self.V), dtype=float)

        self.l1 = [[l1 for l1 in lab if len(l1) == 1] for lab in labs]
        self.l2 = [[l2 for l2 in lab if len(l2) == 2] for lab in labs]
        self.l3 = [[l3 for l3 in lab if len(l3) == 3] for lab in labs]

        self.lablist_l1 = [x for x in self.lablist if len(x) == 1]
        self.lablist_l2 = [x for x in self.lablist if len(x) == 2]
        self.lablist_l3 = [x for x in self.lablist if len(x) == 3]

        self.rawlabs = labs

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

    def sub_corpus(self, parent):
        level = len(parent)
        if level == 0:
            lab_level = self.l1
        elif level == 1:
            lab_level = self.l2
        elif level == 2:
            lab_level = self.l3
        present = np.where([[parent in lab] for lab in self.rawlabs])[0]
        docs = [self.docs[p] for p in present]
        labs = [lab_level[p] for p in present]

        # Only keep the target labels, remove all other labels: they will be
        # gathered as the 'generic' topic
        labs = [[x for x in lab if x[:level] == parent] for lab in labs]
        labset = sorted(list(set([x for sub in labs for x in sub])))
        return docs, labs, labset

    def get_sub_ph(self, docs, labs, labset, it=150, thinning=12):
        sublda = SubLDA(docs, labs, labset, self.v_to_w)
        sublda.run_training(it=it, thinning=thinning)
        return sublda.get_ph()

    def go_down_tree(self, it, s):
        # Starting at 'root' as parent node:
        docs = self.docs
        labs = self.l1
        labset = self.lablist_l1
        sub_ph = self.get_sub_ph(docs, labs, labset, it=it, thinning=s)
        label_ids = [self.labelmap[x] for x in labset]
        self.ph[label_ids, :] = sub_ph
        for l in self.lablist_l1:
            print("Working on parent node", l)
            print(" --- ")
            docs, labs, labset = self.sub_corpus(parent=l)
            sub_ph = self.get_sub_ph(docs, labs, labset, it=it, thinning=s)
            label_ids = [self.labelmap[x] for x in labset]
            self.ph[label_ids, :] = sub_ph
            one_down = [x for x in self.lablist_l2 if x[0] == l]
            for l2 in one_down:
                print("Working on parent node", l2)
                print(" --- ")
                docs, labs, labset = self.sub_corpus(parent=l2)
                sub_ph = self.get_sub_ph(docs, labs, labset, it=it, thinning=s)
                label_ids = [self.labelmap[x] for x in labset]
                self.ph[label_ids, :] = sub_ph

    def test_init_newdoc(self, doc, ph):
        doc = [x for x in doc if x in self.vocab]
        doc = [self.w_to_v[term] for term in doc]
        n_d = len(doc)
        z_dn = []
        l = ph.shape[0]
        n_zk = np.zeros(l, dtype=int)
        probs = ph[:, doc]
        probs /= probs.sum(axis=0)
        for n in range(n_d):
            prob = probs[:, n]
            while prob.sum() > 1:
                prob /= 1.0000005
            new_z = multinom_draw(1, prob).argmax()

            z_dn.append(new_z)
            n_zk[new_z] += 1
        start_state = (doc, z_dn, n_zk)
        return start_state

    def run_test(self, docs, it, thinning, depth="all"):
        if depth in [1, 2, 3]:
            label_inds = np.where([len(x) in [depth, 4] for x in self.lablist])[0]
        if depth == "all":
            label_inds = range(self.K)
        nrdocs = len(docs)
        ph = self.ph[label_inds, :]
        l = len(label_inds)
        th_hat = np.zeros((nrdocs, l), dtype=float)
        for d, doc in enumerate(docs):
            newdoc, z_dn, n_zk = self.test_init_newdoc(doc, ph)
            n_d = len(z_dn)
            for i in range(it):
                for n, v in enumerate(newdoc):
                    v = int(v)
                    z = z_dn[n]
                    n_zk[z] -= 1

                    num_a = n_zk + self.alpha
                    b = ph[:, v]
                    prob = num_a * b
                    prob /= prob.sum()
                    while prob.sum() > 1:
                        prob /= 1.000005
                    new_z = multinom_draw(1, prob).argmax()

                    z_dn[n] = new_z
                    n_zk[new_z] += 1

                # Save the current state in MC chain and calc. average state:
                s = (i+1) / thinning
                if s == int(s):
                    print("Testing iteration #", i+1)
                    print("----")
                    cur_th = n_zk / n_d
                    if s > 1:
                        m = (s-1)/s
                        th = m * th + (1-m) * cur_th
                    else:
                        th = cur_th
            th_hat[d, :] = th
        return th_hat


class SubLDA(object):
    def __init__(self, docs, labs, labelset, v_to_w, alpha=0.001, beta=0.001):
        labelset.insert(0, 'root')
        self.labelmap = dict(zip(labelset, range(len(labelset))))
        self.K = len(self.labelmap)
        self.lablist = labelset

        self.alpha = alpha
        self.beta = beta

        self.labs = np.array([self.set_label(lab) for lab in labs])
        self.docs = docs
        self.v_to_w = v_to_w

        self.V = len(v_to_w)
        self.D = len(docs)

        self.z_dn = []
        self.n_d_k = np.zeros((self.D, self.K), dtype=int)
        self.n_k_v = np.zeros((self.K, self.V), dtype=int)
        self.n_zk = np.zeros(self.K, dtype=int)

        self.ph = np.zeros((self.K, self.V), dtype=float)

        for d, doc, lab in zip(range(self.D), self.docs, self.labs):
            len_d = len(doc)
            prob = lab / lab.sum()
            zets = np.random.choice(self.K, size=len_d, p=prob)
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

    def get_ph(self):
        return self.n_k_v / self.n_k_v.sum(axis=1, keepdims=True)

    def training_iteration(self):
        for d, doc, lab in zip(range(self.D), self.docs, self.labs):
            n_d = len(doc)
            for n in range(n_d):
                v = doc[n]
                z = self.z_dn[d][n]
                self.n_d_k[d, z] -= 1
                self.n_k_v[z, v] -= 1
                self.n_zk[z] -= 1

                num_a = self.n_d_k[d] + self.alpha
                den_a = n_d - 1 + self.K * self.alpha
                num_b = self.n_k_v[:, v] + self.beta
                den_b = self.n_zk + self.V * self.beta

                prob = lab * num_a / den_a * num_b / den_b
                prob /= np.sum(prob)
                z_new = multinom_draw(1, prob).argmax()

                self.z_dn[d][n] = z_new
                self.n_d_k[d, z_new] += 1
                self.n_k_v[z_new, v] += 1
                self.n_zk[z_new] += 1

    def run_training(self, it=120, thinning=15):
        for i in range(it):
            self.training_iteration()
            s = (i+1) / thinning
            if s == int(s):
                print("Training iteration #", i+1)
                cur_ph = self.get_ph()
                if s > 1:
                    m = (s-1)/s
                    self.ph = m * self.ph + (1-m) * cur_ph
                else:
                    self.ph = cur_ph


def split_data(f="thesis_data.csv", d=3):
    a, b, c = load_corpus(filename=f, d=d)
    split = int(len(a) * 0.9)
    train_data = (a[:split], b[:split], c)
    test_data = (a[split:], b[split:], c)
    return train_data, test_data


def train_it(train_data, it=150, s=12):
    a, b, c = train_data
    cascade = CascadeLDA(a, b, c)
    cascade.go_down_tree(it=it, s=s)
    return cascade


def test_it(model, test_data, it=150, s=12, depth=3):
    a, b, c = test_data
    th_hat = model.run_test(docs=a, it=it, thinning=s, depth=depth)
    return th_hat

def print_test_results(theta, depth, lablist, testlabels, n=10):
    for x in range(n):
        print(sorted(zip(theta[x], lablist))[::-1])
        print([lab for lab in testlabels[x] if len(lab) == depth])
        print("---")
