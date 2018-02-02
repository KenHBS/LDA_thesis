from CascadeLDA import *
from sklearn.metrics import auc
from optparse import OptionParser
import pickle


def one_roc(prob, real_binary):
    resorted = np.argsort(prob)[::-1]

    reals = real_binary[resorted]
    probs = prob[resorted]
    thresholds = np.sort(list(set(probs)))[::-1]

    tp = []
    tn = []
    fp = []
    fn = []
    for c in thresholds:
        preds = [1 if x >= c else 0 for x in probs]
        zipped = list(zip(preds, reals))

        tp_pre = sum([x == y for (x, y) in zipped if x == 1])
        tn_pre = sum([x == y for (x, y) in zipped if x == 0])
        fp_pre = sum([x != y for (x, y) in zipped if x == 1])
        fn_pre = sum([x != y for (x, y) in zipped if x == 0])

        tp.append(tp_pre)
        tn.append(tn_pre)
        fp.append(fp_pre)
        fn.append(fn_pre)
    return tp, tn, fp, fn


def fpr_tpr(tp, fp, tn, fn):
    fpr = [x / (x + y) for (x, y) in zip(fp, tn)]
    tpr = [x / (x + y) for (x, y) in zip(tp, fn)]
    return fpr, tpr


def precision_recall(tp, fp, tn, fn):
    precis = [x / (x + y) for (x, y) in zip(tp, fp)]
    recall = [x / (x + y) for (x, y) in zip(tp, fn)]
    return precis, recall


def rates(y_prob, y_real_binary):
    tps = []
    tns = []
    fps = []
    fns = []
    fprs = []
    tprs = []
    for d_prob, d_real in zip(y_prob, y_real_binary):
        tp, tn, fp, fn = one_roc(d_prob, d_real)
        fpr, tpr = fpr_tpr(tp, fp, tn, fn)

        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
        fprs.append(fpr)
        tprs.append(tpr)
    return tps, tns, fps, fns, fprs, tprs


def macro_auc_roc(fprs, tprs):
    areas_under_curve = [auc(fpr, tpr) for (fpr, tpr) in zip(fprs, tprs)]
    return np.mean(areas_under_curve)


def n_error(th_hat, y_real_binary, n):
    ndocs = th_hat.shape[0]
    counter = 0
    for i in range(ndocs):
        ordered = np.argsort(th_hat[i, :])[::-1]
        toplabs = ordered[:n]
        sub_y = y_real_binary[i, :]
        hit = sum(sub_y[toplabs]) > 0
        if hit:
            counter += 1
    return counter / ndocs


def get_f1(tps, fps, tns, fns):
    f1 = []
    for tp, fp, tn, fn in zip(tps, fps, tns, fns):
        prec, rec = precision_recall(tp, fp, tn, fn)
        with np.errstate(invalid='ignore'):
            raw_f1 = [(2 * p * r)/(p + r) for p, r in zip(prec, rec)]
        opt_f1 = np.nanmax(raw_f1)
        f1.append(opt_f1)
    return np.mean(f1)


def setup_theta(l1p, l2p, l3p, model):
    # Start adding the lowest labs and just add the 'rest', too. It will be
    # overwritten later on with the correct value from the upper level
    n = len(l1p)
    k = len(model.labelmap)
    th_hat = np.zeros((n, k), dtype=float)

    for d in range(n):
        sub_th = th_hat[d, :]
        levels = dict()
        for tuplist in l3p[d]:
            levels.update(tuplist)
        for tuplist in l2p[d]:
            levels.update(tuplist)
        levels.update(l1p[d])

        # Multiple probs of local scope with the prob of upper level:
        predecessors = [s for (s, t) in l1p[d]]
        lookup = " ".join(list(levels.keys()))
        for p in predecessors:
            pat = re.compile("(" + p + "[0-9])(?:[^0-9]|$)")
            currents = re.findall(pat, lookup)
            for c in currents:
                levels[c] *= levels[p]
                pat = re.compile(c + "[0-9]")
                finals = re.findall(pat, lookup)
                for f in finals:
                    levels[f] *= levels[c]

        labs, probs = zip(*levels.items())
        inds = [model.labelmap[x] for x in labs]
        sub_th[inds] = probs
    return th_hat


def binary_yreal(label_strings, label_dict):
    ndoc = len(label_strings)
    ntop = len(label_dict)
    y_true = np.zeros((ndoc, ntop), dtype=int)
    for d, lab in enumerate(label_strings):
        for l in lab:
            try:
                ind = label_dict[l]
                y_true[d, ind] = 1
            except KeyError:
                pass
    return y_true


def main():
    parser = OptionParser()
    parser.add_option("-f", dest="file", help="dataset location")
    parser.add_option("-d", dest="lvl", type="int",
                      help="depth of label level", default=3)
    parser.add_option("-i", dest="it", type="int",
                      help="# of iterations - train and test")
    parser.add_option("-s", dest="thinning", type="int",
                      help="inter saving frequency", default=0)
    parser.add_option("-a", dest="alpha", type="float", help="alpha prior",
                      default=0.1)
    parser.add_option("-b", dest="beta", type="float", help="beta prior",
                      default=0.01)
    parser.add_option("-l", dest="lower", type="float",
                      help="lower threshold for dictionary pruning", default=0)
    parser.add_option("-u", dest="upper", type="float",
                      help="upper threshold for dictionary pruning", default=1)
    parser.add_option("-p", action="store_true", dest="pickle",
                      help="save pickle of model?", default=False)

    (opt, arg) = parser.parse_args()

    if opt.thinning == 0:
        opt.thinning = opt.it
    train, test = split_data(f=opt.file)
    model = train_it(train, it=opt.it, s=opt.thinning,
                     l=opt.lower, u=opt.upper, al=opt.alpha, be=opt.beta)

    print("Testing test data, this may take a while")
    l1, l2, l3 = zip(*[model.test_down_tree(x, it=opt.it, thinning=opt.thinning, threshold=0.95) for x in test[0]])
    if opt.pickle:
        pickle.dump(model, open("Cascade_model.pkl", "wb"))
        pickle.dump(test, open("Cascade_testset.pkl", "wb"))
        pickle.dump(l1, open("Cascade_d1_pred.pkl", "wb"))
        pickle.dump(l2, open("Cascade_d2_pred.pkl", "wb"))
        pickle.dump(l3, open("Cascaed_d3_pred.pkl", "wb"))

    # Evaluate quality for all label depths:
    d = int(opt.lvl)
    label_depths = list(range(1, d+1))
    for depth in label_depths:
        c = "Full texts"
        if opt.file == "thesis_data3.csv":
            c = "Abstracts"

        print("Model:               CascadeLDA")
        print("Corpus:             ", c)
        print("Label depth         ", depth)
        print("# of Gibbs samples: ", int(opt.it))
        print("-----------------------------------")

        lab_level = [len(x) == depth for x in model.labelmap.keys()]
        inds = np.where(lab_level)[0]

        y_bin = binary_yreal(test[1], model.labelmap)
        th_hat = setup_theta(l1, l2, l3, model)

        # Selecting the relevant labels
        y_bin = y_bin[:, inds]
        th_hat = th_hat[:, inds]

        # Remove no-prediction and no-label documents
        doc_id1 = np.where(th_hat.sum(axis=1) != 0)[0]
        doc_id2 = np.where(y_bin.sum(axis=1) != 0)[0]
        valid = np.intersect1d(doc_id1, doc_id2)

        y_bin = y_bin[valid, :]
        th_hat = th_hat[valid, :]

        tps, tns, fps, fns, fprs, tprs = rates(th_hat, y_bin)

        one_err = n_error(th_hat, y_bin, 1)
        two_err = n_error(th_hat, y_bin, 2)
        auc_roc = macro_auc_roc(fprs, tprs)
        f1_macro = get_f1(tps, fps, tns, fns)

        print("AUC ROC:                 ", auc_roc)
        print("one error:               ", one_err)
        print("two error:               ", two_err)
        print("F1 score (macro average) ", f1_macro)


if __name__ == "__main__":
    main()
