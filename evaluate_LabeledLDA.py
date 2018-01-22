import pickle
import numpy as np
from sklearn.metrics import auc
from optparse import OptionParser



def load_data(prefix):
    name1 = prefix + ".pkl"
    name2 = prefix + "_th_hat.pkl"
    name3 = prefix + "_testdocs.pkl"

    model = pickle.load(open(name1, "rb"))
    th_hat = pickle.load(open(name2, "rb"))
    test_set = pickle.load(open(name3, "rb"))

    th_hat = np.array(th_hat)
    return model, th_hat, test_set


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


def macro_auc_pr(tps, tns, fps, fns):
    precision = []
    recall = []
    for tp, tn, fp, fn in zip(tps, tns, fps, fns):
        one_prec, one_recall = precision_recall(tp, fp, tn, fn)

        precision.append(one_prec)
        recall.append(one_recall)

    zipped = zip(precision, recall)
    areas_under_curve = [auc(pre, rec) for (pre, rec) in zipped]
    return np.mean(areas_under_curve)


def one_error(th_hat, y_real_binary):
    ndocs = th_hat.shape[0]
    counter = 0
    for i in range(ndocs):
        ordered = np.argsort(th_hat[i, :])[::-1]
        toplab = ordered[0]
        hit = (y_real_binary[i, toplab] == 1)
        if hit:
            counter += 1
    return counter / ndocs


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


def main():
    parser = OptionParser()
    parser.add_option("-p", dest="prefix", help="prefix of pickles")

    (options, args) = parser.parse_args()

    m, corpus, d, it = options.prefix.split("_")
    model, th_hat, test_set = load_data(options.prefix)

    c = "Full texts"
    if corpus == "abs":
        c = "Abstracts"
    modelname = "Labeled LDA"

    print("Model:              ", modelname)
    print("Corpus:             ", c)
    print("Label depth:        ", d)
    print("# of Gibbs samples: ", int(it))
    print("------------------------------------")

    y_bin = binary_yreal(test_set[1], model.labelmap)
    # Remove root label from predictions (also not included in label sets)
    y_bin = y_bin[:, 1:]
    th_hat = th_hat[:, 1:]

    # Remove docs that were assigned to 'root' completely:
    nonzero_load = [x != 0 for x in th_hat.sum(axis=1)]
    nonzero_load = np.where(nonzero_load)[0]
    y_bin = y_bin[nonzero_load, :]
    th_hat = th_hat[nonzero_load, :]

    tps, tns, fps, fns, fprs, tprs = rates(th_hat, y_bin)

    one_err = n_error(th_hat, y_bin, 1)
    two_err = n_error(th_hat, y_bin, 2)
    auc_roc = macro_auc_roc(fprs, tprs)
    f1_macro = get_f1(tps, fps, tns, fns)

    print("one error:               ", one_err)
    print("two error:               ", two_err)
    print("AUC ROC:                 ", auc_roc)
    print("F1 score (macro average) ", f1_macro)


if __name__ == "__main__":
    main()
