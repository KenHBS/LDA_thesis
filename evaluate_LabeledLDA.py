from LabeledLDA import *
from sklearn.metrics import auc
from optparse import OptionParser
import pickle
import numpy as np


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
    parser.add_option("-d", dest="lvl", type="int", default=3,
                      help="depth of lab level")
    parser.add_option("-i", dest="it", type="int", help="# of iterations")
    parser.add_option("-s", dest="thinning", type="int", default=0,
                      help="save frequency")
    parser.add_option("-l", dest="lower", type="float", default=0,
                      help="lower threshold for dictionary pruning")
    parser.add_option("-u", dest="upper", type="float", default=1,
                      help="upper threshold for dictionary pruning")
    parser.add_option("-a", dest="alpha", type="float", default=0.1,
                      help="alpha prior")
    parser.add_option("-b", dest="beta", type="float", default=0.01,
                      help="beta prior")
    parser.add_option("-p", action="store_true", dest="pickle", default=False,
                      help="Save the model as pickle?")
    (opt, arg) = parser.parse_args()
    if opt.thinning == 0:
        opt.thinning = opt.it

    train, test = split_data(f=opt.file, d=opt.lvl)

    print("Starting training...")
    model = train_it(train, it=opt.it, s=opt.thinning,
                     al=opt.alpha, be=opt.beta, l=opt.lower, u=opt.upper)

    print("Testing test data, this may take a while...")
    th, _ = test_it(model, test, it=opt.it, thinning=opt.thinning)
    th = np.array(th)
    if opt.pickle:
        pickle.dump(model, open("LabeledLDA_model.pkl", "wb"))
        pickle.dump(test, open("LabeledLDA_testset.pkl", "wb"))
        pickle.dump(th, open("LabeledLDA_theta.pkl", "wb"))

    c = "Full Texts"
    if opt.file == "thesis_data3.csv":
        c = "Abstracts"

    print("Model:               Labeled LDA")
    print("Corpus:             ", c)
    print("Label depth         ", opt.lvl)
    print("# of Gibbs samples: ", int(opt.it))
    print("-----------------------------------")

    y_bin = binary_yreal(test[1], model.labelmap)

    # Remove root label from predictions (also not included in label sets)
    y_bin = y_bin[:, 1:]
    th = th[:, 1:]

    # Remove docs that were assigned to 'root' completely:
    nonzero_load = [x != 0 for x in th.sum(axis=1)]
    nonzero_load = np.where(nonzero_load)[0]
    y_bin = y_bin[nonzero_load, :]
    th = th[nonzero_load, :]

    tps, tns, fps, fns, fprs, tprs = rates(th, y_bin)

    one_err = n_error(th, y_bin, 1)
    two_err = n_error(th, y_bin, 2)
    auc_roc = macro_auc_roc(fprs, tprs)
    f1_macro = get_f1(tps, fps, tns, fns)

    print("AUC ROC:                 ", auc_roc)
    print("one error:               ", one_err)
    print("two error:               ", two_err)
    print("F1 score (macro average) ", f1_macro)


if __name__ == "__main__":
    main()
