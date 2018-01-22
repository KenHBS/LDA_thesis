from CascadeLDA import *
from evaluate_CascadeLDA import *
import numpy as np
from optparse import OptionParser


def main():
    parser = OptionParser()
    parser.add_option("-f", dest="file", help="dataset location")
    parser.add_option("-d", dest="lvl", help="depth of label level")
    parser.add_option("-i", dest="it", help="# of iterations - train and test")
    parser.add_option("-s", dest="thinning", help="inter saving frequency")
    parser.add_option("-l", dest="lower",
                      help="lower threshold for dictionary pruning")
    parser.add_option("-u", dest="upper",
                      help="upper threshold for dictionary pruning")

    (opt, arg) = parser.parse_args()
    train, test = split_data(f=opt.f)
    model = train_it(train, it=opt.it, s=opt.thinning, l=opt.lower, u=opt.upper)

    print("Testing test data, this may take a while")
    l1, l2, l3 = zip(*[model.test_down_tree(x, it=opt.it, thinning=opt.thinning, threshold=0.95) for x in test[0]])

    # Evaluate quality for all label depths:
    d = int(opt.lvl)
    label_depths = list(range(1, d+1))
    for depth in label_depths:
        c = "Full texts"
        if opt.f == "thesis_data3.csv":
            c = "Abstracts"

        print("Model:              CascadeLDA")
        print("Corpus:             ", c)
        print("Label depth         ", depth)
        print("# of Gibbs samples: ", int(opt.it))
        print("-----------------------------------")

        lab_level = [len(x)==depth for x in model.labelmap.keys()]
        inds = np.where(lab_level)[0]

        y_bin = binary_yreal(test[1], model.labelmap)
        th_hat = setup_theta(l1, l2, l3, model)

        # Selecting the relevant labels
        y_bin = y_bin[:, inds]
        th_hat = th_hat[:, inds]

        # Remove no-prediction documents:
        doc_id = np.where(th_hat.sum(axis=1) != 0)[0]
        y_bin = y_bin[doc_id, :]
        th_hat = th_hat[doc_id, :]

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
