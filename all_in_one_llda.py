from LabeledLDA import *
from evaluate_CascadeLDA import *
import numpy as np
from optparse import OptionParser


def main():
    parser = OptionParser()
    parser.add_option("-f", dest="file", help="dataset location")
    parser.add_option("-d", dest="lvl", type="int", help="depth of lab level")
    parser.add_option("-i", dest="it", type="int", help="# of iterations")
    parser.add_option("-s", dest="thinning", type="int", help="save frequency")
    parser.add_option("-l", dest="lower", type="float",
                      help="lower threshold for dictionary pruning")
    parser.add_option("-u", dest="upper", type="float",
                      help="upper threshold for dictionary pruning")
    parser.add_option("-a", dest="alpha", type="float", help="alpha prior",
                      default=0.001)
    parser.add_option("-b", dest="beta", type="float", help="beta prior",
                      default=0.001)
    (opt, arg) = parser.parse_args()

    train, test = split_data(f=opt.file, d=opt.lvl)

    print("Starting training...")
    model = train_it(train, it=opt.it, s=opt.thinning,
                    al=opt.alpha, be=opt.beta, l=opt.lower, u=opt.upper)

    print("Testing test data, this may take a while...")
    th, _ = test_it(model, test, it=opt.it, thinning=opt.thinning)
    th = np.array(th)

    c = "Full Texts"
    if opt.file == "thesis_data3.csv":
        c = "Abstracts"

    print("Model:              Labeled LDA")
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
