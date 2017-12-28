from BaseLDA import *


## With complete depth:
train_data, test_data = split_data(f="thesis_data.csv", d=3)

llda = train_it(train_data, it=10, thinning=2, al=0.001, be=0.001)
th_hat, preds = test_it(llda, test_data, it=30, thinning=6)


## With depth to second level:
train_data_d2, test_data_d2 = split_data(f="thesis_data.csv", d=2)

llda_d2 = train_it(train_data_d2, it=1000, thinning=99, al=0.001, be=0.001)
th_hat_d2, preds_d2 = test_it(llda_d2, test_data_d2, it=1000, thinning=99)


from HSLDA import *

train_data, test_data = split_data(f="thesis_data.csv", d=3)
hs1 = train_it(train_data, it=500, s=25, opt=1)
hs2 = train_it(train_data, it=500, s=25, opt=2)
hs3 = train_it(train_data, it=500, s=25, opt=3)


# probs1 = test_it(hs1, test_data)
# labprobs1 = []
# for n in range(len(probs1)):
#     labprobs1.append([(x, y) for x, y in hs1.label_predictions(probs1[n]) if y in test_data[1][n]])
#
# probs2 = test_it(hs2, test_data)
# labprobs2 = []
# for n in range(len(probs2)):
#     labprobs2.append([(x, y) for x, y in hs2.label_predictions(probs2[n]) if y in test_data[1][n]]))
#
# probs3 = test_it(hs3, test_data)
# labprobs3 = []
# for n in range(len(probs3)):
#     labprobs3.append([(x, y) for x, y in hs3.label_predictions(probs3[n]) if y in test_data[1][n]]))


from CascadeLDA import *

train_data, test_data = split_data(f="thesis_data.csv", d=3)
cascade = train_it(train_data, it=500, s=25)
theta_l1 = test_it(cascade, test_data, it=500, s=25, depth=1)
theta_l2 = test_it(cascade, test_data, it=250, s=15, depth=2)
theta_l3 = test_it(cascade, test_data, it=150, s=15, depth=3)

a, b, c = test_data
print_test_results(theta_l1, 1, cascade.lablist_l1, b, n=25)
print_test_results(theta_l2, 2, cascade.lablist_l2, b, n=25)
print_test_results(theta_l3, 3, cascade.lablist_l3, b, n=25)
