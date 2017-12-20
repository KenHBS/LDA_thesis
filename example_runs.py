from BaseLDA import *


## With complete depth:
train_data, test_data = split_data(f="thesis_data.csv", d=3)

llda = train_it(train_data, it=10, thinning=2, al=0.001, be=0.001)
th_hat, preds = test_it(llda, test_data, it=30, thinning=6)


## With depth to second level:
train_data_d2, test_data_d2 = split_data(f="thesis_data.csv", d=2)

llda_d2 = train_it(train_data_d2, it=1000, thinning=99, al=0.001, be=0.001)
th_hat_d2, preds_d2 = test_it(llda_d2, test_data_d2, it=1000, thinning=99)

