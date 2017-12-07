from CGS import *
import doc_prepare
import os.path
import csv
import random

# Import the textual data from csv file
if os.path.isfile("thesis_data.csv"):
    filename = "thesis_data.csv"
    wd = '/Users/Ken/Desktop/LDA_Thesis'
else:
    filename = 'C:\\Users\\schroedk.hub\\Documents\\thesis_data.csv'
    wd = 'C:\\Users\\schroedk.hub\\LDA_Thesis'
with open(filename, 'r') as doc:
    reader = csv.reader(doc)
    reader = list(reader)

# Split training and test data
random.shuffle(reader)
split = int(len(reader)*0.8)
train_data = reader[:split]
test_data = reader[split:]

#import pickle
#hslda2 = pickle.load(open('hslda_250samples.pkl', 'rb'))
#hs = pickle.load(open('hslda_60samples3.pkl', 'rb'))

# Prepare textual data for corpus: tokenization etc.

# 1) For regular LDA:
# rawdata_lda = doc_prepare.PrepLabeledData(train_data, level=3)
# cgs = GibbsSampling(documents=rawdata_lda)
# cgs.run(nsamples=10)



# 2) For HSLDA:
rawdata = doc_prepare.PrepLabeledData(train_data, hslda=True)
hslda = HSLDA_Gibbs(documents=rawdata, rm_=True)
hslda.sample_to_next_state(nsamples=250, thinning=25)
# Prepare & Run Collapsed Gibbs Sampling on training data


# Prepare test data for HSLDA:
test_docs = [x[1] for x in test_data]
test_labs = [x[2] for x in test_data]

post_hs = UnseenPosterior(hslda)
post_hs.get_probs_z(test_docs)

post_hs.posterior_a_sampling()

label_predictions = post_hs.all_lab_preds()

result = [[x for x in docpred if len(x)==3] for docpred in label_predictions]


#new_docs, new_labs = doc_prepare.PrepLabeledData.split_testdata(test_data)
#test_docs = doc_prepare.new_docs_prep(new_docs=new_docs, lda_dict=hslda.dict)
#labels = [label.split(" ") for label in new_labs]


# Calculate posterior for test data:
# thetas = cgs.post_theta(test_docs)
#test = list(zip(thetas, test_labs))

### Inspection of the predictions:
# preds = [[x[0] for x in theta] for theta in thetas]

# hits = list(map(evaluate.recall, preds, test_labs))
# print("The exact hit rate on the test data is %.4f%%" % (100*sum(hits)/len(hits)))



#for name in dir():
#    if not name.startswith('_'):
#        del globals()[name]

## POSTERIOR ON FULL DOCUMENTS:
wd_data = os.path.join(wd, 'data')
filenames = [os.path.join(wd_data, doc) for doc in os.listdir(wd_data)]

docs = [doc_prepare.open_txt_doc(doc) for doc in filenames]
prep_docs = doc_prepare.new_docs_prep(docs, cgs.dict)
fulldoc_thetas = cgs.post_theta(prep_docs)

