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

# Prepare textual data for corpus: tokenization etc.

# 1) For regular LDA:
# rawdata_lda = doc_prepare.PrepLabeledData(train_data, level=3)
# cgs = GibbsSampling(documents=rawdata_lda)
# cgs.run(nsamples=10)

# 2) For HSLDA:
rawdata = doc_prepare.PrepLabeledData(train_data, hslda=True)
hslda = HSLDA_Gibbs(documents=rawdata)
hslda.sample_to_next_state(nsamples=5)
# Prepare & Run Collapsed Gibbs Sampling on training data


# wordspertopic = cgs.get_topiclist()
# print(wordspertopic)

# Prepare test data for HSLDA:
new_docs, new_labs = doc_prepare.PrepLabeledData.split_testdata(test_data)
test_docs = doc_prepare.new_docs_prep(new_docs=new_docs, lda_dict=hslda.dict)
labels = [label.split(" ") for label in new_labs]
# labels = [[label[:2] for label in doclab] for doclab in labels]
# test_labs = [list(set(label)) for label in labels]


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

