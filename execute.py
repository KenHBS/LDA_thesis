from CGS import *
import doc_prepare
import evaluate
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

# Prepare textual data for corpus: tokenization etc.
rawdata = doc_prepare.PrepLabeledData(train_data, level=2)
# rawdata.prep_labels(level=2)

# Prepare & Run Collapsed Gibbs Sampling on training data
cgs = GibbsSampling(documents=rawdata)
cgs.run(nsamples=250)

# wordspertopic = cgs.get_topiclist()
# print(wordspertopic)

# Prepare test data:
# new_docs, new_labs = doc_prepare.PrepLabeledData.split_testdata(test_data)
# test_docs = doc_prepare.new_docs_prep(new_docs=new_docs, lda_dict=cgs.dict)
# labels = [label.split(" ") for label in new_labs]
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

