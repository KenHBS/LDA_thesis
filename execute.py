from CGS import *
import os.path
import csv
import random

# Import the textual data from csv file
if os.path.isfile("thesis_data.csv"):
    filename = "thesis_data.csv"
else:
    filename = 'C:\\Users\\schroedk.hub\\Documents\\thesis_data.csv'
with open(filename, 'r') as doc:
    reader = csv.reader(doc)
    reader = list(reader)

# Split training and test data
random.shuffle(reader)
split = int(len(reader)*0.8)
train_data = reader[:split]
test_data = reader[split:]

# Prepare textual data for corpus: tokenization etc.
rawdata = DocDump(train_data)
rawdata.prep_labels(level=1)

# Prepare & Run Collapsed Gibbs Sampling
cgs = GibbsSampling(documents=rawdata)
cgs.run(nsamples=500)

wordspertopic = cgs.get_topiclist()
# print(wordspertopic)

new_docs, new_labs = split_testdata(test_data)
test_docs = new_docs_prep(new_docs=new_docs, lda_dict=cgs.dict)
labels = [label.split(" ") for label in new_labs]
labels = [[label[:1] for label in doclab] for doclab in labels]
test_labs = [list(set(label)) for label in labels]

# test = cgs.sample_for_posterior(test_docs[0])

thetas = cgs.posterior(test_docs)

list(zip(thetas, test_labs))

# testdoc = test_docs[1]
# testzet, testzcounts = cgs.sample_for_posterior(testdoc)

#for name in dir():
#    if not name.startswith('_'):
#        del globals()[name]

