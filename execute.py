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
cgs.run(nsamples=1)

wordspertopic = cgs.get_topiclist()
print(wordspertopic)


#for name in dir():
#    if not name.startswith('_'):
#        del globals()[name]
