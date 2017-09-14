from CGS import *
import doc_prepare
import evaluate
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
rawdata = doc_prepare.PrepLabeledData(train_data, level=2)
# rawdata.prep_labels(level=2)

# Prepare & Run Collapsed Gibbs Sampling on training data
cgs = GibbsSampling(documents=rawdata)
cgs.run(nsamples=2)

# wordspertopic = cgs.get_topiclist()
# print(wordspertopic)

# Prepare test data:
new_docs, new_labs = doc_prepare.PrepLabeledData.split_testdata(test_data)
test_docs = doc_prepare.new_docs_prep(new_docs=new_docs, lda_dict=cgs.dict)
labels = [label.split(" ") for label in new_labs]
labels = [[label[:2] for label in doclab] for doclab in labels]
test_labs = [list(set(label)) for label in labels]


# Calculate posterior for test data:
thetas = cgs.post_theta(test_docs)
test = list(zip(thetas, test_labs))

### Inspection of the predictions:
preds = [[x[0] for x in theta] for theta in thetas]

hits = list(map(evaluate.recall, preds, test_labs))
print("The exact hit rate on the test data is %.4f%%" % (100*sum(hits)/len(hits)))



#for name in dir():
#    if not name.startswith('_'):
#        del globals()[name]


#lvb = "How does a negative labor demand shock impact individual-level fertility? I analyze this question in the context of the East German fertility decline after the fall of the Berlin Wall in 1989. Exploiting dierential pressure for restructuring across industries, I nd that throughout the 1990s, women more severely impacted by the demand shock had more children on average than their counterparts who were less severely impacted. I argue that in uncertain economic circumstances, women with relatively more favorable labor market outcomes postpone childbearing in order not to put their labor market situations at further risk. This mechanism is relevant for all qualication groups, including high-skilled women. There is some evidence for an impact on completed fertility."
#lvb2 = "This paper compares the effectiveness of date- and state-based forward guidance issued by the Federal Reserve since mid-2011 accounting for the influence of disagreement within the FOMC. Effectiveness is investigated through the lens of interest rates sensitivity to macroeconomic news and I find that the Feds forward guidance reduces the sensitivity and therefore crowds out other public information. The sensitivity shrinkage is stronger in the case of date-based forward guidance due to its unconditional nature. Yet, high levels of disagreement among monetary policy makers as published through the FOMCs dot projections since 2012 partially restore sensitivity to macroeconomic news. Thus, disagreement appears to lower the information content of forward guidance and to weaken the Feds commitment as perceived by financial markets. The dot projections are therefore able to reduce the focal point character of forward guidance."

#lvb_docs = new_docs_prep(new_docs = [lvb, lvb2], lda_dict = cgs.dict)
#lvb_theta = cgs.post_theta(lvb_docs)