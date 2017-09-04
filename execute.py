from CGS import *
from copy import deepcopy
import os.path
import csv

if os.path.isfile("thesis_data.csv"):
    filename = "thesis_data.csv"
else:
    filename = 'C:\\Users\\schroedk.hub\\Documents\\thesis_data.csv'

with open(filename, 'r') as doc:
    reader = csv.reader(doc)
    reader = list(reader)


rawdata = DocDump(reader)
rawdata.prep_labels(level = 1)

gibbstart = GibbsStart(rawdata)

sample_input = deepcopy(gibbstart)
cgs = GibbsSampling(sample_input, alpha_strength = 50)

cgs.run(nsamples = 100)