import csv
import sys
import re

import gensim.parsing import preprocessing
from gensim.corpora import dictionary
import numpy as np
from numpy.random import multinomial as multinom_draw


def load_corpus(filename, d):
    # Increase max line length for csv.reader:
    max_int = sys.maxsize
    decrement = True
    while decrement:
        decrement = False
        try:
            csv.field_size_limit(max_int)
        except OverflowError:
            max_int = int(max_int/10)
            decrement = True

    docs = []
    labs = []
    labelmap = dict()

    pat = re.compile("[A-Z]\d{2}")
    with open(filename, 'r') as fp:
        reader = csv.reader(fp)
        for row in reader:
            doc = row[1]
            lab = row[2]

            labels = (x[:d] for x in lab.split(" ") if pat.search(x))
            for label in labels:
                labelmap[label] = 1

            all_labels.append(list(set(labels)))
            docs.append(doc)

    print("Stemming documents ....")
    docs = preprocess_documents(docs)
    return docs, labs, list(labelmap.keys())
