import csv
from typing import Dict, Set


def import_data(f_name: str) -> Dict:
    """ Import dataset to dictionary """
    with open(f_name, 'r') as fp:
        reader = csv.reader(fp)
        d = dict()

        all_dois = []
        all_texts = []
        all_labels = []
        for row in reader:
            doi, text, labels = row

            labels = labels.split(' ')
            all_labels.append(labels)
            all_dois.append(doi)
            all_texts.append(text)

        d['doi'] = all_dois
        d['text'] = all_texts
        d['labels'] = all_labels
        return d


def get_pruned_g_dict(docs: List[str], lower=0.1, upper=0.9):
    """ Return gensim dict with filtered rare and frequent words """
    g_dict = gensim.corpora.dictionary.Dictionary(docs)
    lower *= len(docs)

    return g_dict.filter_extremes(no_above=upper, no_below=lower)
