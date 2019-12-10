from typing import Iterable, Dict, List, Any, Optional

import gensim
import csv


class PrepareCorpus:
    def __init__(self,
                 docs: Iterable[str],
                 labs: Iterable[str],
                 vocabulary: Optional[gensim.corpora.dictionary.Dictionary]=None,
                 vocab_no_below: float=0.01,
                 vocab_no_above: float=0.95):
        self.docs = gensim.parsing.preprocessing.preprocess_documents(docs)
        self.labs = labs

        self.vocab_no_below = vocab_no_below
        self.vocab_no_above = vocab_no_above

        self.vocabulary = vocabulary or self.get_clipped_vocabulary()
        self.doc_tuples = [self.vocabulary.doc2bow(doc) for doc in self.docs]
        pass

    def get_clipped_vocabulary(self) -> gensim.corpora.dictionary.Dictionary:
        """ Returns a clipped gensim dictionary of corpus' vocabulary """
        vocabulary = gensim.corpora.dictionary.Dictionary(self.docs)

        lower = self.vocab_no_below * len(self.docs)
        upper = self.vocab_no_above
        
        vocabulary.filter_extremes(no_above=upper, no_below=lower)
        return vocabulary

    #def create_w_to_v_mapping(self):
    #    return self.vocabulary.token2id

    #def create_v_to_w_mapping(self):
    #    pass self.vocabulary.id2token


def import_data(f_name: str) -> Dict[str, List[Any]]:
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
