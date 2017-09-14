from gensim.parsing import preprocessing
from gensim import corpora

# Static methods to prepare unseen documents
def keep_in_dict(new_doc, lda_dict):
    assert isinstance(lda_dict, corpora.dictionary.Dictionary)
    dict_words = list(lda_dict.token2id.keys())
    return [word for word in new_doc if word in dict_words]


def new_doc_prep(new_doc, lda_dict):
    _ = preprocessing.preprocess_string(new_doc)
    return keep_in_dict(_, lda_dict)


def new_docs_prep(new_docs, lda_dict):
    return [new_doc_prep(doc, lda_dict) for doc in new_docs]


def open_txt_doc(filename):
    """
    Convenient function that drops all non-utf8 characters
    and therefore allows for smooth transformation between latin1 encoded txt
    files and the utf8 format required by gensim text preparation.
    :param filename: Absolute path to .txt document
    :return: Whole document as bytes
    """
    f = open(filename, 'rb')
    doc = f.read()
    doc = doc.decode('utf-8', 'ignore').encode('utf-8')
    return doc


class PrepLabeledData:
    """
    This class transform extracts of databases to objects that can be used
    for topicmodeling by this package. The database extracts must contain 3
    variables/columns for each document:
    1) Unique identifier
    2) A long string representing the document
    3) The labels contained in one string

    The labels in the database extract are prepared and "shortened" to the
    desired depth of JEL-hierarchy (codes A, A1, A11), where 'A' correspond to
    level=1, 'A1' to level=2, and 'A11' to level=3. The level of granularity can
    be changed after initiating the class, too.

    This class contains a static method that prepares the same type labeled
    database extracts for posterior inference and testing, i.e. prepares for
    labeled documents to be used as a test dataset.
    """
    # TODO make this class more elaborate, also available for non-conform data

    def __init__(self, db_extract, level):
        assert level in [1, 2, 3], 'The level of the labels must be 1, 2 or 3'
        self.id = [x[0] for x in db_extract]
        self.docs = [x[1] for x in db_extract]
        self.lab = [x[2] for x in db_extract]

        self.prepped_labels = self.label_level(level=level)

    def label_level(self, level):
        # Prepare the labels according to 'level' hierarchy:
        labels = [label.split(" ") for label in self.lab]
        labels = [[label[0:level] for label in doclab] for doclab in labels]
        return [list(set(label)) for label in labels]

    def split_testdata(test_data):
        if any(isinstance(i, list) for i in test_data):
            new_docs = [x[1] for x in test_data]
            new_labs = [x[2] for x in test_data]
        else:
            new_docs = test_data[1]
            new_labs = test_data[2]
        return new_docs, new_labs
# End of PrepLabeledData class
