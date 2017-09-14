def recall(pred, real):
    # assert len(pred) == len(real), "The number of predictions does not match" \
    #                               "real predictions"
    """
    Evaluation metric for multi-label classification.
    :param pred: List with label prediction
    :param real: List with real labels
    :return: Cardinality of intersection divided by cardinality pred
    """
    _ = [x for x in real if x in pred]
    return len(_)/len(real)


def precision(pred, real):
    """
    Evaluation metric for multi-label classification.
    :param pred: List with label prediction
    :param real: List with real labels
    :return: Cardinality of intersection divided by cardinality real
    """
    _ = [x for x in pred if x in real]
    return len(_)/len(pred)


def f1_score(met1, met2):
    """
    Harmonic mean of evaluation metrics for multi-label classification
    :param met1: Metric1 is an evaluation metric (float)
    :param met2: Metric2 is another evaluation metric (float)
    :return: Harmonic mean of met1 and met2
    """
    return 2 / (1/recall + 1/precision)


def jaccard_index(pred, real):
    """
    Evaluation metric for multi-label classification. Number of
    correctly predicted labels divided by the union of predicted
    and true labels.
    :param pred: List with label predictions
    :param real: List with real labels
    :return: Jaccard Index (float) in [0,1]
    """
    l = [pred, real]
    intersect = [x for x in real if x in pred]
    union = set([item for sublist in l for item in sublist])
    return len(intersect) / len(union)

