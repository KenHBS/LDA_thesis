#TODO: Add label-value based label_tree creator

from flavourlda.models import BaseLDA, PosteriorLDA
from flavourlda.helpers import PrepareCorpus

from typing import List, Tuple, Dict, Set


doctup_type = List[Tuple[int]]


class CascadeLDA(BaseLDA):
    
    def __init__(self,
                 label_tree: Dict[str, Set[str]],
                 hierarchy: Dict[int, Set[str]],
                 **kwargs):
        __super__.__init__(
            **kwargs
        )
        self.label_tree = label_tree
        self.hierarchy = hierarchy

    def filter_corpus_for_label(self, label: str):
        pass

    def filter_doc_for_word_assignment(self, doc: doctup_type, label: str):
        pass

    def run_lower_lda(self,
                      doc_tuples: List[doctup_type],
                      labs=List[,
                      iter: int) -> Dict[str, Any]:
        sub_model = BaseLDA(
            doc_tuples=doc_tuples,
            labs=labels,
            vocabulary=self.vocabulary,
            alpha=self.alpha,
            beta=self.beta
        )
        sub_model.run_training(

