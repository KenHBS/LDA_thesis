# TODO: Add label-value based label_tree creator
# TODO: Add a root-label argument to BaseLDA or CascadeLDA, to filter out words not associated with root label during training (avoid gravitational pull)


from ldaflavours.models import BaseLDA, PosteriorLDA
from ldaflavours.helpers import PrepareCorpus

from typing import List, Tuple, Dict, Set, Any, Optional
import gensim


doctup_type = List[Tuple[int]]


class CascadeLDA(PosteriorLDA)):

    def __init__(self,
                 label_tree: Dict[str, Set[str]],
                 levels: Dict[int, Set[str]],
                 doc_tuples: List[doctup_type],
                 labs: List[List[str]],
                 vocabulary: gensim.corpora.dictionary.Dictionary,
                 alpha: float,
                 beta: float):
        self.label_tree = label_tree

        self.labels = self.parse_to_cascading_labels(labs)

        super().__init__(
            doc_tuples=doc_tuples,
            vocabulary=vocabulary,
            alpha=alpha,
            beta=beta,
            labs=self.labels
        )

        self.label_tree = label_tree
        self.levels = levels
        self.level_mapper = {lab: k for k, v in levels.items() for lab in v}

#        self.label_mapping = self.generate_label_mapping(self.cascade_labels)
        pass

    def filter_doc_for_word_assignment(self, doc: doctup_type, label: str):
        """ Main improvement over previous implementation of CascadeLDA """
        pass

    def run_training(self, iters: int, thinning: int) -> None:
        """ Orchestrate and train local LDA and combine into Cascade"""
        non_leafs = [k for k, v in self.label_tree.items() if v]
        for i, node in enumerate(non_leafs):

            print(f'{i}/{len(non_leafs)}: Training for root {node}')
            node_scope = self.filter_corpus_for_root(root=node)

            local_state = self.run_lower_lda(
                doc_tuples=node_scope['doctups'],
                labs=node_scope['labs'],
                iters=iters,
                thinning=thinning
            )

            self._process_lower_lda(submodel_state=local_state)
        pass

    def filter_corpus_for_root(self, root: str) -> Dict[str, List[Any]]:
        """ Return labeled corpus starting with this root label """
        doctups = []
        labs = []
        for lab, doctup in zip(self.labels, self.doc_tups):
            if root in lab:
                descendants = self.label_tree.get(root)

                if descendants is None:
                    next

                intersection = descendants.intersection(set(lab))
                if len(intersection) > 0:
                    labs.append(intersection)
                    doctups.append(doctup)
        return {
            'labs': labs,
            'doctups': doctups
        }

    def run_lower_lda(self,
                      doc_tuples: List[doctup_type],
                      labs: List[List[str]],
                      iters: int,
                      thinning: int) -> Dict[str, Any]:
        """ Run a local-node Labeled LDA and return trained state """
        sub_model = BaseLDA(
            doc_tuples=doc_tuples,
            labs=labs,
            vocabulary=self.vocabulary,
            alpha=self.alpha,
            beta=self.beta
        )

        sub_model.run_training(iters=iters, thinning=thinning)

        return {
            'label_mapping': sub_model.label_mapping,
            'ph_hat': sub_model.ph_hat,
            'th_hat': sub_model.th_hat
        }

    def _process_lower_lda(self, submodel_state: Dict[str, Any]):
        """ Insert local LDA state into global Cascade phi """
        local_label_mapping = submodel_state['label_mapping']
        local_th_hat = submodel_state['th_hat']
        local_ph_hat = submodel_state['ph_hat']

        global_label_mapping = self.label_mapping

        del local_label_mapping['root']

        for label, local_label_id in local_label_mapping.items():
            global_label_id = global_label_mapping[label]

            self.ph_hat[global_label_id, :] = local_ph_hat[local_label_id, :]
        pass

    def _single_doc_posterior_cascade(self,
                                      docs: List[str], # List of strings
                                      iters: int,
                                      thinning: int):
        """ Propagate unseen document through label tree """
        # TODO: We can use a self._test_iteration, self._save_test_iteration
        # TODO: We can use self._import_new_docs(new_docs)
        # TODO: Which branches to pursue after each level?
        # TODO: Should not use default self.run_test()

        # Only some branches
        tree_depth = max(x for x in cascade_lda.levels.keys())

        for level in range(1, tree_depth):
            
            descendants = list(self.levels[level])

            th_hat = self.run_test(
                new_docs=docs,
                it=iters, 
                thinning=thinning,
                label_subset=descendants
            )

    def run_posterior_cascade(self, docs: List[doctup_type]):
        pass

    def choose_next_root(th_hat):
        pass

    def parse_to_cascading_labels(self, labels: List[List[str]]):
        """ Parse a document's labels into cascading labels """
        parents = {lab: k for k, v in self.label_tree.items() for lab in v}

        def parse_label(labels: List[str], parent_dict: Dict[str, str]):
            """ Return cascade parsed labels of """
            cascade_labels = []
            for lab in labels:
                cascade_labels.append(lab)
                while True:
                    parent = parent_dict.get(lab)
                    if parent is None:
                        break

                    cascade_labels.append(parent)
                    lab = parent

            return list(set(cascade_labels))
        return [parse_label(lab, parents) for lab in labels]
