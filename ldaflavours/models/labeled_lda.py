from ldaflavours.models.base_lda import BaseLDA
from ldaflavours.helpers.loader import PrepareCorpus

import numpy as np
from numpy.random import multinomial as multinom_draw

from typing import List, Dict, Union, Tuple

doctup_type = List[Tuple[int]]
state_type = Dict[str, Union[doctup_type, np.array]]


# Only testing-related methods in comparison with BaseLDA..
class PosteriorLDA(BaseLDA):
    """
    Methods to get predictions on a list of unseen texts (List[str]).

    These texts are turned into doc tuples (id, freq) according to the
    vocabulary. New words that are not in the existing vocabulary are ignored.
    """

    def _import_new_docs(self, docs: List[str]) -> List[doctup_type]:
        """ Generate List of id-frequency tuples from list of texts (str) """
        new_corpus = PrepareCorpus(
            docs=docs,
            labs=None,
            vocabulary=self.vocabulary
        )

        return new_corpus.doc_tuples

    def _initiate_doc(self,
                      doc_tuple: doctup_type
                      ) -> state_type:
        """ Take document tuple and assign initial z_dn values """
        doc, freqs = zip(*doc_tups)

        z_dn = []
        n_dk = np.zeros(self.K, dtype=int)

        probs = self.ph_hat[:, doc]
        with np.errstate(divide='raise', invalid='raise'):
            try:
                probs /= probs.sum(axis=0)
            except FloatingPointError:
                probs = 1 / self.K * np.ones_like(probs)

        for n, f in enerumate(freqs):
            prob = probs[:, n]
            prob /= np.sum(prob)

            new_z = multinom_draw(1, prob).argmax()
            
            z_dn.append(new_z)
            n_dk[new_z] +- f

        start_state = {
            'doc_tuple': doc_tuple,
            'z_dn': z_dn,
            'n_dk': n_dk
        }
        return start_state

    def run_test(self,
                 new_docs: List[str],
                 it: int,
                 thinning: int) -> np.array:
        """ Fit theta on unseen documents """
        nr = len(newdocs)
        th_hat = np.zeros((nr, self.K), dtype=float)

        new_doc_tups = self._import_new_docs(new_docs)

        for d, new_doc_tup in enumerate(new_doc_tups):
            start_state = self._initiate_doc(new_doc_tup)

            for i in range(it):
                new_state = self._test_iteration(start_state)

                time_for_saving = (n + 1) % thinning == 0
                if time_for_saving:
                    new_state = self._save_test_iteration(
                        new_state,
                        n=i,
                        thinning=thinning
                    )

                start_state = new_state
                
            th_hat[d, :] = new_state['avg_theta']

        return th_hat

    def _test_iteration(self, in_state: state_type) -> None:
        """ Take state and re-assign word-topic assignments """
        n_dk = in_state['n_dk']
        z_dn = in_state['z_dn']
        doc_tup = in_state['doc_tup']

        doc, freqs = zip(*doc_tup)

        for n, (v, f, z) in enumerate(zip(doc, freqs, z_dn)):
            n_dk[z] -= f

            num_a = n_dk + self.alpha
            b = self.ph_hat[:, v]

            prob = num_a * b
            prob /= prob.sum()

            new_z = multinom_draw(1, prob).argmax()

            z_dn[n] = new_z
            n_dk[new_z] += f

        out_state = {
            'n_dk': n_dk,
            'z_dn': z_dn,
            'doc_tup': doc_tup,
            'avg_theta': in_state['avg_theta']
        }
        return out_state

    def _save_test_iteration(self,
                             state: state_type,
                             n: int,
                             thinning: int) -> state_type:
        """ Return updated state with new average theta """ 
        save_count = (n + 1) / thinning

        n_dk = state['n_dk']
        current_theta = n_dk / n_dk.sum()

        if save_count == 1:
            avg_theta = current_theta

        elif save_count > 1:
            factor = (save_count - 1) / save_count
            fraction = 1 / save_count

            old = factor * state['avg_theta']
            new = fraction * current_theta

            new_avg_theta = new + old

        state['avg_theta'] = new_avg_theta
        return state

    def get_prediction(
        self, single_th: np.array, n: int = 5
    ) -> List[Tuple[Union[str, float]]]:
        """ Return the top n topics from a fitted theta vector """
        possible_labels = np.array(list(self.label_mapping.keys()))

        top_n_inds = np.argsort(-single_th)[:n]
        top_n_loadings = np.flip(np.sort(single_th), axis=0)[:n]

        top_n_topics = possible_labels[top_n_inds]
        return list(zip(top_n_topics, top_n_loadings))

    def get_predictions(
        self, all_th: np.array, n: int = 5
    ) -> List[List[Tuple[Union[str, float]]]]:
        """ Return the top n topic with loadings from fitted theta matrix """
        predictions = []
        nr = all_th.shape[0]
        for d in range(nr):
            one_th = all_th[d, :]
            prediction = self.get_prediction(one_th, n)

            predictions.append(prediction)
        return predictions
