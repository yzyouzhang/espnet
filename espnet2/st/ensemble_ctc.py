"""ScorerInterface implementation for CTC."""

from typing import List

import numpy as np
import torch

from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.scorer_interface import BatchPartialScorerInterface


class EnsembleCTC(BatchPartialScorerInterface):
    """Ensemble for CTC module"""

    def __init__(
        self,
        ctc: List[torch.nn.Module],
        eos: int,
        weights: List[float] = None,
    ):
        """Initialize class.

        Args:
            ctc (List[torch.nn.Module]): The CTC implementation.
                For example, :class:`espnet.nets.pytorch_backend.ctc.CTC`
            eos (int): The end-of-sequence id.
            weights (List[flost]): weights for CTC ensembling

        """
        ctc_list = []
        self.valid = 0

        for module in ctc:
            if module is not None:
                self.valid += 1
                ctc_list.append(CTCPrefixScorer(ctc=module, eos=eos))
            else:
                ctc_list.append(None)
        self.ctc = torch.nn.ModuleList(ctc_list)
        self.eos = eos
        self.ctc_num = len(ctc)
        self.weights = [1 / self.valid] * self.valid if weights is None else weights

    def init_state(self, x: List[torch.Tensor]):
        """Get an initial state for decoding.

        Args:
            x (List[torch.Tensor]): The encoded feature tensor

        Returns: initial state

        """
        initial_states = []
        for i in range(self.ctc_num):
            if self.ctc[i] is None:
                initial_states.append(None)
            else:
                initial_state = self.ctc[i].initial_state(x[i])
                initial_states.append(initial_state)
        return initial_states

    def select_state(self, state, i, new_id=None):
        """Select state with relative ids in the main beam search.

        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search
            new_id (int): New label id to select a state if necessary

        Returns:
            state: pruned state

        """
        selected_states = []
        for i in range(self.ctc_num):
            if self.ctc[i] is None:
                selected_states.append(None)
            else:
                selected_state = self.ctc.select_state(state[i], i, new_id)
                selected_states.append(selected_state)
        return selected_states

    def score_partial(self, y, ids, state, x):
        """Score new token.

        Args:
            y (torch.Tensor): 1D prefix token
            next_tokens (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (List[torch.Tensor]): 2D encoder feature that generates ys

        Returns:
            tuple[torch.Tensor, List[Any]]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        tscores = []
        sub_states = []
        for i in range(self.ctc_num):
            if self.ctc[i] is None:
                sub_states.append(None)
            else:
                sub_tscore, sub_state = self.ctc[i].score_partial(
                    y, ids, state[i], x[i]
                )
                tscores.append(np.log(self.weights[i]) + sub_tscore)
                sub_states.append(sub_state)

        return torch.logsumexp(torch.stack(tscores, dim=0), dim=0), sub_states

    def batch_init_state(self, x: List[torch.Tensor]):
        """Get an initial state for decoding.

        Args:
            x (torch.Tensor): The encoded feature tensor

        Returns: initial state

        """
        for i in range(self.ctc_num):
            if self.ctc[i] is None:
                continue
            else:
                self.ctc[i].batch_init_state(x[i])
        return None

    def batch_score_partial(self, y, ids, state, x):
        """Score new token.

        Args:
            y (torch.Tensor): 1D prefix token
            ids (torch.Tensor): torch.int64 next token to score
            state: decoder state for prefix tokens
            x (torch.Tensor): 2D encoder feature that generates ys

        Returns:
            tuple[torch.Tensor, Any]:
                Tuple of a score tensor for y that has a shape `(len(next_tokens),)`
                and next state for ys

        """
        n_batch = len(state)
        new_states = []
        scores = []
        for i in range(self.ctc_num):
            if self.ctc[i] is None:
                new_states.append(None)
            else:
                state_ctc = [state[j][i] for j in range(len(state))]
                sub_score, new_state_ctc = self.ctc[i].batch_score_partial(
                    y, ids, state_ctc, x[i]
                )
                scores.append(np.log(self.weights[i]) + sub_score)
                new_states.append(new_state_ctc)
        score = torch.logsumexp(torch.stack(scores, dim=0), dim=0)

        # transpose state
        transpose_state_list = []
        for i in range(n_batch):
            state_ctc = []
            for j in range(self.ctc_num):
                if new_states[j] is not None:
                    state_ctc.append(new_states[j][i])
                else:
                    state_ctc.append(None)
            transpose_state_list.append(state_ctc)
        return score, transpose_state_list

    def extend_prob(self, x: torch.Tensor):
        """Extend probs for decoding.

        This extension is for streaming decoding
        as in Eq (14) in https://arxiv.org/abs/2006.14941

        Args:
            x (torch.Tensor): The encoded feature tensor

        """
        for i in range(self.ctc_num):
            if self.ctc[i] is None:
                continue
            else:
                self.ctc[i].extend_prob(x[i])
