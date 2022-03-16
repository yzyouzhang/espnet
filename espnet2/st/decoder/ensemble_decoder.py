# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""
from typing import Any
from typing import List
from typing import Tuple
from typing import Union
from typing import Optional

import numpy as np
import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.scorer_interface import BatchScorerInterface
from espnet2.asr.decoder.abs_decoder import AbsDecoder


class EnsembleDecoder(AbsDecoder, BatchScorerInterface):
    """Base class of Transfomer decoder module.

    Args:
        decoders: ensemble decoders
    """

    def __init__(
        self,
        decoders: List[Union[AbsDecoder, None]],
        weights: List[float] = None,
        return_hidden: bool = False
    ):
        assert check_argument_types()
        super().__init__()
        assert len(decoders) > 0, "At least one decoder is needed for ensembling"

        self.return_hidden = return_hidden
        # Note (jiatong): an invalid index to skip some decoder
        # (useful when some decoders are none)
        self.valid = 0
        for decoder in decoders:
            if decoder is not None:
                self.valid += 1

        # Note (jiatong): different from other'decoders
        self.decoders = torch.nn.ModuleList(decoders)
        self.weights = [1 / self.valid] * self.valid if weights is None else weights

    def init_state(self, x: torch.Tensor) -> Any:
        """Get an initial state for decoding (optional).

        Args:
            x (torch.Tensor): The encoded feature tensor
        Returns: initial state
        """
        return [None for i in range(len(self.decoders))]

    def batch_init_state(self, x: torch.Tensor) -> Any:
        """Get an initial state for decoding (optional).

        Args:
            x (torch.Tensor): The encoded feature tensor
        Returns: initial state
        """
        return self.init_state(x)

    def forward(
        self,
        hs_pad: List[torch.Tensor],
        hlens: List[torch.Tensor],
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dummy forward"""
        # FIXME (jiatong): currently only select one decoder to perform forward
        for i in range(len(self.decoders)):
            if self.decoders[i] is not None:
                return self.decoders[i](
                    hs_pad[i], hlens[i], ys_in_pad, ys_in_lens, return_hidden=True
                )

    def score(self, ys, state, x, speech=None):
        """Score."""
        assert len(x) == len(
            self.decoders
        ), "Num of encoder output does not match number of decoders"
        logps = []
        states = []
        for i in range(len(self.decoders)):
            if self.decoders[i] is None:
                continue
            if speech is not None and speech[i] is not None:
                logp, sub_state = self.decoders[i].score(ys, state[i], x[i], speech[i])
            else:
                logp, sub_state = self.decoders[i].score(ys, state[i], x[i])
            logps.append(np.log(self.weights[i]) + logp.squeeze(0))
            states.append(sub_state)
        return torch.logsumexp(torch.stack(logps, dim=0), dim=0), states

    def batch_score(
        self,
        ys: torch.Tensor,
        states: List[Any],
        xs: Union[torch.Tensor, List[torch.Tensor]],
        speech: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        return_hidden: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (Union[torch.Tensor, List[torch.Tensor]]):
                The encoder feature that generates ys (n_batch, xlen, n_feat).
        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.
        """
        n_batch = len(states)
        n_decoders = len(self.decoders)

        hs_list = []
        all_state_list = []
        logps = []
        import logging
        logging.info("decoder_num: {}, decoder_batch: {}, self.valid: {}, x: {}".format(n_decoders, n_batch, self.valid, len(xs)))
        decoder_index = 0
        for i in range(n_decoders):
            if self.decoders[i] is None:
                all_state_list.append(None)
                hs_list.append(None)
                continue
            decoder_batch = [states[h][i] for h in range(n_batch)]
            if speech is not None and speech[i] is not None:
                # in these case, there will be no return hidden considered
                logp, state_list = self.decoders[i].batch_score(
                    ys, decoder_batch, xs[i], speech[i]
                )
                hs = None
            else:
                if self.return_hidden:
                    logp, hs, state_list = self.decoders[i].batch_score(
                        ys, decoder_batch, xs[i], return_hidden=self.return_hidden
                    )
                else:
                    logp, state_list = self.decoders[i].batch_score(
                        ys, decoder_batch, xs[i]
                    )
                    hs = None
            hs_list.append(hs)
            all_state_list.append(state_list)
            logps.append(np.log(self.weights[decoder_index]) + logp)
        score = torch.logsumexp(torch.stack(logps, dim=0), dim=0)

        transpose_state_list = []
        transpose_hs_list = []
        for i in range(n_batch):
            decoder_state = []
            sub_hs = []
            for j in range(n_decoders):
                if all_state_list[j] is not None:
                    decoder_state.append(all_state_list[j][i]) 
                else:
                    decoder_state.append(None)
                if hs_list[j] is not None:
                    sub_hs.append(hs_list[j][i])
                else:
                    sub_hs.append(None)
            transpose_state_list.append(decoder_state)
            transpose_hs_list.append(transpose_hs_list)
        if return_hidden:
            return score, transpose_hs_list, transpose_state_list
        return score, transpose_state_list
