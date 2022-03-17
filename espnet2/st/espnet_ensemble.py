import argparse
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet2.st.espnet_model_md import ESPnetSTMDModel
from espnet2.st.decoder.ensemble_decoder import EnsembleDecoder
from espnet2.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetSTEnsemble(AbsESPnetModel):
    """Ensemble ST ESPnet model (a wrapper)"""

    def __init__(
        self,
        models: List[AbsESPnetModel],
        configs: List[argparse.Namespace],
    ):
        assert check_argument_types()
        assert len(models) > 0, "At least one model should presents for ensemble"
        super().__init__()
        self.model_num = len(models)
        self.single_model = models[0]
        self.sos = self.single_model.sos
        self.eos = self.single_model.eos
        self.ignore_id = self.single_model.ignore_id
        self.src_sos = -1
        self.src_eos = -1
        self.models = torch.nn.ModuleList(models)
        self.configs = configs
        decoders = []

        # add decoder
        for model in self.models:
            decoders.append(model.decoder)
            if isinstance(model, ESPnetSTMDModel):
                self.src_sos = model.src_sos
                self.src_eos = model.src_eos

        # add asr decoder
        # FIXME(jiatong): just use one asr encoder
        asr_decoder_index = -1
        asr_decoder_all = []
        for i in range(len(self.models)):
            model = self.models[i]
            if hasattr(model, "asr_decoder"):
                asr_decoder = model.asr_decoder
                asr_decoder_index = i
                asr_decoder_all.append(model.asr_decoder)
            else:
                asr_decoder_all.append(None)

        # add ctc decoder
        # FIXME(jiatong): just use one CTC
        ctc = self.models[asr_decoder_index].ctc

        self.decoder = EnsembleDecoder(decoders, return_hidden=False)
        self.asr_decoder = asr_decoder
        self.asr_decoder_all = asr_decoder_all
        self.asr_decoder_index = asr_decoder_index
        self.ctc = CTCPrefixScorer(ctc=ctc, eos=self.src_eos)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Dummy forward"""
        pass

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Dummy collect feats"""
        pass

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Frontend + Encoder. Note that this method is used by st_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        encoder_out = []
        encoder_out_lens = []

        for model in self.models:
            single_encoder_out, single_encoder_out_lens = model.encode(
                speech, speech_lengths
            )
            encoder_out.append(single_encoder_out)
            encoder_out_lens.append(single_encoder_out_lens)

        return encoder_out, encoder_out_lens

    def encoder_mt(
        self, asr_hs: List[torch.Tensor], asr_hs_lengths: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # TODO (jiatong) add function description
        # assume only one asr
        encoder_out = []
        encoder_out_lens = []

        for i in range(self.model_num):
            model = self.models[i]
            if hasattr(model, "encoder_mt") and asr_hs[i] is not None:
                single_encoder_out, single_encoder_out_lens, _ = model.encoder_mt(
                    asr_hs[i], asr_hs_lengths[i]
                )
            else:
                single_encoder_out, single_encoder_out_lens = None, None
            encoder_out.append(single_encoder_out)
            encoder_out_lens.append(single_encoder_out_lens)

        return encoder_out, encoder_out_lens
