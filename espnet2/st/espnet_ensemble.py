import argparse
from contextlib import contextmanager
from distutils.version import LooseVersion
from typing import Dict
from typing import List
from typing import Tuple

import torch
from typeguard import check_argument_types

from espnet2.st.espnet_model_md import ESPnetSTMDModel
from espnet2.st.decoder.ensemble_decoder import EnsembleDecoder
from espnet2.st.ensemble_ctc import EnsembleCTC
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
        asr_decoders = []
        for model in self.models:
            if hasattr(model, "asr_decoder"):
                asr_decoders.append(model.asr_decoder)
            else:
                asr_decoders.append(None)

        # add ctc decoder
        ctcs = []
        src_eos = -1
        for model in self.models:
            if hasattr(model, "ctc"):
                ctcs.apppend(model.ctc)
                src_eos = model.src_eos
            else:
                ctcs.append(None)

        self.decoder = EnsembleDecoder(decoders)
        self.asr_decoder = EnsembleDecoder(asr_decoders)
        self.ctc = EnsembleCTC(ctcs, src_eos)

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
        self, asr_hs: torch.Tensor, asr_hs_lengths: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        # TODO (jiatong) add function description
        encoder_out = []
        encoder_out_lens = []

        for model in self.models:
            single_encoder_out, single_encoder_out_lens = model.mt_encoder(
                asr_hs, asr_hs_lengths
            )
            encoder_out.append(single_encoder_out)
            encoder_out_lens.append(single_encoder_out_lens)

        return encoder_out, encoder_out_lens
