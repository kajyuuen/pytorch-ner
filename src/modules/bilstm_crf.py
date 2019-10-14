import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_partial_crf import CRF
from pytorch_partial_crf import PartialCRF
from typing import Tuple

from src.common.config import PAD_TAG
from src.common.utils import create_possible_tag_masks
from src.modules.bilstm import BiLSTM

class BiLSTM_CRF(nn.Module):
    def __init__(self,
                 num_tags,
                 label_vocab,
                 char_vocab,
                 word_vocab,
                 emb_dict,
                 dropout_rate = 0,
                 batch_first = True,
                 inference_type = "CRF"):
        super().__init__()
        self.encoder = BiLSTM(num_tags,
                              label_vocab,
                              char_vocab,
                              word_vocab,
                              emb_dict,
                              batch_first = batch_first)
        if inference_type in ["CRF", "Simple", "Hard"]:
            self.inferencer = CRF(num_tags)
        elif inference_type == "PartialCRF":
            self.inferencer = PartialCRF(num_tags)
        else:
            raise ModuleNotFoundError
        self.num_tags = num_tags

    def forward(self, batch) -> torch.Tensor:
        emissions, tags, mask = self._get_variable_for_decode(batch)
        loss = self.inferencer(emissions, tags, mask)
        return loss

    def decode(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        emissions, tags, mask = self._get_variable_for_decode(batch)
        best_tags_list = self.inferencer.viterbi_decode(emissions, mask)
        return best_tags_list

    def restricted_decode(self, base_batch, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        possible_tags = create_possible_tag_masks(self.num_tags, base_batch.label)
        emissions, _, mask = self._get_variable_for_decode(batch)
        best_tags_list = self.inferencer.restricted_viterbi_decode(emissions, possible_tags, mask)
        return best_tags_list

    def _get_variable_for_decode(self, batch) -> torch.Tensor:
        emissions = self.encoder(batch)
        tags = batch.label
        mask = tags.clone().byte()
        mask[mask != 0] = 1
        return emissions, tags, mask
