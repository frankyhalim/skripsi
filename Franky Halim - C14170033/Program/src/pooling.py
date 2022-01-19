import logging
import torch
from torch import nn

logger = logging.getLogger(__name__)


class Pooling(nn.Module):
    def __init__(self):
        super(Pooling, self).__init__()

    def forward(
        self,
        word_vectors=None,
        sent_rep_token_ids=None,
        sent_rep_mask=None
    ):
        output_vectors = [] 
        output_masks = []
        sents_vec = word_vectors[
            torch.arange(word_vectors.size(0)).unsqueeze(1), sent_rep_token_ids
        ]
        sents_vec = sents_vec * sent_rep_mask[:, :, None].float()
        output_vectors.append(sents_vec)
        output_masks.append(sent_rep_mask)

        output_vector = torch.cat(output_vectors, 1)
        output_mask = torch.cat(output_masks, 1)

        return output_vector, output_mask
