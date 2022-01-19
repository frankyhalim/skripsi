import logging
import torch
from torch import nn
import math

logger = logging.getLogger(__name__)

# Sinusoidal position encoding
class PositionalEncoding(nn.Module):

    def __init__(self, d_model=768, dropout=0.1, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# Transformer encoder with sinusoidal position encoding
class TransformerEncoderClassifier(nn.Module):
    def __init__(
        self,
        d_model=768,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        num_layers=2
    ):
        super(TransformerEncoderClassifier, self).__init__()
        self.nhead = nhead
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        layer_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=layer_norm)
        wo = nn.Linear(d_model, 1, bias=True)
        self.reduction = wo

    def forward(self, x, mask):
        
        # apply sinusoidal position encoding to input sequence of token
        x = self.pos_encoder(x)
        
        # add dimension in the middle
        attn_mask = mask.unsqueeze(1)
        
        # expand the middle dimension to the same size as the last dimension (the number of sentences/source length)
        attn_mask = attn_mask.expand(-1, attn_mask.size(2), -1)
        
        # repeat the mask for each attention head
        attn_mask = attn_mask.repeat(self.nhead, 1, 1)
        
        # attn_mask is shape (batch size*num_heads, target sequence length, source sequence length)
        # set all the 0's (False) to negative infinity and the 1's (True) to 0.0 because the attn_mask is additive
        attn_mask = (
            attn_mask.float()
            .masked_fill(attn_mask == 0, float("-inf")) 
            .masked_fill(attn_mask == 1, float(0.0)) 
        )
        
        x = x.transpose(0, 1)
        # x is shape (source sequence length, batch size, feature number)

        x = self.encoder(x, mask=attn_mask)
        # x is still shape (source sequence length, batch size, feature number)
        
        x = x.transpose(0, 1).squeeze()
        # x is shape (batch size, source sequence length, feature number)
        
        x = self.reduction(x)
        
        # x is shape (batch size, source sequence length, 1)
        # mask is shape (batch size, source sequence length)
        sent_scores = x.squeeze(-1) * mask.float()
        
        # to preserve loss calculation  
        sent_scores[sent_scores == 0] = -9e3
        
        return sent_scores
