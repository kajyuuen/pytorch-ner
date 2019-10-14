import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CharBiLSTM(nn.Module):
    def __init__(self,
                 char_size,
                 embedding_size,
                 hidden_dim,
                 dropout = 0.5):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.char_embeddings = nn.Embedding(char_size, embedding_size)
        self.hidden_dim = hidden_dim
        self.char_lstm = nn.LSTM(embedding_size,
                                 hidden_dim // 2,
                                 num_layers = 1,
                                 batch_first = True,
                                 bidirectional = True)

    def forward(self, char_input, seq_lengths):
        return self.get_last_hiddens(char_input, seq_lengths)
                            
    def get_last_hiddens(self, char_seq_tensor, char_seq_len):
        char_seq_len[char_seq_len == 0] = 1
        batch_size, sequence_length, _ = char_seq_tensor.shape
        char_seq_tensor = char_seq_tensor.view(batch_size * sequence_length, -1)
        char_seq_len = char_seq_len.view(batch_size * sequence_length)

        sorted_seq_len, perm_idx = char_seq_len.sort(0, descending=True)
        _, recover_idx = perm_idx.sort(0, descending=False)
        sorted_seq_tensor = char_seq_tensor[perm_idx]

        char_embeds = self.dropout(self.char_embeddings(sorted_seq_tensor))
        pack_input = pack_padded_sequence(char_embeds, sorted_seq_len, batch_first=True)

        _, char_hidden = self.char_lstm(pack_input, None)
        hidden = char_hidden[0].transpose(1,0).contiguous().view(batch_size * sequence_length, 1, -1)
        return hidden[recover_idx].view(batch_size, sequence_length, -1)
