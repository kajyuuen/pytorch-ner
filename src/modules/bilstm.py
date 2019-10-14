import torch
import torch.nn as nn
import torch.nn.functional as F

from src.modules.char_bilstm import CharBiLSTM
from src.common.config import PAD_TAG, UNK_TAG
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class BiLSTM(nn.Module):
    def __init__(self,
                 num_tags,
                 label_vocab,
                 char_vocab,
                 word_vocab,
                 emb_dict,
                 dropout_rate = 0.5,
                 batch_first = True):
        super().__init__()
        self.num_tags = num_tags

        # Char embedding encoder
        self.char_lstm = CharBiLSTM(len(char_vocab),
                                    emb_dict["char_emb_dim"],
                                    emb_dict["char_hidden_dim"],
                                    dropout = dropout_rate)

        # Word embedding
        self.word_embedding = nn.Embedding.from_pretrained(word_vocab.vectors, freeze=False)
        self.word_drop = nn.Dropout(dropout_rate)

        # Word + Char embedding encoder
        self.lstm = nn.LSTM(emb_dict["word_emb_dim"] + emb_dict["char_hidden_dim"],
                            emb_dict["hidden_dim"] // 2,
                            num_layers = 1,
                            batch_first = batch_first,
                            bidirectional = True)

        self.drop_layer = nn.Dropout(p = dropout_rate)
        self.hidden2tag = nn.Linear(emb_dict["hidden_dim"], num_tags)

    def forward(self, batch) -> torch.Tensor:
        word_seq_tensor, word_seq_len = batch.word
        char_seq_tensor, _, char_seq_len = batch.char
        batch_size, sequence_length = word_seq_tensor.shape

        # Get embedding
        inputs_word_emb = self.word_embedding(word_seq_tensor)           # (batch, sequence_length, word_emb_dim)
        inputs_char_emb = self.char_lstm(char_seq_tensor, char_seq_len)  # (batch, sequence_length, word_emb_dim)

        # Combine word and chat emb
        inputs_word_char_emb = torch.cat([inputs_word_emb, inputs_char_emb], 2)
        inputs_word_char_emb = self.word_drop(inputs_word_char_emb)

        # Convert embedding to feature
        sorted_seq_len, perm_idx = word_seq_len.sort(0, descending=True)
        _, recover_idx = perm_idx.sort(0, descending=False)
        sorted_seq_tensor = inputs_word_char_emb[perm_idx]

        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len, True)
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        feature_out = self.drop_layer(lstm_out)
        outputs = self.hidden2tag(feature_out)

        return outputs[recover_idx]
