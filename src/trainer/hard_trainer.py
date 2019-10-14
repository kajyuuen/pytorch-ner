import os
import re
import copy
import random
from collections import Counter

import torch
import torch.optim as optim
from torchtext import data, datasets
from torchtext.vocab import GloVe

from tqdm import tqdm
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

from src.trainer import Trainer
from src.modules import BiLSTM_CRF

from src.common.utils import convert
from src.common.utils import lr_decay
from src.data.conll_dataset import Conll2003Dataset

from src.common.config import PAD_TAG, UNK_TAG, UNLABELED_TAG, START_TAG, STOP_TAG, UNLABELED_ID

import logging
logger = logging.getLogger(__name__)

class HardTrainer:
    def __init__(self,
                 num_tags,
                 label_vocab,
                 char_vocab,
                 word_vocab,
                 emb_dict,
                 config,
                 trainer_config,
                 train_dataset,
                 valid_dataset,
                 test_dataset = None,
                 label_dict = None,
                 dropout_rate = 0.5,
                 is_every_all_train = False):
        # Hard train config
        self.min_epochs = trainer_config["min_epochs"]
        self.split_num = 2
        self.config = config
        self.is_every_all_train = is_every_all_train

        # Model config
        self.num_tags = num_tags
        self.label_vocab = label_vocab
        self.char_vocab = char_vocab
        self.word_vocab = word_vocab
        self.emb_dict = emb_dict
        self.dropout_rate = dropout_rate
        self.device = trainer_config["device"]
        
        # Dataset
        self.trainer_config = trainer_config
        train_batch_size = self.trainer_config["train_batch_size"]
        eval_batch_size = len(valid_dataset) if self.trainer_config["eval_batch_size"] is None else self.trainer_config["eval_batch_size"]
        test_batch_size = len(test_dataset) if self.trainer_config["test_batch_size"] is None else self.trainer_config["test_batch_size"]
        self.first_train_datasets = list(train_dataset.split(1/self.split_num, random_state = random.getstate()))
        self.train_datasets = list(train_dataset.split(1/self.split_num, random_state = random.getstate()))
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

        # Other
        self.label_dict = label_dict
        self.base_save_path = self.trainer_config["path"]
        self.trainers = []
        for i in range(self.split_num):
            trainer = self._create_trainer(i)
            self.trainers.append(trainer)
            
    def train(self, iter_num):
        best_iteration, best_valid_f1, best_test_f1 = 0, 0, 0
        for ind in range(1, iter_num+1):
            # Train
            for train_idx in range(self.split_num):
                self.trainers[train_idx].train(self.min_epochs)
            # Labeling
            for train_idx in range(self.split_num):
                self.train_datasets[train_idx-1] = self._labeling(self.trainers[train_idx].best_model_path,
                                                                  self.first_train_datasets[train_idx-1],
                                                                  self.train_datasets[train_idx-1])
            # Update trainers
            for train_idx in range(self.split_num):
                self.trainers[train_idx] = self._create_trainer(train_idx)
            
            if self.is_every_all_train and ind < iter_num+1:
                self._all_train(self.min_epochs, ind)
        # All Train
        self._all_train(self.min_epochs)

    def _labeling(self, best_model_path, base_dataset, dataset, decode_type = "normal", batch_size = 100):
        assert decode_type in ["normal", "restricted"]

        # Create model
        model = self._create_model()
        model.load_state_dict(torch.load(best_model_path))
        if self.device != "cpu":
            model = model.to(self.device)
        model.eval()

        # Create iterator
        dataset_iteration = data.Iterator(dataset,
                                          batch_size = batch_size,
                                          shuffle = False,
                                          device = self.device)
        base_dataset_iteration = data.Iterator(dataset,
                                               batch_size = batch_size,
                                               shuffle = False,
                                               device = self.device)

        # Predict tags
        predict_tags = []
        for base_batch, batch in zip(base_dataset_iteration, dataset_iteration):
            if decode_type == "normal":
                predict_tags.extend(convert(model.decode(batch), self.label_dict))
            else:
                predict_tags.extend(convert(model.restricted_decode(base_batch, batch), self.label_dict))

        # Labeling
        for example, predict_label in zip(dataset, predict_tags): 
            assert len(example.label) == len(predict_label)
            example.label = predict_label
        
        return dataset

    def _create_trainer(self, i):
        self.trainer_config["path"] = self.base_save_path + "/{}".format(i)
        trainer = Trainer(self._create_model(),
                          self.trainer_config,
                          self.train_datasets[i],
                          self.valid_dataset,
                          self.test_dataset,
                          self.label_dict)
        return trainer
    
    def _create_model(self):
        model = BiLSTM_CRF(self.num_tags,
                        self.label_vocab,
                        self.char_vocab,
                        self.word_vocab,
                        self.emb_dict,
                        dropout_rate = self.dropout_rate,
                        inference_type = "Hard")
        if self.device != "cpu":
            model = model.to(self.device)
        return model

    def _all_train(self, num_epochs, model_label=None):
        # Create all train dataset
        concat_train_datasets = self.train_datasets[0]
        for train_idx in range(1, self.split_num):
            concat_train_datasets += self.train_datasets[train_idx]
        all_examples = [example for example in concat_train_datasets]
        # Create field
        word = data.Field(include_lengths = True, batch_first = True, lower = True,
                            preprocessing = data.Pipeline(lambda w: re.sub('\d', '0', w) if self.config.is_digit else w ))
        char_nesting = data.Field(tokenize = list, batch_first=True, lower = self.config.is_lower, init_token = START_TAG, eos_token = STOP_TAG,
                            preprocessing = data.Pipeline(lambda s: re.sub('\d', '0', s) if self.config.is_digit else s ))
        char = data.NestedField(char_nesting, include_lengths = True)
        label = data.Field(unk_token = UNLABELED_TAG, batch_first = True)
        fields = [(('word', 'char'), (word, char)), ('label', label)]
        # Load train, valid, test datasets
        all_train_dataset = Conll2003Dataset(examples = all_examples, fields = fields)
        _, valid_dataset, test_dataset = Conll2003Dataset.splits(fields = fields,
                                                                    path = self.config.dataset_path,
                                                                    separator = " ",
                                                                    train = "eng.train", 
                                                                    validation = "eng.testa", 
                                                                    test = "eng.testb")

        # Build vocab
        word.build_vocab(all_train_dataset, valid_dataset, test_dataset, vectors=GloVe(name='6B', dim='100'))
        char.build_vocab(all_train_dataset, valid_dataset, test_dataset)
        label.build_vocab(all_train_dataset, valid_dataset, test_dataset)
        # UNKNOWN tag is -1
        label.vocab.stoi = Counter({ k: v - 1 for k, v in label.vocab.stoi.items() })
        # Don't count UNKNOWN tag
        num_tags = len(label.vocab) - 1
        assert label.vocab.stoi[UNLABELED_TAG] == UNLABELED_ID
        # Create model
        model = BiLSTM_CRF(num_tags,
                        label.vocab,
                        char.vocab,
                        word.vocab,
                        self.config.emb_dict,
                        dropout_rate = self.config.dropout_rate,
                        inference_type = self.config.inference_type)
        if self.device != "cpu":
            model = model.to(self.device)
        self.trainer_config["path"] = self.base_save_path
        # Trainer
        if model_label is not None:
            self.trainer_config["path"] = self.base_save_path + "/all_train/{}".format(model_label)
        trainer = Trainer(model,
                          self.trainer_config,
                          all_train_dataset,
                          self.valid_dataset,
                          test_dataset = self.test_dataset,
                          label_dict = self.label_dict)
        trainer.train(num_epochs)
