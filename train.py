import os
import re
import sys
import json
import random
from collections import Counter

import torch

from torchtext import data, datasets
from torchtext.vocab import GloVe

from src.trainer import Trainer
from src.trainer import HardTrainer
from src.modules import BiLSTM_CRF
from src.data.conll_dataset import Conll2003Dataset
from src.common.config import Config
from src.common.config import PAD_TAG, UNK_TAG, UNLABELED_TAG, START_TAG, STOP_TAG, UNLABELED_ID

def main(file_name):
    # Load json
    with open(file_name, "r") as f:
        setting_json = json.load(f)
    config = Config(setting_json)
    config_log = config.get_log()
    print(config_log)

    os.makedirs(config.trainer_config["path"], exist_ok = True)
    with open(config.trainer_config["path"] + "/config.json", "w") as f:
        json.dump(setting_json, f)
    with open(config.trainer_config["path"] + "/config.log", "w") as f:
        f.write(config_log)

    # Setting seed
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.device != "cpu":
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True

    # Define field
    WORD = data.Field(include_lengths = True, batch_first = True, lower = True,
                        preprocessing = data.Pipeline(lambda w: re.sub('\d', '0', w) if config.is_digit else w ))
    CHAR_NESTING = data.Field(tokenize = list, batch_first=True, lower = config.is_lower, init_token = START_TAG, eos_token = STOP_TAG,
                        preprocessing = data.Pipeline(lambda s: re.sub('\d', '0', s) if config.is_digit else s ))
    CHAR = data.NestedField(CHAR_NESTING, include_lengths = True)
    LABEL = data.Field(unk_token = UNLABELED_TAG, batch_first = True)
    fields = [(('word', 'char'), (WORD, CHAR)), ('label', LABEL)]

    # Load datasets
    train_dataset, valid_dataset, test_dataset = Conll2003Dataset.splits(fields = fields,
                                                                        path = config.dataset_path,
                                                                        separator = " ",
                                                                        train = "eng.train", 
                                                                        validation = "eng.testa", 
                                                                        test = "eng.testb")

    # Build vocab
    WORD.build_vocab(train_dataset, valid_dataset, test_dataset, vectors=GloVe(name='6B', dim='100'))
    CHAR.build_vocab(train_dataset, valid_dataset, test_dataset)
    LABEL.build_vocab(train_dataset, valid_dataset, test_dataset)

    # UNKNOWN tag is -1
    LABEL.vocab.stoi = Counter({ k: v - 1 for k, v in LABEL.vocab.stoi.items() })
    # Don't count UNKNOWN tag
    num_tags = len(LABEL.vocab) - 1
    assert LABEL.vocab.stoi[UNLABELED_TAG] == UNLABELED_ID

    if config.inference_type in ["Simple", "Hard"]:
        LABEL.vocab.stoi[UNLABELED_TAG] = LABEL.vocab.stoi["O"]

    # Create trainer
    if config.inference_type == "Hard":
        trainer = HardTrainer(num_tags,
                        LABEL.vocab,
                        CHAR.vocab,
                        WORD.vocab,
                        config.emb_dict,
                        config,
                        config.trainer_config,
                        train_dataset,
                        valid_dataset,
                        test_dataset = test_dataset,
                        label_dict = LABEL.vocab.stoi,
                        is_every_all_train = config.is_every_all_train)
    else:
        model = BiLSTM_CRF(num_tags,
                        LABEL.vocab,
                        CHAR.vocab,
                        WORD.vocab,
                        config.emb_dict,
                        dropout_rate = config.dropout_rate,
                        inference_type = config.inference_type)
        if config.device != "cpu":
            model = model.to(config.device)

        trainer = Trainer(model,
                        config.trainer_config,
                        train_dataset,
                        valid_dataset,
                        test_dataset = test_dataset,
                        label_dict = LABEL.vocab.stoi)
    trainer.train(config.trainer_config["epochs"])

if __name__ == "__main__":
    file_name = sys.argv[1]
    main(file_name)