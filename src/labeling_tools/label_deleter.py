import re
from collections import defaultdict

import numpy as np

from torchtext import data, datasets

from src.data.conll_dataset import Conll2003Dataset
from src.labeling_tools.data import Entity
from src.common.config import UNLABELED_TAG

class LabelDeleter:
    def __init__(self, file_path):
        np.random.seed(42)
        WORD = data.Field(include_lengths = True, batch_first = True, lower = False)
        LABEL = data.Field(unk_token = UNLABELED_TAG, batch_first = True)
        fields = [('word', WORD), ('label', LABEL)]

        self.dataset = Conll2003Dataset(fields = fields, path = file_path, separator = " ")
        self.e_type_counter = defaultdict(int)
        self.create_entity()

    def create_entity(self):
        entities = []
        for example in self.dataset:
            sentence_entities = []
            words = []
            for label, word in zip(example.label, example.word):
                words.append(word)
                if label == "O":
                    entity = Entity(label, words, self.e_type_counter[label])
                    sentence_entities.append(entity)
                    self.e_type_counter[label] += 1
                    words = []
                else:
                    position, e_type = label.split("-")
                    if position == "S" or position == "E":
                        entity = Entity(e_type, words, self.e_type_counter[e_type])
                        sentence_entities.append(entity)
                        self.e_type_counter[e_type] += 1
                        words = []
            entities.append(sentence_entities)
        self.entities = entities

    def delete_label(self, p = 0, other_delete_all = False):
        entity_count = sum(self.e_type_counter.values())
        delete_entity_nums = int(entity_count * p)
        delete_dict = {}
        delete_tag_cnt = 0
        for e_type, num in self.e_type_counter.items():
            if other_delete_all and e_type == "O":
                e_type_delete_nums = num
            else:
                e_type_delete_nums = int(num/entity_count*delete_entity_nums)
            delete_dict[e_type] = np.random.choice(num, e_type_delete_nums, replace=False)
            delete_tag_cnt += e_type_delete_nums
            print("TYPE:{:<15}before {},\tafter {}".format(e_type + ",", num, num - e_type_delete_nums))

        for sentence_entities in self.entities:
            for entity in sentence_entities:
                if entity.e_type_index in delete_dict[entity.e_type]:
                    entity.e_type = UNLABELED_TAG

        print("TYPE:{:<15}before {},\t\tafter {}".format(UNLABELED_TAG + ",", 0, delete_tag_cnt))

    def write(self, file_path):
        text = ""
        for sentence_entities in self.entities:
            for entity in sentence_entities:
                text += entity.to_line()
            text += "\n"

        with open(file_path, mode="w") as f:
            f.write(text)
