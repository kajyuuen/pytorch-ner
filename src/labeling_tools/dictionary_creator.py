import numpy as np

from collections import defaultdict

from torchtext import data, datasets

from src.data.conll_dataset import Conll2003Dataset
from src.labeling_tools.data import Entity
from src.common.config import UNLABELED_TAG

class DictionaryCreator:
    def __init__(self, file_path):
        WORD = data.Field(include_lengths = True, batch_first = True, lower = False)
        LABEL = data.Field(unk_token = UNLABELED_TAG, batch_first = True)
        fields = [('word', WORD), ('label', LABEL)]

        self.dataset = Conll2003Dataset(fields = fields, path = file_path, separator = " ")
        self.e_type_counter = defaultdict(int)
        self.create_entity()

    def create_entity(self):
        entities = []
        entities_words = set()
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
            for t_entity in sentence_entities:
                entity_word = " ".join(t_entity.words)
                if entity_word not in entities_words:
                    entities_words.add(entity_word)
                    entities.append(t_entity)
        self.entities = entities
    
    def write(self, file_path):
        text = ""
        for entity in self.entities:
            if entity.e_type == "O":
                continue
            text += "{} {}\n".format(" ".join(entity.words), entity.e_type)

        with open(file_path, mode="w") as f:
            f.write(text)
