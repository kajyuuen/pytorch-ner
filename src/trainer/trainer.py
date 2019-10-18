import os

import torch
import torch.optim as optim
from torchtext import data, datasets

from tqdm import tqdm
from seqeval.metrics import classification_report
from seqeval.metrics import f1_score

from src.common.utils import convert
from src.common.utils import lr_decay

import logging
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self,
                 model,
                 trainer_config,
                 train_dataset,
                 valid_dataset,
                 test_dataset = None,
                 label_dict = None):
        self.model = model
        self.label_dict = label_dict
        if trainer_config["optimizer"] == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(),
                                    lr = trainer_config["learning_rate"],
                                    weight_decay = trainer_config["weight_decay"])
        elif trainer_config["optimizer"] == "Adam":
            self.optimizer = optim.Adam(self.model.parameters())
        else:
            raise ModuleNotFoundError

        self.save_path = trainer_config["path"]
        self.clipping = trainer_config["clipping"]
        self.train_only = trainer_config["train_only"]
        if os.path.exists(self.save_path + "/result.log"):
            os.remove(self.save_path + "/result.log")
        if self.save_path:
            os.makedirs(self.save_path, exist_ok = True)
            self.model_path = self.save_path + "/models"
            self.best_model_path = self.model_path + "/best_model"
            os.makedirs(self.model_path, exist_ok = True)

        # Create iterator
        train_batch_size = trainer_config["train_batch_size"]            
        eval_batch_size = len(valid_dataset) if trainer_config["eval_batch_size"] is None else trainer_config["eval_batch_size"]
        test_batch_size = len(test_dataset) if trainer_config["test_batch_size"] is None else trainer_config["test_batch_size"]
        if test_dataset is None or label_dict is None:
            self.train_iter, self.valid_iter = data.Iterator.splits((train_dataset, valid_dataset), 
                                                                    batch_sizes = (train_batch_size, eval_batch_size),
                                                                    device = trainer_config["device"])
            self.test_iter = None                                                        
        else:
            self.train_iter, self.valid_iter, self.test_iter = data.Iterator.splits((train_dataset, valid_dataset, test_dataset), 
                                                                        batch_sizes = (train_batch_size, eval_batch_size, test_batch_size),
                                                                        device = trainer_config["device"])
        self.train_iter.repeat = False

    def train(self, num_epochs):
        best_iteration, best_valid_f1, best_test_f1 = 0, 0, 0
        for ind in range(1, num_epochs+1):
            train_loss = self._iteration(ind, self.train_iter, "Training", is_train = True)
            valid_loss = self._iteration(ind, self.valid_iter, "Validing", is_train = False)
            y_true, y_pred = self._get_labels(self.valid_iter)
            valid_f1 = f1_score(y_true, y_pred)
            if best_valid_f1 < valid_f1:
                best_valid_f1 = valid_f1
                best_iteration = ind
                # Save model
                if self.save_path is not None:
                    try:
                        os.remove(self.best_model_path)
                    except FileNotFoundError as e:
                        pass
                    torch.save(self.model.state_dict(), self.best_model_path)

            loss_text = "Train Loss: {}, Valid Loss: {}".format(train_loss, valid_loss)
            logger.info(loss_text)

            # Test
            if self.test_iter is None:
                continue
            y_true, y_pred = self._get_labels(self.test_iter)
            with open(self.save_path + "/result.log", "a") as f:
                f.write("\nepoch: {}, {}, Valid F1: {}\n".format(ind, loss_text, valid_f1))
                f.write(classification_report(y_true, y_pred, digits=5))
                if best_iteration == ind:
                    best_test_f1 = f1_score(y_true, y_pred)

        with open(self.save_path + "/result.log", "a") as f:
            f.write("\nBest epoch: {}, Test F1: {}\n".format(best_iteration, best_test_f1))

    def _iteration(self, epoch_ind, iteration, type_name, is_train = False):
        if is_train:
            self.model.train()
            self.model.zero_grad()
        else:
            self.model.eval()

        all_loss = 0
        count = 0
        progress_iter = tqdm(iteration, leave=False)
        for batch in progress_iter:
            if is_train:
                self.optimizer.zero_grad()
            loss = self.model(batch)
            if is_train:
                loss.backward()
                self.optimizer.step()
                self.model.zero_grad()

            all_loss += loss.item()
            count += 1
            mean_loss = all_loss/count

            progress_iter.set_description('[{}] {}'.format(epoch_ind, type_name))
            progress_iter.set_postfix(mean_loss=(mean_loss))
        return all_loss

    def _get_labels(self, iteration):
        self.model.eval()
        y_true, y_pred = [], []
        for batch in iteration:
            predict_tags = self.model.decode(batch)
            label_seq_tensor = batch.label
            y_true.extend(convert(label_seq_tensor.tolist(), self.label_dict))
            y_pred.extend(convert(predict_tags, self.label_dict))
        return y_true, y_pred