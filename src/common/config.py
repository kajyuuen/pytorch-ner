import torch

PAD_TAG = "<pad>"
UNK_TAG = "<unk>"
START_TAG = "<bos>"
STOP_TAG = "<eos>"
PAD_TAG = "<pad>"

UNLABELED_TAG = "NOANNOTATION"
UNLABELED_ID = -1

class Config:
    def __init__(self, setting_json):
        self.seed = 42

        # File path
        self.dataset_path = setting_json["dataset_path"]

        # Word embedding
        self.emb_dict = {
            "word_emb_dim": setting_json["model"]["embedding"]["word_embedding"]["dim"],
            "hidden_dim": setting_json["model"]["embedding"]["hidden_dim"],
            "char_emb_dim": setting_json["model"]["embedding"]["char_embedding"]["dim"],
            "char_hidden_dim": setting_json["model"]["embedding"]["char_embedding"]["hidden_dim"]
        }

        self.word_embedding_type = setting_json["model"]["embedding"]["word_embedding"]["type"]

        # Other
        self.inference_type = setting_json["model"]["type"]
        self.dropout_rate = setting_json["model"]["dropout_rate"]
        self.is_digit = setting_json["is_digit"]
        self.is_lower = setting_json["is_lower"]
        if "seed" in setting_json:
            self.seed = setting_json["seed"]
        else:
            self.seed = 42

        if "is_every_all_train" in setting_json:
            self.is_every_all_train = setting_json["is_every_all_train"]
        else:
            self.is_every_all_train = False

        # Trainer config
        if "clipping" in setting_json["train"]:
            clipping = setting_json["train"]["clipping"]
        else:
            clipping = None

        if "train_only" in setting_json["train"]:
            train_only = setting_json["train"]["train_only"]
        else:
            train_only = False

        if "weight_decay" in setting_json["train"]:
            weight_decay = setting_json["train"]["weight_decay"]
        else:
            weight_decay = 1e-08

        if "min_epochs" in setting_json["train"]:
            min_epochs = setting_json["train"]["min_epochs"]
        else:
            min_epochs = None

        batch_size = setting_json["train"]["batch_size"]
        if "eval_batch_size" in setting_json:
            eval_batch_size = setting_json["eval_batch_size"]
        else:
            eval_batch_size = None

        if "test_batch_size" in setting_json:
            test_batch_size = setting_json["test_batch_size"]
        else:
            test_batch_size = None

        if setting_json["device"] == -1:
            self.device = torch.device('cpu')
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda:{}'.format(setting_json["device"]))
            else:
                raise EnvironmentError("cuda:{} is not available.")

        self.trainer_config = {
            "train_batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "test_batch_size": test_batch_size,
            "learning_rate": setting_json["train"]["learning_rate"],
            "epochs": setting_json["train"]["epochs"],
            "min_epochs": min_epochs,
            "clipping": clipping,
            "train_only": train_only,
            "weight_decay": weight_decay,
            "path": setting_json["save_model_path"],
            "device": self.device
        }

        # Check
        assert self.inference_type in ["CRF", "PartialCRF", "Simple", "Hard"]

    def get_log(self):
        text = ""
        text += "====Path====\n"
        text += "Dataset path: {}\n".format(self.dataset_path)
        text += "Model path: {}\n".format(self.trainer_config["path"])

        text += "====Embedding====\n"
        text += "Character embedding: {}\n".format(self.emb_dict["char_emb_dim"])
        text += "Character hidden embedding: {}\n".format(self.emb_dict["char_hidden_dim"])
        text += "Word embedding: {}\n".format(self.emb_dict["word_emb_dim"])
        text += "Hideen embedding: {}\n".format(self.emb_dict["hidden_dim"])
        text += "Word type: {}\n".format(self.word_embedding_type)

        text += "====Model====\n"
        text += "Inference type: {}\n".format(self.inference_type)
        text += "Dropout rate: {}\n".format(self.dropout_rate)

        text += "====Trainer====\n"
        text += "SGD: lr {}, L2 regularization: {}\n".format(self.trainer_config["learning_rate"], self.trainer_config["weight_decay"])
        text += "epochs: {}\n".format(self.trainer_config["epochs"])
        text += "clipping: {}\n".format(self.trainer_config["clipping"])
        text += "weight_decay: {}\n".format(self.trainer_config["weight_decay"])
        text += "batch_size: {}\n".format(self.trainer_config["train_batch_size"])
    
        return text
