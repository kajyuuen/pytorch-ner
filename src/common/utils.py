import torch

from src.common.config import PAD_TAG, UNLABELED_ID

def create_possible_tag_masks(num_tags: int, tags: torch.Tensor) -> torch.Tensor:
    copy_tags = tags.clone()
    no_annotation_idx = (copy_tags == UNLABELED_ID)
    copy_tags[copy_tags == UNLABELED_ID] = 0

    tags_ = torch.unsqueeze(copy_tags, 2)
    masks = torch.zeros(tags_.size(0), tags_.size(1), num_tags, dtype=torch.uint8, device=tags.device)
    masks.scatter_(2, tags_, 1)
    masks[no_annotation_idx] = 1
    return masks

def convert(predict_tags, label2idx):
    index_dict = {v: k for k, v in label2idx.items()}
    PAD_INDEX = label2idx[PAD_TAG]
    labels = []
    for predict_tag in predict_tags:
        label = [ index_dict[p] for p in predict_tag if p != PAD_INDEX ]
        labels.append(label)
    return labels

def lr_decay(learning_rate, lr_decay, optimizer, epoch):
    lr = learning_rate / (1 + lr_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer
