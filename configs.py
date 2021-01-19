import torch


class BaseConfigs:
    MAX_SENTENCE_LENGTH=80+80
    DEVICE = torch.device('cuda')

class RobertaConfigs(BaseConfigs):
    BATCH_SIZE = 16

    ROBERTA_DIM=1024
    LINEAR_HIDDEN_DIM=256
