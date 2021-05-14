import torch

TRAIN = 'train'
VALID = 'validation'
TEST = 'test'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
