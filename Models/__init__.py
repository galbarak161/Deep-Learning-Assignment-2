import os

import torch

from Models.GTSRBModel import GTSRBModel

# load the trained model
from Models.model.ModelMeta import PATH_TO_MODEL

model_num = 3

model = GTSRBModel(model_num, dropout=True, batch_normalization=True, fully_connected_nn=False)
model.load_state_dict(torch.load(os.path.join(PATH_TO_MODEL, f'model_{model_num}.pth')))
