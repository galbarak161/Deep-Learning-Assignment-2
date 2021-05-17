import os

import torch

from Models.GTSRBModel import GTSRBModel

from Models.model.ModelMeta import PATH_TO_MODEL

# load the trained model
model_num = 3
model = GTSRBModel(model_num, dropout=True, batch_normalization=True, fully_connected_layers=False)

path_to_model = os.path.join(PATH_TO_MODEL, f'model_{model_num}.pth')
if os.path.isfile(path_to_model):
    model.load_state_dict(torch.load(path_to_model))
