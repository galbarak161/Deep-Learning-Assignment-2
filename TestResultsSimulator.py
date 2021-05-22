import os

import torch
from torch.utils.data import DataLoader

from GTSRBDataset import GTSRBDataset
from GTSRBModel import GTSRBModel, DEVICE
from Transforms import DEFAULT_TRANSFORM
from model.ModelMeta import PATH_TO_MODEL


def main():

    batch_size = 128
    num_workers = 4
    test_dataset = GTSRBDataset(train=False, transform=DEFAULT_TRANSFORM)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    model_num = 3
    model = GTSRBModel(model_num, dropout=True, batch_normalization=True, fully_connected_layers=False)

    path_to_model = os.path.join(PATH_TO_MODEL, f'model_{model_num}.pth')
    if os.path.isfile(path_to_model):
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device(DEVICE)))

    test_acc, test_loss = model.calculate_accuracy_and_loss(test_loader)
    print(f'Test: Accuracy = {(test_acc * 100):.2f}%, Avg Loss = {test_loss:.2f}')


if __name__ == '__main__':
    main()
