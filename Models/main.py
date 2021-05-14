import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.transforms import transforms

from Models.GTSRBDataset import GTSRBDataset
from Models.GTSRBModel import GTSRBModel
from Models.util.Consts import *


def main():
    # dataset initialization
    transform = transforms.Compose([
        transforms.Resize((30, 30)),
        transforms.ToTensor(),
        transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
    ])

    trainDataset = GTSRBDataset(transform=transform)
    testDataset = GTSRBDataset(train=False, transform=transform)

    validationRatio = 0.2

    indices = list(range(len(trainDataset)))
    np.random.shuffle(indices)
    split = int(np.floor(validationRatio * len(trainDataset)))
    trainSample = SubsetRandomSampler(indices[:split])
    validSample = SubsetRandomSampler(indices[split:])

    dataLoaders = {
        TRAIN: DataLoader(trainDataset, batch_size=64, num_workers=4, sampler=trainSample),
        VALID: DataLoader(trainDataset, batch_size=64, num_workers=4, sampler=validSample),
        TEST: DataLoader(testDataset, batch_size=64, num_workers=4)
    }

    epochs = 1

    # 1st model
    print('\n------------------------1st Model------------------------')
    model1 = GTSRBModel(1)
    print(model1)
    model1.trainModel(epochs, dataLoaders)

    # 2nd model
    print('\n------------------------2nd Model-----------------------')
    model2 = GTSRBModel(2, dropout=True, batch_normalization=True)
    print(model2)
    model2.trainModel(epochs, dataLoaders)

    # 3rd model
    print('\n------------------------3rd Model------------------------')
    model3 = GTSRBModel(3, dropout=True, batch_normalization=True, fully_connected_nn=False)
    print(model3)
    model3.trainModel(epochs, dataLoaders)


if __name__ == '__main__':
    main()
