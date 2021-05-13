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

    epochs = 2

    # 1st model
    model1 = GTSRBModel(dataLoaders)
    print(model1)
    print(model1.countParameters())
    model1.trainModel(epochs)

    # 2nd model
    model2 = GTSRBModel(dataLoaders, dropout=True, batchNormalization=True)
    model2.trainModel(epochs)

    # 3rd
    model3 = GTSRBModel(dataLoaders, dropout=True, batchNormalization=True, withFC=False)
    model3.trainModel(epochs)


if __name__ == '__main__':
    main()
