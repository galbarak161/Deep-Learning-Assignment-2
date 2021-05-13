from torch import nn
from torch.utils.data import DataLoader

from Models.util.Consts import *


class GTSRBModel(nn.Module):

    def __init__(self, dataLoaders: dict, dropout=False, batchNormalization=False, withFC=True):
        super().__init__()
        self.dataLoaders = dataLoaders

        self.lossFunction = torch.nn.CrossEntropyLoss()

        self.logSoftMax = nn.LogSoftmax()

        self.featureExtractor = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=100, kernel_size=5, stride=1, padding=1, bias=False),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=3),
            # nn.Conv2d(in_channels=24, out_channels=150, kernel_size=(3,)),
            # nn.ReLU()
        )

        if withFC:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                # nn.Linear(in_features=150, out_features=43)
                # can add more Relu and linear
            )
        else:
            self.classifier = nn.Sequential(
                # ...

                # nn.Conv2d(in_channels=6, out_channels=43, kernel_size=(1,))
            )

        self.optimizer = torch.optim.Adam(self.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3)
        self.to(DEVICE)

    def forward(self, x):
        features = self.featureExtractor(x)
        classScores = self.classifier(features)
        return self.logSoftMax(classScores)

    def countParameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def getPredictions(self, pathToImage):
        # need to return prediction with prob
        # it's for the UI, will implement later
        pass

    def calculateAccuracy(self, dataSet: DataLoader, withGrad=True) -> float:

        with torch.set_grad_enabled(withGrad):
            nCorrect = 0
            total = 0
            for data, label in dataSet:
                # calculate output
                predictionsProbabilities = self(data)

                # get the prediction
                prediction = torch.argmax(predictionsProbabilities, dim=1)
                nCorrect += torch.sum(prediction == label.to(DEVICE)).type(torch.float32)
                total += data.shape[0]

            return (nCorrect / total).item()

    def trainModel(self, epochs, withEarlyStopping=True):

        # early stopping params
        bestValidationAcc = 0
        patienceLimit = 20
        patienceCounter = 0
        needToStop = False

        epoch = 0
        while epoch < epochs and not needToStop:

            for data, labels in self.dataLoaders[TRAIN]:
                self.optimizer.zero_grad()

                predictions = self(data)
                loss = self.lossFunction(predictions, labels).to(DEVICE)
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            if withEarlyStopping:
                valAcc = self.calculateAccuracy(self.dataLoaders[VALID], withGrad=False)

                if valAcc > bestValidationAcc:
                    bestValidationAcc = valAcc
                    torch.save(self.state_dict(), PATH_TO_MODEL)
                    patienceCounter = 0
                else:
                    patienceCounter += 1
                    if patienceCounter >= patienceLimit:
                        needToStop = True

            epoch += 1
