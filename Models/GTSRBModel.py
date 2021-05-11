from torch import nn
from torch.utils.data import DataLoader

from Models.util.Consts import *


class GTSRBModel(nn.Module):

    def __init__(self, dataLoaders: dict, dropout=False, batchNormalization=False, withFC=False):
        super().__init__()
        self.dataLoaders = dataLoaders
        self.optimizer = torch.optim.Adam(self.parameters())
        self.lossFunction = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3)

        self.to(DEVICE)

    def forward(self):
        pass

    def getPredictions(self, pathToImage):
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
