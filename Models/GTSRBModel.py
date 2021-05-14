from torch import nn
from torch.utils.data import DataLoader

from Models.util.Consts import *


class GTSRBModel(nn.Module):

    def __init__(self, data_loaders: dict, dropout=False, batch_normalization=False, fully_connected_nn=True):
        super().__init__()

        self.dataLoaders = data_loaders

        self.lossFunction = torch.nn.CrossEntropyLoss()

        self.logSoftMax = nn.LogSoftmax()

        self.featureExtractor = nn.Sequential(
            nn.Conv2d(3, out_channels=6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential()
        model_id = 0
        if fully_connected_nn:
            self.classifier.add_module(f'{model_id}', nn.Flatten())
            model_id += 1
            self.classifier.add_module(f'{model_id}', nn.Linear(16 * 4 * 4, 120))
            model_id += 1
            if batch_normalization:
                self.classifier.add_module(f'{model_id}', nn.BatchNorm1d(120))
                model_id += 1

        else:
            self.classifier.add_module(f'{model_id}', nn.Conv2d(in_channels=16, out_channels=120, kernel_size=4))
            model_id += 1
            if batch_normalization:
                self.classifier.add_module(f'{model_id}', nn.BatchNorm2d(120))
                model_id += 1

        if dropout:
            self.classifier.add_module(f'{model_id}', nn.Dropout(0.2))
            model_id += 1

        self.classifier.add_module(f'{model_id}', nn.ReLU())
        model_id += 1

        if fully_connected_nn:
            self.classifier.add_module(f'{model_id}', nn.Linear(120, 84))
            model_id += 1
            if batch_normalization:
                self.classifier.add_module(f'{model_id}', nn.BatchNorm1d(84))
                model_id += 1

        else:
            self.classifier.add_module(f'{model_id}', nn.Conv2d(in_channels=120, out_channels=84, kernel_size=1))
            model_id += 1
            if batch_normalization:
                self.classifier.add_module(f'{model_id}', nn.BatchNorm2d(84))
                model_id += 1

        if dropout:
            self.classifier.add_module(f'{model_id}', nn.Dropout(0.2))
            model_id += 1

        self.classifier.add_module(f'{model_id}', nn.ReLU())
        model_id += 1

        if fully_connected_nn:
            self.classifier.add_module(f'{model_id}', nn.Linear(84, 43))
            model_id += 1

        else:
            self.classifier.add_module(f'{model_id}', nn.Conv2d(in_channels=84, out_channels=43, kernel_size=1))
            model_id += 1

        self.optimizer = torch.optim.Adam(self.parameters())
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3)

        self.dropout = dropout

        self.to(DEVICE)

    def forward(self, x):
        features = self.featureExtractor(x)
        class_scores = self.classifier(features)
        return class_scores

    def countParameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def getPredictions(self, path_to_image):
        self.eval()
        # need to return prediction with prob
        # it's for the UI, will implement later
        pass

    def calculateAccuracy(self, data_set: DataLoader, with_grad=True) -> float:

        with torch.set_grad_enabled(with_grad):
            nCorrect = 0
            total = 0
            for data, label in data_set:
                data, label = data.to(DEVICE), label.to(DEVICE)

                # calculate output
                predictionsProbabilities = self(data)

                # get the prediction
                prediction = torch.argmax(predictionsProbabilities, dim=1)
                nCorrect += torch.sum(prediction == label.to(DEVICE)).type(torch.float32)
                total += data.shape[0]

            return (nCorrect / total).item()

    def trainModel(self, epochs, with_early_stopping=True):

        # early stopping params
        bestValidationAcc = 0
        patienceLimit = 20
        patienceCounter = 0
        needToStop = False

        epoch = 0
        while epoch < epochs and not needToStop:
            self.train()

            for data, labels in self.dataLoaders[TRAIN]:
                data, labels = data.to(DEVICE), labels.to(DEVICE)

                self.optimizer.zero_grad()
                predictions = self(data).to(DEVICE)
                loss = self.lossFunction(predictions, labels).to(DEVICE)
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            if with_early_stopping:
                self.eval()
                valAcc = self.calculateAccuracy(self.dataLoaders[VALID], with_grad=False)
                if valAcc > bestValidationAcc:
                    bestValidationAcc = valAcc
                    torch.save(self.state_dict(), PATH_TO_MODEL)
                    patienceCounter = 0
                else:
                    patienceCounter += 1
                    if patienceCounter >= patienceLimit:
                        needToStop = True

            epoch += 1
