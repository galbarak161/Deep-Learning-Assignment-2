import os

from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from Models.plots.PlotsMeta import PATH_TO_PLOTS
from Models.model.ModelMeta import PATH_TO_MODEL
from Models.util.Consts import *


class GTSRBModel(nn.Module):

    def __init__(self, modelId, dropout=False, batch_normalization=False, fully_connected_nn=True):
        super().__init__()

        self.modelId = modelId
        self.lossFunction = torch.nn.CrossEntropyLoss()

        self.logSoftMax = nn.LogSoftmax(dim=1)

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
        probabilities = self.logSoftMax(class_scores)
        return probabilities

    def countParameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def getPredictions(self, pathToImage):
        img = Image.open(pathToImage)

        transform = transforms.Compose([
            transforms.Resize((30, 30)),
            transforms.ToTensor(),
            transforms.Normalize((0.3337, 0.3064, 0.3171), (0.2672, 0.2564, 0.2629))
        ])

        img = transform(img)
        img = img.unsqueeze(0)
        with torch.no_grad():
            self.eval()
            probabilities = self(img).squeeze()
            prediction = torch.argmax(probabilities).item()

            return prediction

    def calculateAccuracy(self, data_set: DataLoader, with_grad=True) -> float:

        with torch.set_grad_enabled(with_grad):
            nCorrect = 0
            total = 0
            for data, labels in data_set:
                data, labels = data.to(DEVICE), labels.to(DEVICE)

                # calculate output
                predictionsProbabilities = self(data).squeeze()

                # get the prediction
                predictions = torch.argmax(predictionsProbabilities, dim=1)
                nCorrect += torch.sum(predictions == labels).item()
                total += data.shape[0]

            return nCorrect / total

    def trainModel(self, epochs, dataLoaders, with_early_stopping=True):

        # early stopping params
        bestValidationAcc = 0
        patienceLimit = 20
        patienceCounter = 0
        needToStop = False
        best_model_epoch_number = 0

        val_accuracies = []

        epoch = 0
        while epoch < epochs and not needToStop:
            self.train()

            for data, labels in dataLoaders[TRAIN]:
                data, labels = data.to(DEVICE), labels.to(DEVICE)

                self.optimizer.zero_grad()
                predictions = self(data).to(DEVICE)
                predictions = predictions.squeeze()

                loss = self.lossFunction(predictions, labels).to(DEVICE)
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()

            if with_early_stopping:
                self.eval()
                valAcc = self.calculateAccuracy(dataLoaders[VALID], with_grad=False)
                val_accuracies.append(valAcc)
                if valAcc > bestValidationAcc:
                    bestValidationAcc = valAcc
                    torch.save(self.state_dict(), os.path.join(PATH_TO_MODEL, f'model_{self.modelId}.pth'))
                    patienceCounter = 0
                    best_model_epoch_number = epoch
                else:
                    patienceCounter += 1
                    if patienceCounter >= patienceLimit:
                        needToStop = True

            epoch += 1

        fig = plt.figure()
        plot_name = f'Validation accuracy per epoch - Model_{self.modelId}'
        plt.title(plot_name)
        plt.plot(val_accuracies)
        if with_early_stopping:
            plt.plot(best_model_epoch_number, val_accuracies[best_model_epoch_number], 'r*',
                     label='Best Validation Accuracy')
        plt.legend()
        fig.savefig(os.path.join(PATH_TO_PLOTS, plot_name))
        plt.close(fig)
        plt.clf()

        validationAcc = self.calculateAccuracy(dataLoaders[VALID], with_grad=False)
        testAcc = self.calculateAccuracy(dataLoaders[TEST], with_grad=False)
        print(f'Validation Accuracy: {(validationAcc * 100):.2f}%')
        print(f'Test Accuracy: {(testAcc * 100):.2f}%')
        print()
