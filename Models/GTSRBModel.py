import os

from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as nn_functional
from torch.utils.data import DataLoader

from Models.Transforms import DEFAULT_TRANSFORM
from Models.plots.PlotsMeta import PATH_TO_PLOTS
from Models.model.ModelMeta import PATH_TO_MODEL
from Models.util.Consts import *


class GTSRBModel(nn.Module):

    def __init__(self, model_id, dropout=False, batch_normalization=False, fully_connected_nn=True):
        super().__init__()

        # Hyper-parameters
        rate = 0.001
        weight_decay = 0.001
        dropout_p = 0.2

        self.modelId = model_id
        self.lossFunction = torch.nn.CrossEntropyLoss()

        self.logSoftMax = nn.LogSoftmax(dim=1)

        self.featureExtractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
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
            self.classifier.add_module(f'{model_id}', nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(4, 4)))
            model_id += 1
            if batch_normalization:
                self.classifier.add_module(f'{model_id}', nn.BatchNorm2d(120))
                model_id += 1

        self.classifier.add_module(f'{model_id}', nn.ReLU())
        model_id += 1

        if dropout:
            self.classifier.add_module(f'{model_id}', nn.Dropout(dropout_p))
            model_id += 1

        if fully_connected_nn:
            self.classifier.add_module(f'{model_id}', nn.Linear(120, 84))
            model_id += 1
            if batch_normalization:
                self.classifier.add_module(f'{model_id}', nn.BatchNorm1d(84))
                model_id += 1

        else:
            self.classifier.add_module(f'{model_id}', nn.Conv2d(in_channels=120, out_channels=84, kernel_size=(1, 1)))
            model_id += 1
            if batch_normalization:
                self.classifier.add_module(f'{model_id}', nn.BatchNorm2d(84))
                model_id += 1

        self.classifier.add_module(f'{model_id}', nn.ReLU())
        model_id += 1

        if dropout:
            self.classifier.add_module(f'{model_id}', nn.Dropout(dropout_p))
            model_id += 1

        if fully_connected_nn:
            self.classifier.add_module(f'{model_id}', nn.Linear(84, 43))
            model_id += 1

        else:
            self.classifier.add_module(f'{model_id}', nn.Conv2d(in_channels=84, out_channels=43, kernel_size=(1, 1)))
            model_id += 1

        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(7, 7)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=(5, 5)),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 4 * 4, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.8)

        self.to(DEVICE)

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 4 * 4)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = nn_functional.affine_grid(theta, x.size(), align_corners=True)
        x = nn_functional.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        """# Perform forward pass
        x = self.bn1(F.max_pool2d(F.leaky_relu(self.conv1(x)), 2))
        x = self.conv_drop(x)
        x = self.bn2(F.max_pool2d(F.leaky_relu(self.conv2(x)), 2))
        x = self.conv_drop(x)
        x = self.bn3(F.max_pool2d(F.leaky_relu(self.conv3(x)), 2))
        x = self.conv_drop(x)
        x = x.view(-1, 250 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)"""
        features = self.featureExtractor(x)
        class_scores = self.classifier(features)
        probabilities = self.logSoftMax(class_scores)
        return probabilities

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_predictions(self, path_to_image):
        img = Image.open(path_to_image)
        img = DEFAULT_TRANSFORM(img)
        img = img.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            probabilities = self(img).squeeze()
            prediction = torch.argmax(probabilities).item()

            return prediction

    def calculate_accuracy(self, data_set: DataLoader, with_grad=True) -> float:
        self.eval()
        with torch.set_grad_enabled(with_grad):
            n_correct = 0
            total = 0
            for data, labels in data_set:
                data, labels = data.to(DEVICE), labels.to(DEVICE)

                # calculate output
                predictions_probabilities = self(data).squeeze()

                # get the prediction
                predictions = torch.argmax(predictions_probabilities, dim=1)
                n_correct += torch.sum(predictions == labels).item()
                total += data.shape[0]

            return n_correct / total

    def train_model(self, epochs: int, data_loaders: dict, with_early_stopping=True):

        # early stopping params
        best_validation_acc = 0
        patience_limit = 20
        patience_counter = 0
        need_to_stop = False
        best_model_epoch_number = 0
        val_accuracies = []

        epoch = 0
        while epoch < epochs and not need_to_stop:
            self.train()

            for data, labels in data_loaders[TRAIN]:
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
                val_acc = self.calculate_accuracy(data_loaders[VALID], with_grad=False)
                val_accuracies.append(val_acc)
                if val_acc > best_validation_acc:
                    best_validation_acc = val_acc
                    torch.save(self.state_dict(), os.path.join(PATH_TO_MODEL, f'model_{self.modelId}.pth'))
                    patience_counter = 0
                    best_model_epoch_number = epoch
                else:
                    patience_counter += 1
                    if patience_counter >= patience_limit:
                        need_to_stop = True
                        print(f'early stopping after {epoch} / {epochs}')

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

        validation_acc = self.calculate_accuracy(data_loaders[VALID], with_grad=False)
        test_acc = self.calculate_accuracy(data_loaders[TEST], with_grad=False)
        print(f'Validation Accuracy: {(validation_acc * 100):.2f}%')
        print(f'Test Accuracy: {(test_acc * 100):.2f}%')
        print()
