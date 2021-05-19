import os
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

from Models.GTSRBDataset import TRAIN, VALID, TEST
from Models.Transforms import DEFAULT_TRANSFORM
from Models.plots.PlotsMeta import PATH_TO_PLOTS
from Models.model.ModelMeta import PATH_TO_MODEL

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GTSRBModel(nn.Module):

    def __init__(self, model_id, dropout=False, batch_normalization=False, fully_connected_layers=True):
        super().__init__()

        # Hyper parameters
        rate = 0.001
        weight_decay = 0.001
        dropout_p = 0.2

        self.modelId = model_id
        self.lossFunction = torch.nn.CrossEntropyLoss()

        self.logSoftMax = nn.LogSoftmax(dim=1)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential()
        model_id = 0
        if fully_connected_layers:
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

        if fully_connected_layers:
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

        if fully_connected_layers:
            self.classifier.add_module(f'{model_id}', nn.Linear(84, 43))
            model_id += 1

        else:
            self.classifier.add_module(f'{model_id}', nn.Conv2d(in_channels=84, out_channels=43, kernel_size=(1, 1)))
            model_id += 1

        self.optimizer = torch.optim.Adam(self.parameters(), lr=rate, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.8)

        self.to(DEVICE)

    def forward(self, x):
        features = self.feature_extractor(x)
        class_scores = self.classifier(features)
        probabilities = self.logSoftMax(class_scores)
        return probabilities

    def get_predictions(self, path_to_image):
        img = Image.open(path_to_image)
        img = DEFAULT_TRANSFORM(img)
        img = img.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            probabilities = self(img).squeeze()
            prediction = torch.argmax(probabilities).item()

            return prediction

    def calculate_accuracy_and_loss(self, data_set: DataLoader):
        self.eval()
        with torch.no_grad():
            n_correct = 0
            total = 0

            losses = []

            for data, labels in data_set:
                data, labels = data.to(DEVICE), labels.to(DEVICE)

                # calculate output
                predictions_probabilities = self(data).squeeze().to(DEVICE)

                loss = self.lossFunction(predictions_probabilities, labels).to(DEVICE)
                losses.append(loss.detach())
                # get the prediction
                predictions = torch.argmax(predictions_probabilities, dim=1)
                n_correct += torch.sum(predictions == labels).item()
                total += data.shape[0]

            return n_correct / total, torch.mean(torch.stack(losses).cpu(), dim=0)

    def train_model(self, epochs: int, data_loaders: dict):
        # early stopping params
        best_validation_acc = 0
        patience_limit = 20
        patience_counter = 0
        need_to_stop = False
        best_model_epoch_number = 0

        # results lists
        train_accuracies = []
        train_losses = []
        val_accuracies = []
        valid_losses = []

        # training loop
        epoch = 0
        while epoch < epochs and not need_to_stop:

            # train phase
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

            # calc acc and losses
            train_acc, train_loss = self.calculate_accuracy_and_loss(data_loaders[TRAIN])
            train_accuracies.append(train_acc)
            train_losses.append(train_loss)
            val_acc, val_loss = self.calculate_accuracy_and_loss(data_loaders[VALID])
            valid_losses.append(val_loss)
            val_accuracies.append(val_acc)

            # early stopping check
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

        # accuracy plots
        fig = plt.figure()
        val_acc_plot_name = f'Accuracy per epoch - Model_{self.modelId}'
        plt.title(val_acc_plot_name)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Validation Accuracy')

        plt.plot(best_model_epoch_number, val_accuracies[best_model_epoch_number], 'r*',
                 label='Best Validation Accuracy')
        plt.legend()
        fig.savefig(os.path.join(PATH_TO_PLOTS, val_acc_plot_name))
        plt.close(fig)
        plt.clf()

        # loss plots
        val_loss_plot_name = f'Loss per epoch - Model_{self.modelId}'
        fig = plt.figure()
        plt.title(val_loss_plot_name)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(valid_losses, label='Validation Loss')

        plt.plot(best_model_epoch_number, valid_losses[best_model_epoch_number], 'r*',
                 label='Best Validation Accuracy')
        plt.legend()
        fig.savefig(os.path.join(PATH_TO_PLOTS, val_loss_plot_name))
        plt.close(fig)
        plt.clf()

        # final results
        test_acc, test_loss = self.calculate_accuracy_and_loss(data_loaders[TEST])
        print(f'Train: Accuracy = {(train_accuracies[-1] * 100):.2f}%, Avg Loss = {train_losses[-1]:.2f}')
        print(f'Validation: Accuracy = {(val_accuracies[-1] * 100):.2f}%, Avg Loss = {valid_losses[-1]:.2f}')
        print(f'Test: Accuracy = {(test_acc * 100):.2f}%, Avg Loss = {test_loss:.2f}')
        print()
