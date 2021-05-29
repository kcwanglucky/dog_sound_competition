import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset, DataLoader
from IPython.display import Audio, display

from sklearn.metrics import roc_auc_score
from torch.utils.data import random_split

from utils import *
from config import path_config

class AudioProcessor:
    """ Preprocess audio files """
    def read(audio_file):
        """Load an audio file. Return the signal as a tensor and the sample rate"""
        waveform, sample_rate = torchaudio.load(audio_file)
        return (waveform, sample_rate) #if waveform.shape == self.shape else None
    
    def spectro_gram(waveform, sample_rate, n_mels=64, n_fft=1024, hop_len=None):
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(waveform)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)

class AudioDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, path_prefix, files, y):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path_prefix = path_prefix
        self.files = files
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        filename = self.files[idx]
        filepath = os.path.join(self.path_prefix, filename)
        
        waveform, sample_rate = AudioProcessor.read(filepath)
        
        specgram = AudioProcessor.spectro_gram(waveform, sample_rate, n_mels=64, n_fft=1024, hop_len=None)
        
        # TODO: Need to support using "Remark" as well
        y = self.y[self.y["Filename"] == filename.split('.')[0]]["Label"].to_numpy()
        
        sample = {'x': specgram, 'y': y}

        return sample

class AudioClassifier(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        torch.nn.init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        torch.nn.init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)
        
        # Linear Classifier
        self.pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.linear = nn.Linear(in_features=16, out_features=num_class)

    def forward(self, x):
        # Run the convolutional blocks
        x = self.conv(x)

        # Adaptive pool and flatten for input to linear layer
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)

        # Linear layer
        x = self.linear(x)

        # Final output
        return x

def explore_y(metadata):
    """ Show the number of class. Two labelling schemes available """
    label = metadata["Label"].unique()
    print("Number of y".format(len(label)))

    print("Number of label: {}".format(len(label)))
    print("Label list: {}".format(label))

    remark = metadata["Remark"].unique()
    print("Number of remark: {}".format(len(remark)))
    print("remark list: {}".format(remark))

    return {"metadata": metadata, "label": label, "remark": remark}

def explore_x(path_to_train_dir):
    files = os.listdir(path_to_train_dir)
    waveform, sample_rate = AudioProcessor.read(os.path.join(path_to_train_dir, files[0]))

    print("Number of samples: {}".format(len(files)))
    print("waveform shape: {}".format(waveform.shape))

    return {"len": len(files), "waveform_shape": waveform.shape}

def data_cleaning(path_to_train_dir, shape, metadata):
    """ Remove the sample that is an anomaly """
    files = os.listdir(path_to_train_dir)

    print("Normal shape: {}\n".format(shape))

    num_files_bf = len(files)
    to_remove = []  # collect samples whose shape is different from input `shape`
    count = 0
    for path in files:
        # print(os.path.join(TRAIN_AUDIO_PATH, path))
        waveform, sample_rate = AudioProcessor.read(os.path.join(path_to_train_dir, path))
        if waveform.shape != shape:
            to_remove.append(path)
            count += 1
            print("Anomaly Shape Detected: {}\nShape: {}\n".format(path, waveform.shape))

    for file in to_remove:
        drop_idx = metadata[metadata["Filename"] == file.split('.')[0]].index
        metadata = metadata.drop(drop_idx)
        files.remove(file)
        
    print("Total Anomaly: {}".format(count))
    print("Before deleting anomaly: {} numbers of audios".format(num_files_bf))
    print("After deleting anomaly: {} numbers of audios".format(len(files)))

    assert len(files) == len(metadata)
    print("Number of valid sample x: {}".format(len(files)))
    print("Number of valid sample y: {}".format(len(metadata)))

    return {"x_file_paths": files, "y": metadata}

def data_split(dataset):
    num_items = len(dataset)
    num_train = round(num_items * 0.8)
    num_val = num_items - num_train
    train_ds, val_ds = random_split(dataset, [num_train, num_val])

    train_dataloader = DataLoader(train_ds, batch_size=32,
                        shuffle=True, num_workers=0)

    val_dataloader = DataLoader(val_ds, batch_size=32,
                        shuffle=False, num_workers=0)

    return {"train_ds": train_ds, "val_ds": val_ds, "train_dl": train_dataloader, "val_dl": val_dataloader}

def check_shape(dataloader):
    for i_batch, sample_batched in enumerate(dataloader):
        print("x_size: {} \ty_size: {}".format(
            sample_batched['x'].size(), sample_batched['y'].size()))
        if i_batch == 1:
            break

def train(data, num_class):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    train_ds, val_ds = data["train_ds"], data["val_ds"]
    train_dl, val_dl = data["train_dl"], data["val_dl"]

    model = AudioClassifier(num_class = num_class)
    device = torch.device(device)
    model = model.to(device)
    print(model)

    loss_fnc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 15

    training_loop(train_dl, val_dl, model, loss_fnc, optimizer, num_epochs)

def training_loop(train_dl, val_dl, model, loss_fn, optimizer, num_epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Loss Function, Optimizer and Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            x, y_true = data['x'].to(device), data['y'].to(device).reshape(-1)

            # Normalize the inputs
            mean, std = x.mean(), x.std()
            x = (x - mean) / std

            # forward + backward + optimize
            outputs = model(x)
            loss = loss_fn(outputs, y_true)

            # Backpropagation
            optimizer.zero_grad()   # Zero the parameter gradients

            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, y_pred = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (y_pred == y_true).sum().item()
            total_prediction += y_pred.shape[0]

            if i % 10 == 0:    # print every 10 mini-batches
                oh_ytrue = torch.nn.functional.one_hot(y_true)
                y_prob = torch.nn.functional.softmax(outputs, dim=1).detach().numpy()
                auc = roc_auc_score(oh_ytrue, y_prob, multi_class="ovr")
                print("Epoch {} Batch {} Result".format(epoch + 1, i + 1))
                print('\tLoss: %.3f' % (running_loss / (i + 1)))

        val_acc, val_avg_loss, val_auc = validation_loop(val_dl, model, loss_fn)

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction
        
        print("Epoch: {}".format(epoch + 1))
        print("Train Stats:")
        print(f'\tLoss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

        print("Validation Stats:")
        print(f'\tLoss: {val_avg_loss:.2f}, Accuracy: {val_acc:.2f}, Auc: {val_auc:.2f}')

    print('Finished Training')

def validation_loop(val_dl, model, loss_fn):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    size = len(val_dl.dataset)

    val_loss = correct_prediction = total_prediction = 0
    y_true_all, y_pred_all, y_prob_all = [], [], []

    with torch.no_grad():
        for i, data in enumerate(val_dl):
            x, y_true = data['x'].to(device), data['y'].to(device).reshape(-1)

            # Normalize the inputs
            mean, std = x.mean(), x.std()
            x = (x - mean) / std

            # forward + backward + optimize
            outputs = model(x)
            y_prob = torch.nn.functional.softmax(outputs, dim=1)

            val_loss += loss_fn(outputs, y_true).item()

            # Get the predicted class with the highest score
            _, y_pred = torch.max(outputs, 1)

            # Count of predictions that matched the target label
            correct_prediction += (y_pred == y_true).sum().item()
            total_prediction += y_pred.shape[0]

            y_true_all.append(y_true.cpu())
            y_pred_all.append(y_pred.cpu())
            y_prob_all.append(y_prob.cpu())

    acc = correct_prediction/total_prediction
    
    # oh_ytrue = torch.nn.functional.one_hot(y_true)
    # y_prob = torch.nn.functional.softmax(outputs, dim=1).detach().numpy()
    # auc = roc_auc_score(oh_ytrue, y_prob, multi_class="ovr")

    # y_prob = torch.nn.functional.softmax(outputs, dim=1)
    y_true_all = torch.cat(y_true_all, dim = 0).detach().cpu().numpy()
    y_pred_all = torch.cat(y_pred_all, dim = 0).detach().cpu().numpy()
    y_prob_all = torch.cat(y_prob_all, dim = 0).detach().cpu().numpy()

    auc = roc_auc_score(y_true_all, y_prob_all, multi_class="ovr")

    return acc, val_loss / size, auc

def main():
    train_audio_path = path_config["train_audio_path"]
    metadata = pd.read_csv("./train/meta_train.csv")

    x_summary = explore_x(train_audio_path)
    y_summary = explore_y(metadata)

    data = data_cleaning(train_audio_path, x_summary["waveform_shape"], metadata)

    audio_dataset = AudioDataset(train_audio_path, data["x_file_paths"], data["y"])
    # print(audio_dataset[500])

    data = data_split(audio_dataset)

    # check_shape(data["train_dataloader"])

    train(data, len(y_summary["label"]))



if __name__ == "__main__":
    main()