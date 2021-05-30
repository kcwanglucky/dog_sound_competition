import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import logging

import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset, DataLoader, dataset
from IPython.display import Audio, display

from sklearn.metrics import roc_auc_score
from torch.utils.data import random_split

from datetime import datetime

from utils import *
from config import configs
from pprint import pformat

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

    def __init__(self, dir_name, filenames, y=None, label_name=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dir_name = dir_name
        self.filenames = filenames
        self.is_test = y is None
        self.y = y
        self.label_name = label_name

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Return `x` and `y` for train and validation dataset; `x` and `filename` otherwise
        """
        filename = self.filenames[idx]
        filepath = os.path.join(self.dir_name, filename)
        
        waveform, sample_rate = AudioProcessor.read(filepath)
        specgram = AudioProcessor.spectro_gram(waveform, sample_rate, n_mels=64, n_fft=1024, hop_len=None)
        
        if not self.is_test:        # Train or val set
            y = self.y[self.y["Filename"] == filename.split('.')[0]][self.label_name].to_numpy()
            sample = {"x": specgram, "y": y}
            return sample
        else:                       # Test set
            return {"x": specgram, "filename": filename.split('.')[0]}

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

def explore_y(metadata, run_eda):
    """ Show the number of class. Two labelling schemes available """
    label = metadata["Label"].unique()
    remark = metadata["Remark"].unique()

    logging.info("Number of label: {}".format(len(label)))
    logging.info("Number of remark: {}".format(len(remark)))
    
    if run_eda:
        logging.info("Label list: {}".format(label))
        logging.info("label distribution:\n{}\n".format(metadata["Label"].value_counts()))

        logging.info("remark list: {}".format(remark))
        logging.info("remark distribution:\n{}\n".format(metadata["Remark"].value_counts()))

    return {"metadata": metadata, "label": label, "remark": remark}

def explore_x(path_to_train_dir):
    files = os.listdir(path_to_train_dir)
    waveform, sample_rate = AudioProcessor.read(os.path.join(path_to_train_dir, files[0]))

    logging.info("Number of samples: {}".format(len(files)))
    logging.info("waveform shape: {}\n".format(waveform.shape))

    return {"len": len(files), "waveform_shape": waveform.shape}

def data_cleaning(path_to_train_dir, shape, metadata):
    """ Remove the sample that is an anomaly """
    files = os.listdir(path_to_train_dir)
    logging.info("Normal shape: {}\n".format(shape))

    num_files_bf = len(files)
    to_remove = []  # collect samples whose shape is different from input `shape`
    count = 0
    for path in files:
        # print(os.path.join(TRAIN_AUDIO_PATH, path))
        waveform, sample_rate = AudioProcessor.read(os.path.join(path_to_train_dir, path))
        if waveform.shape != shape:
            to_remove.append(path)
            count += 1
            logging.info("Anomaly Shape Detected: {}\nShape: {}\n".format(path, waveform.shape))

    for file in to_remove:
        drop_idx = metadata[metadata["Filename"] == file.split('.')[0]].index
        metadata = metadata.drop(drop_idx)
        files.remove(file)
        
    logging.info("Total Anomaly: {}".format(count))
    logging.info("Before deleting anomaly: {} numbers of audios".format(num_files_bf))
    logging.info("After deleting anomaly: {} numbers of audios\n".format(len(files)))

    assert len(files) == len(metadata)
    logging.info("Number of valid sample x: {}".format(len(files)))
    logging.info("Number of valid sample y: {}\n".format(len(metadata)))

    return {"x_file_paths": files, "y": metadata}

def data_split(dataset, config):
    num_items = len(dataset)
    num_train = round(num_items * config["train_perc"])
    num_val = num_items - num_train
    train_ds, val_ds = random_split(dataset, [num_train, num_val])

    train_dataloader = DataLoader(train_ds, batch_size=config["train_batch_size"],
                        shuffle=True, num_workers=0)

    val_dataloader = DataLoader(val_ds, batch_size=config["val_batch_size"],
                        shuffle=False, num_workers=0)

    return {"train_ds": train_ds, "val_ds": val_ds, "train_dl": train_dataloader, "val_dl": val_dataloader}

def check_shape(dataloader):
    for i_batch, sample_batched in enumerate(dataloader):
        print("x_size: {} \ty_size: {}\n".format(
            sample_batched['x'].size(), sample_batched['y'].size()))
        if i_batch == 1:
            break

def train(data, num_class, config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Using {} device\n'.format(device))

    train_ds, val_ds = data["train_ds"], data["val_ds"]
    train_dl, val_dl = data["train_dl"], data["val_dl"]

    model = AudioClassifier(num_class = num_class)
    device = torch.device(device)
    model = model.to(device)

    # Loss Function, Optimizer and Scheduler
    loss_fnc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, config["max_lr"],
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=config["num_epochs"],
                                                anneal_strategy='linear')

    model = training_loop(train_dl, val_dl, model, loss_fnc, optimizer, scheduler, config["num_epochs"], config["show_plot"])

    return model

def training_loop(train_dl, val_dl, model, loss_fn, optimizer, scheduler, num_epochs, show_plot=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_train_loss, all_train_acc, all_val_loss, all_val_acc, all_val_auc = [], [], [], [], []
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
            running_loss += loss.item() * x.size(0)

            # Get the predicted class with the highest score
            _, y_pred = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (y_pred == y_true).sum().item()
            total_prediction += y_pred.shape[0]

            if i % 10 == 0:    # print every 10 mini-batches
                print("epoch {} batch {}".format(epoch + 1, i + 1))
                # oh_ytrue = torch.nn.functional.one_hot(y_true)
                # y_prob = torch.nn.functional.softmax(outputs, dim=1).detach().numpy()
                # auc = roc_auc_score(oh_ytrue, y_prob, multi_class="ovr", average='weighted')

                logging.info("Epoch {} Batch {}:".format(epoch + 1, i + 1))
                logging.info('\tLoss/Sample: %.3f' % (loss.item()))

        # Print stats at the end of the epoch
        # num_batches = len(train_dl)
        # avg_loss = running_loss / num_batches
        avg_loss = running_loss / len(train_dl.dataset)
        acc = correct_prediction/total_prediction

        logging.info("---------Epoch {}---------".format(epoch + 1))
        logging.info("Train Stats:")
        logging.info(f'\tLoss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

        val_ret = validation_loop(val_dl, model, val=True, loss_fn=loss_fn)

        logging.info("Validation Stats:")
        logging.info(f'\tLoss: {val_ret["loss"]:.2f}, Accuracy: {val_ret["acc"]:.2f}, Auc: {val_ret["auc"]:.2f}\n')

        if show_plot:
            all_train_loss.append(avg_loss)
            all_train_acc.append(acc)
            all_val_loss.append(val_ret["loss"])
            all_val_acc.append(val_ret["acc"])
            all_val_auc.append(val_ret["auc"])

    if show_plot:
        logging.info("Overall Result across epochs: ")
        logging.info("Train: ")
        logging.info("\ttrain_loss: {}".format(",".join(map(str, all_train_loss))))
        logging.info("\ttrain_acc: {}".format(",".join(map(str, all_train_acc))))

        logging.info("Val: ")
        logging.info("\tval_loss: {}".format(",".join(map(str, all_val_loss))))
        logging.info("\tval_acc: {}".format(",".join(map(str, all_val_acc))))
        logging.info("\tval_auc: {}".format(",".join(map(str, all_val_auc))))

        all_ret = [all_train_loss, all_train_acc, all_val_loss, all_val_acc, all_val_auc]
        titles = ["all_train_loss", "all_train_acc", "all_val_loss", "all_val_acc", "all_val_auc"]
        
        for title, ret in zip(titles, all_ret):
            plot_result(title, ret)

    logging.info('Finished Training\n')

    return model

def plot_result(title, nums):
    plt.plot(np.arange(1, len(nums) + 1), nums)
    plt.title(title)
    plt.xlabel("epochs")
    plt.ylabel("val")
    plt.savefig(os.path.join(logging_dir, title + ".jpg"))
    plt.close()

def validation_loop(val_dl, model, val=True, loss_fn=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    size = len(val_dl.dataset)

    val_loss = correct_prediction = total_prediction = 0
    y_true_all, y_pred_all, y_prob_all = [], [], []
    filenames = []

    with torch.no_grad(): 
        for i, data in enumerate(val_dl):
            if i % 100 == 0:
                logging.info("batch {}".format(i))
            if val:
                x, y_true = data['x'].to(device), data['y'].to(device).reshape(-1)
            else:
                x = data['x'].to(device)
                filenames.extend(data['filename'])

            # Normalize the inputs
            mean, std = x.mean(), x.std()
            x = (x - mean) / std

            # forward + backward + optimize
            outputs = model(x)
            y_prob = torch.nn.functional.softmax(outputs, dim=1)

            # Get the predicted class with the highest score
            _, y_pred = torch.max(outputs, 1)

            y_prob_all.append(y_prob.cpu())

            if val:
                val_loss += loss_fn(outputs, y_true).item()

                # Count of predictions that matched the target label
                correct_prediction += (y_pred == y_true).sum().item()
                total_prediction += y_pred.shape[0]
                y_true_all.append(y_true.cpu())

            else:
                y_pred_all.append(y_pred.cpu())
    
    # oh_ytrue = torch.nn.functional.one_hot(y_true)
    # y_prob = torch.nn.functional.softmax(outputs, dim=1).detach().numpy()
    # auc = roc_auc_score(oh_ytrue, y_prob, multi_class="ovr", average='weighted')

    # y_prob = torch.nn.functional.softmax(outputs, dim=1)
    y_prob_all = torch.cat(y_prob_all, dim = 0).detach().cpu().numpy()
    if val:
        acc = correct_prediction/total_prediction
        y_true_all = torch.cat(y_true_all, dim = 0).detach().cpu().numpy()
        
        auc = roc_auc_score(y_true_all, y_prob_all, multi_class="ovr", average='weighted')
        return {"acc": acc, "loss": val_loss / size, "auc": auc}
    else:
        y_pred_all = torch.cat(y_pred_all, dim = 0).detach().cpu().numpy()
        return {"pred": y_pred_all, "prob": y_prob_all, "filenames": filenames}

def main():
    global time
    time = datetime.now().strftime('%m%d%H%M')

    global logging_dir
    logging_dir = os.path.join("logging", time)
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)

    logging.basicConfig(format='%(message)s', filemode='w', filename=os.path.join(logging_dir, 'logging.log'), encoding='utf-8', level=logging.INFO)
    logging.info("Configs:")
    logging.info(pformat(configs))
    logging.info("")

    train_audio_path = configs["path_config"]["train_audio_path"]
    metadata = pd.read_csv("./train/meta_train.csv")

    run_eda = configs["general_config"]["run_eda"]
    x_summary = explore_x(train_audio_path)
    y_summary = explore_y(metadata, run_eda)

    data = data_cleaning(train_audio_path, x_summary["waveform_shape"], metadata)

    audio_dataset = AudioDataset(train_audio_path, data["x_file_paths"], y=data["y"], label_name=configs["label_config"]["label_name"])

    data = data_split(audio_dataset, configs["data_config"])

    model = train(data, len(y_summary["label"]), configs["train_config"])

    dir_path = os.path.join("model", time)# + ".pt" 
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    torch.save(model.state_dict(), os.path.join(dir_path, "model.pt"))

    test_audio_path = configs["path_config"]["test_audio_path"]
    
    test_files = os.listdir(test_audio_path)
    test_ds = AudioDataset(test_audio_path, test_files)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)

    test_ret = validation_loop(test_dl, model, val=False)

    y_prob_all = test_ret["prob"]
    filenames = np.array([test_ret["filenames"]])

    col_name = ["Filename", "Barking", "Howling", "Crying", "COSmoke", "GlassBreaking", "Other"]

    sample = pd.read_csv(os.path.join("pred", "sample_submission.csv"))
    output = np.concatenate((filenames.T, y_prob_all), axis=1)
    output = pd.DataFrame(output, columns=col_name)
    output = output.append(sample.iloc[10000:], ignore_index=True)

    pred_dir = os.path.join("pred", time)
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    output.to_csv(os.path.join(pred_dir, "pred.csv"), index=False)

    # Load Model
    # model = AudioClassifier(6)
    # model.load_state_dict(torch.load("model/05292344.pt"))
    # model.eval()
    # output = pd.read_csv("prediction1.csv")
    # sample = pd.read_csv("sample_submission.csv")
    # output = output.append(sample.iloc[10000:], ignore_index=True)
    # output.to_csv("prediction.csv", index=False)


if __name__ == "__main__":
    main()
    # metadata = pd.read_csv("./train/meta_train.csv")
    # explore_y(metadata)
