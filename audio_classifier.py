import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import logging

from torch.utils.data import DataLoader, dataset
from IPython.display import Audio, display
from torchsummary import summary

from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import random_split

from datetime import datetime

from utils import *
from model import AudioClassifier
from audio_util import AudioDataset, AudioProcessor
from config import configs, get_configs, remark2idx
from pprint import pformat

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

    return {"metadata": metadata, "Label": label, "Remark_label": remark}

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
    to_remove = set()  # collect samples whose shape is different from input `shape`
    count = 0

    other_files = [f + ".wav" for f in metadata[metadata["Remark"] == "Other"]["Filename"].values]

    for path in other_files:
        to_remove.add(path)
        count += 1

    for path in files:
        # print(os.path.join(TRAIN_AUDIO_PATH, path))
        waveform, sample_rate = AudioProcessor.read(os.path.join(path_to_train_dir, path))
        if waveform.shape != shape:
            to_remove.add(path)
            count += 1
            logging.info("Anomaly Shape Detected: {}\nShape: {}\n".format(path, waveform.shape))

        """ To remove Label 4, 5, 6 so that I could train a model for label 1, 2, 3"""
        #TODO: Make it flexible
        if metadata[metadata["Filename"] == path.split('.')[0]]["Label"].values[0] not in (1, 2):
            if path not in to_remove:
                to_remove.add(path)

    for file in to_remove:
        drop_idx = metadata[metadata["Filename"] == file.split('.')[0]].index
        metadata = metadata.drop(drop_idx)
        files.remove(file)

    #TODO: Make it flexible
    metadata = metadata.loc[metadata['Label'].isin([1, 2])]
    metadata["Label"] = metadata["Label"] - 1
    
    assert len(metadata) == len(files)
    logging.info("Total Anomaly: {}".format(count))
    logging.info("Before deleting anomaly: {} numbers of audios".format(num_files_bf))
    logging.info("After deleting anomaly: {} numbers of audios\n".format(len(files)))

    # assert len(files) == len(metadata)
    logging.info("Number of valid sample x: {}".format(len(files)))
    logging.info("Number of valid sample y: {}\n".format(len(metadata)))

    return {"x_file_paths": files, "y": metadata}

def to_dataloader(dataset, config):
    num_items = len(dataset)
    num_train = round(num_items * config["train_perc"])
    num_val = num_items - num_train
    train_ds, val_ds = random_split(dataset, [num_train, num_val])

    train_dataloader = DataLoader(train_ds, batch_size=config["train_batch_size"],
                        shuffle=True, num_workers=0)

    val_dataloader = DataLoader(val_ds, batch_size=config["val_batch_size"],
                        shuffle=False, num_workers=0)
    
    check_shape(train_dataloader)
    return {"dataset": dataset, "train_ds": train_ds, "val_ds": val_ds, "train_dl": train_dataloader, "val_dl": val_dataloader}

def check_shape(dataloader):
    for i_batch, sample_batched in enumerate(dataloader):
        logging.info("Data Shape")
        logging.info("x_size: {} \ty_size: {}\n".format(
            sample_batched['x'].size(), sample_batched['y'].size()))
        if i_batch == 1:
            break

def train(data, config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Using {} device\n'.format(device))

    num_class = len(data["dataset"].y[configs["label_config"]["label_name"]].unique())
    print(num_class)
    train_dl, val_dl = data["train_dl"], data["val_dl"]

    model_config = config["model_config"]
    model = AudioClassifier(num_class, model_config)
    device = torch.device(device)
    model = model.to(device)

    # Loss Function, Optimizer and Scheduler
    loss_fnc = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, config["max_lr"],
                                                steps_per_epoch=int(len(train_dl)),
                                                epochs=config["num_epochs"],
                                                anneal_strategy='linear')

    model, result = training_loop(train_dl, val_dl, model, loss_fnc, optimizer, scheduler, config["num_epochs"])

    return model, result

def training_loop(train_dl, val_dl, model, loss_fn, optimizer, scheduler, num_epochs):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_train_loss, all_train_acc, all_train_auc, all_val_loss, all_val_acc, all_val_auc = [], [],[], [], [], []

    # Repeat for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        y_true_all, y_prob_all = [], []

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            x, y_true = data['x'].to(device), data['y'].to(device).reshape(-1)

            # Normalize the inputs
            mean, std = x.mean(), x.std()
            x = (x - mean) / std

            # forward + backward + optimize
            outputs = model(x)
            y_prob = torch.nn.functional.softmax(outputs, dim=1)

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

                # logging.info("Epoch {} Batch {}:".format(epoch + 1, i + 1))
                # logging.info('\tLoss/Sample: %.3f' % (loss.item()))
            
            y_true_all.append(data['y'])
            y_prob_all.append(y_prob.cpu())

        # Print stats at the end of the epoch
        # num_batches = len(train_dl)
        # avg_loss = running_loss / num_batches
        avg_loss = running_loss / len(train_dl.dataset)
        acc = correct_prediction/total_prediction

        y_true_all = torch.cat(y_true_all, dim = 0).detach().cpu().numpy()
        y_prob_all = torch.cat(y_prob_all, dim = 0).detach().cpu().numpy()

        # auc = roc_auc_score(y_true_all, y_prob_all, multi_class="ovr", average='weighted')
        auc = roc_auc_score(y_true_all, y_prob_all[:,1], average='weighted')

        logging.info("---------Epoch {}---------".format(epoch + 1))
        logging.info("Train Stats:")
        logging.info(f'\tLoss: {avg_loss:.2f}, Accuracy: {acc:.2f}, Auc: {auc:.2f}')

        all_train_loss.append(avg_loss)
        all_train_acc.append(acc)
        all_train_auc.append(auc)

        if len(val_dl) > 0:
            val_ret = validation_loop(val_dl, model, epoch, val=True, loss_fn=loss_fn)

            logging.info("Validation Stats:")
            logging.info(f'\tLoss: {val_ret["loss"]:.2f}, Accuracy: {val_ret["acc"]:.2f}, Auc: {val_ret["auc"]:.2f}\n')

            all_val_loss.append(val_ret["loss"])
            all_val_acc.append(val_ret["acc"])
            all_val_auc.append(val_ret["auc"])
        
    logging.info('Finished Training\n')

    result = {"all_train_loss": all_train_loss, "all_train_acc": all_train_acc, "all_train_auc": all_train_auc, "all_val_loss": all_val_loss, "all_val_acc": all_val_acc, "all_val_auc": all_val_auc}

    return model, result

def plot_result(title, nums):
    plt.plot(np.arange(1, len(nums) + 1), nums)
    plt.title(title)
    plt.xlabel("epochs")
    plt.ylabel("val")
    plt.axhline(y=max(nums), color='r', linestyle='-', label="max={}".format(round(max(nums), 4)))
    plt.axhline(y=min(nums), color='g', linestyle='-', label="min={}".format(round(min(nums), 4)))
    plt.legend()

    plt.savefig(os.path.join(logging_dir, title + ".jpg"))
    plt.close()


def plot(result):
    all_train_loss, all_train_acc, all_train_auc, all_val_loss, all_val_acc, all_val_auc = \
        result["all_train_loss"], result["all_train_acc"], result["all_train_auc"], result["all_val_loss"], result["all_val_acc"], result["all_val_auc"]

    logging.info("Overall Result across epochs: ")
    logging.info("Train: ")
    logging.info("\ttrain_loss: {}".format(",".join(map(str, all_train_loss))))
    logging.info("\ttrain_acc: {}".format(",".join(map(str, all_train_acc))))
    logging.info("\ttrain_auc: {}".format(",".join(map(str, all_train_auc))))

    logging.info("Val: ")
    logging.info("\tval_loss: {}".format(",".join(map(str, all_val_loss))))
    logging.info("\tval_acc: {}".format(",".join(map(str, all_val_acc))))
    logging.info("\tval_auc: {}".format(",".join(map(str, all_val_auc))))

    all_ret = [all_train_loss, all_train_acc, all_train_auc, all_val_loss, all_val_acc, all_val_auc]
    titles = ["all_train_loss", "all_train_acc", "all_train_auc", "all_val_loss", "all_val_acc", "all_val_auc"]
    for title, ret in zip(titles, all_ret):
        plot_result(title, ret)


def validation_loop(val_dl, model, epoch=0, val=True, loss_fn=None):
    if val:
        print("Validation starts")
    else:
        print("Test starts")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    size = len(val_dl.dataset)

    val_loss = correct_prediction = total_prediction = 0
    y_true_all, y_pred_all, y_prob_all = [], [], []
    filenames = []

    with torch.no_grad(): 
        for i, data in enumerate(val_dl):
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
            y_pred_all.append(y_pred.cpu())

            if val:
                val_loss += loss_fn(outputs, y_true).item()

                # Count of predictions that matched the target label
                correct_prediction += (y_pred == y_true).sum().item()
                total_prediction += y_pred.shape[0]
                y_true_all.append(y_true.cpu())
    
    # oh_ytrue = torch.nn.functional.one_hot(y_true)
    # y_prob = torch.nn.functional.softmax(outputs, dim=1).detach().numpy()
    # auc = roc_auc_score(oh_ytrue, y_prob, multi_class="ovr", average='weighted')

    # y_prob = torch.nn.functional.softmax(outputs, dim=1)
    y_prob_all = torch.cat(y_prob_all, dim = 0).detach().cpu().numpy()
    y_pred_all = torch.cat(y_pred_all, dim = 0).detach().cpu().numpy()

    if val:
        acc = correct_prediction/total_prediction
        y_true_all = torch.cat(y_true_all, dim = 0).detach().cpu().numpy()
        
        # auc = roc_auc_score(y_true_all, y_prob_all, multi_class="ovr", average='weighted')
        auc = roc_auc_score(y_true_all, y_prob_all[:,1], average='weighted')

        logging.info("Confusion Matrix: ")
        print("Confusion Matrix: ")

        cm = confusion_matrix(y_true_all, y_pred_all)
        print("{}".format(cm))
        logging.info("{}".format(cm))

        if epoch > 12:
            cm_summary(cm)

        print("")
        logging.info("")

        return {"acc": acc, "loss": val_loss / size, "auc": auc}
    else:
        return {"pred": y_pred_all, "prob": y_prob_all, "filenames": filenames}

def cm_summary(cm):
    for i, row in enumerate(cm):
        print("Label {}".format(i))
        print("Accuracy: {}".format(round(row[i] / np.sum(row), 4)))
        print("Most likely confused with label {}\n".format(row.argsort()[-2]))
        
        logging.info("Label {}".format(i))
        logging.info("Accuracy: {}".format(round(row[i] / np.sum(row), 4)))
        logging.info("Most likely confused with label {}\n".format(row.argsort()[-2]))

def main():
    all_configs = get_configs()

    for i, configs in enumerate(all_configs):
        print("{} config".format(i + 1))
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
        
        metadata["Remark_label"] = list(map(lambda x: remark2idx[x], metadata["Remark"]))

        run_eda = configs["general_config"]["run_eda"]
        x_summary = explore_x(train_audio_path)
        y_summary = explore_y(metadata, run_eda)

        data = data_cleaning(train_audio_path, x_summary["waveform_shape"], metadata)


        general_config = configs["general_config"]
        scores = []

        for pass_id in range(general_config["n_pass"]):
            logging.info("--------------Pass {}--------------".format(pass_id + 1))
            print("--------------Pass {}--------------".format(pass_id + 1))
            audio_dataset = AudioDataset(train_audio_path, data["x_file_paths"], configs["preprocess_config"], y=data["y"], label_name=configs["label_config"]["label_name"])

            data_dl = to_dataloader(audio_dataset, configs["data_config"])

            model, result = train(data_dl, configs["train_config"])

            if configs["data_config"]["train_perc"] < 1:
                if configs["train_config"]["show_plot"]:
                    plot(result)
                
                scores.append(result["all_val_auc"][-1])

        criteria = general_config["score_criteria"]

        if configs["data_config"]["train_perc"] < 1:
            logging.info("Scoring criteria: {}".format(criteria))
            logging.info("\t{}".format(scores))

            final_score = round(sum(scores) / len(scores), 4)
            logging.info("Final Score: {}".format(final_score))

        summary(model, (configs["model_config"]["num_channel"], 64, 79))

        dir_path = os.path.join("model", time) 
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        torch.save(model.state_dict(), os.path.join(dir_path, "model.pt"))

        if configs["general_config"]["do_test"]:
            test_audio_path = configs["path_config"]["test_audio_path"]
            test_files = os.listdir(test_audio_path)
            test_ds = AudioDataset(test_audio_path, test_files, configs["preprocess_config"])
            test_dl = DataLoader(test_ds, batch_size=configs["data_config"]["test_batch_size"], shuffle=False, num_workers=0)

            test_ret = validation_loop(test_dl, model, val=False)

            y_prob_all = test_ret["prob"]
            filenames = np.array([test_ret["filenames"]])

            col_name = ["Filename", "Barking", "Howling", "Crying", "COSmoke", "GlassBreaking", "Other"]

            sample = pd.read_csv(os.path.join("pred", "sample_submission.csv"))
            output = np.concatenate((filenames.T, y_prob_all), axis=1)
            output = pd.DataFrame(output, columns=col_name)
            output = output.sort_values(by=["Filename"])
            output = output.append(sample.iloc[10000:], ignore_index=True)

            pred_dir = os.path.join("pred", time)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            output.to_csv(os.path.join(pred_dir, "pred.csv"), index=False)

    # Load Model
    # model = AudioClassifier(6, config["model_config"])
    # model.load_state_dict(torch.load("model/05292344.pt"))
    # model.eval()
    # output = pd.read_csv("prediction1.csv")
    # sample = pd.read_csv("sample_submission.csv")
    # output = output.append(sample.iloc[10000:], ignore_index=True)
    # output.to_csv("prediction.csv", index=False)


if __name__ == "__main__":
    main()
    # output = pd.read_csv("pred/05302216/pred.csv")
    # output = output.iloc[:10000].sort_values(by=["Filename"])

    # sample = pd.read_csv(os.path.join("pred", "sample_submission.csv"))
    # output = output.append(sample.iloc[10000:], ignore_index=True)
    # output = output.to_csv("prediction.csv", index=False)

    # output = pd.read_csv("prediction1.csv")
    # sample = pd.read_csv("sample_submission.csv")
    # output = output.append(sample.iloc[10000:], ignore_index=True)
    # output.to_csv("prediction.csv", index=False)
