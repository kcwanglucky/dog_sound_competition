path_config = {
    "train_audio_path": "./train/train",
    "test_audio_path": "./public_test",
}

label_config = {
    "label_name": "Label",   # ("Remark_label", "Label")
}

model_config = {
    "num_channel": 1,
    "conv1_channel": 16,
    "conv2_channel": 32,
    "conv3_channel": None,
    "conv4_channel": None,
}

train_config = {
    "num_epochs": 30,
    "optimizer": "adam",
    "lr": 0.005,
    "max_lr": 0.02,
    "show_plot": True,
    "model_config": model_config,
}

data_config = {
    "train_perc": 0.7,
    "train_batch_size": 32,
    "val_batch_size": 128,
    "test_batch_size": 128,
}

preprocess_config = {
    "n_mels": 64,
    "n_fft": 400,
}

general_config = {
    "run_eda": False,
    "score_criteria": "all_val_auc",
    "n_pass": 3,
    "do_test": False,
}

idx2label = {
    0: "Barking",
    1: "Howling",
    2: "Crying",
    3: "COSmoke",
    4: "GlassBreaking",
    5: "Other",
}

label2idx = {
    "Barking": 0,
    "Howling": 1,
    "Crying": 2,
    "COSmoke": 3,
    "GlassBreaking": 4,
    "Other": 5,
}

remark2idx = {
    'Barking': 0,
    'Howling': 1,
    'Crying': 2,
    'COSmoke': 3,
    'GlassBreaking': 4,
    'Vacuum': 5,
    'Blender': 6,
    'Electrics': 7,
    'Cat': 8,
    'Dishes': 9,
    'Other': 10,
}

configs = {
    "path_config": path_config,
    "label_config": label_config,
    "train_config": train_config,
    "data_config": data_config,
    "general_config": general_config,
    "model_config": model_config,
    "preprocess_config": preprocess_config,
}

def get_configs():
    cfgs = []
    
    cfg = configs.copy()
    cfgs.append(cfg)

    # cfg = configs.copy()
    # cfg["data_config"]["train_batch_size"] = 128
    # cfgs.append(cfg)

    # cfg = configs.copy()
    # cfg["train_config"]["lr"] = 0.01
    # cfg["train_config"]["max_lr"] = 0.1
    # cfgs.append(cfg)

    # cfg = configs.copy()
    # cfg["train_config"]["lr"] = 0.01
    # cfg["train_config"]["max_lr"] = 0.05
    # cfgs.append(cfg)

    # cfg = configs.copy()
    # cfg["model_config"]["conv4_channel"] = None
    # cfgs.append(cfg)

    # cfg = configs.copy()
    # cfg["model_config"]["conv1_channel"] = 16
    # cfg["model_config"]["conv4_channel"] = 32
    # cfg["model_config"]["conv4_channel"] = 64
    # cfg["model_config"]["conv4_channel"] = None
    # cfgs.append(cfg)

    # cfg = configs.copy()
    # cfg["model_config"]["conv1_channel"] = 16
    # cfg["model_config"]["conv4_channel"] = 32
    # cfg["model_config"]["conv4_channel"] = 64
    # cfg["model_config"]["conv4_channel"] = None
    # cfg["train_config"]["lr"] = 0.01
    # cfg["train_config"]["max_lr"] = 0.05
    # cfgs.append(cfg)

    return cfgs
