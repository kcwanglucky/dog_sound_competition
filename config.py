path_config = {
    "train_audio_path": "./train/train",
    "test_audio_path": "./public_test",
}

label_config = {
    "label_name": "Label",
}

train_config = {
    "num_epochs": 20,
    "optimizer": "adam",
    "lr": 0.01,
    "max_lr": 0.1,
    "show_plot": True,
}

data_config = {
    "train_perc": 0.75,
    "train_batch_size": 32,
    "val_batch_size": 128,
}

general_config = {
    "run_eda": False,

}

label_mapping = {
    0: "Barking",
    1: "Howling",
    2: "Crying",
    3: "COSmoke",
    4: "GlassBreaking",
    5: "Other",
}

configs = {
    "path_config": path_config,
    "label_config": label_config,
    "train_config": train_config,
    "data_config": data_config,
    "general_config": general_config,
}
