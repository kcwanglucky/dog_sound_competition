from audio_classifier import test, AudioClassifier, AudioClassifierCNNGRU
from config import configs
import torch
import os
from datetime import datetime

if __name__ == "__main__":
    time = datetime.now().strftime('%m%d%H%M')
    num_class = 6
    model_path = "06061807/model_0.pt"

    model = AudioClassifier(num_class, configs["model_config"])
    # model = AudioClassifierCNNGRU(2, 1, 32, num_class, 64)
    model.load_state_dict(torch.load(os.path.join("model", model_path)))
    model.eval()

    test(model, configs, time)
