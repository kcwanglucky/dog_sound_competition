import os
import numpy as np

import torchaudio
from torchaudio import transforms
from torch.utils.data import Dataset

class AudioProcessor:
    """ Preprocess audio files """
    def read(audio_file):
        """Load an audio file. Return the signal as a tensor and the sample rate"""
        waveform, sample_rate = torchaudio.load(audio_file)
        return (waveform, sample_rate) #if waveform.shape == self.shape else None
    
    def spectro_gram(waveform, sample_rate, n_mels=64, n_fft=512, hop_len=None):
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(waveform)
        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)

        # mfcc_transform = transforms.MFCC(sample_rate, n_mfcc=n_mels)
        # mfcc = mfcc_transform(waveform)

        # concat = torch.cat((spec, mfcc), dim=0)

        return spec

    def time_shift(waveform, sample_rate, shift_limit):
        sig_len = waveform.shape[1]

        shift_amt = int(np.random.rand() * shift_limit * sig_len)
        return waveform.roll(shift_amt), sample_rate

class AudioDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dir_name, filenames, config, y=None, label_name=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dir_name = dir_name
        self.filenames = filenames
        self.config = config
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
        specgram = AudioProcessor.spectro_gram(waveform, sample_rate, n_mels=self.config["n_mels"], n_fft=self.config["n_fft"], hop_len=None)
        
        if not self.is_test:        # Train or val set
            y = self.y[self.y["Filename"] == filename.split('.')[0]][self.label_name].to_numpy()
            sample = {"x": specgram, "y": y}
            return sample
        else:                       # Test set
            return {"x": specgram, "filename": filename.split('.')[0]}