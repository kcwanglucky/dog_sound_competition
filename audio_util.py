import os
import torch
import numpy as np

import torchaudio
from torchaudio import transforms
import librosa
import librosa.display
from torch.utils.data import Dataset

class AudioProcessor:
    """ Preprocess audio files """
    def read(audio_file):
        """Load an audio file. Return the signal as a tensor and the sample rate"""
        # librosa
        # waveform, sample_rate = librosa.load(audio_file, sr=None)

        # torchaudio
        waveform, sample_rate = torchaudio.load(audio_file)
    
        return waveform.numpy().flatten(), sample_rate #if waveform.shape == self.shape else None
    
    def spectro_gram(waveform, sample_rate, n_mels=64, n_fft=512, hop_len=None):
        "transform waveform to melspectrogram, then change it to DB scale"
        top_db = 80

        # melspec has shape [channel, n_mels, time], where channel is mono, stereo etc

        # torchaudio
        waveform = torch.tensor(waveform, dtype=torch.float)
        melspec = transforms.MelSpectrogram(sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(waveform)
        melspec = transforms.AmplitudeToDB(top_db=top_db)(melspec)

        # librosa
        # melspec = librosa.feature.melspectrogram(waveform, sample_rate, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)
        # # Convert to decibels
        # melspec = librosa.amplitude_to_db(melspec, top_db=top_db)
        # melspec = np.expand_dims(melspec, 0)

        # mfcc_transform = transforms.MFCC(sample_rate, n_mfcc=n_mels)
        # mfcc = mfcc_transform(waveform)

        # concat = torch.cat((spec, mfcc), dim=0) 

        return melspec

    def spectro_augment(melspec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        "data augmentation based on spectrogram, including frequency masking and time masking"
        _, n_mels, n_steps = melspec.shape
        mask_value = melspec.mean()
        aug_spec = melspec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec

    "Data augmentation techniques"
    def add_white_noise(waveform, sample_rate, noise_factor):
        waveform = waveform + noise_factor * np.random.normal(0, 1, len(waveform))
        return waveform, sample_rate

    def time_shift(waveform, sample_rate, shift_limit):
        sig_len = len(waveform)

        shift_amt = int(np.random.rand() * shift_limit * sig_len)
        return np.roll(waveform, shift_amt), sample_rate

    def time_stretch(waveform, sample_rate):
        input_length = len(waveform)
        
        streching = waveform.copy()
        stretch_factor = np.random.uniform(0.8, 1.2)
        streching = librosa.effects.time_stretch(streching.flatten(), stretch_factor)

        if len(streching) > input_length:
            streching = streching[:input_length]
        else:
            streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")
            
        return streching, sample_rate


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

        if not self.is_test:
            # Only do this on training set
            waveform, sample_rate = AudioProcessor.add_white_noise(waveform, sample_rate, 0.1)
            waveform, sample_rate = AudioProcessor.time_shift(waveform, sample_rate, 0.1)
            # waveform, sample_rate = AudioProcessor.time_stretch(waveform, sample_rate)

        specgram = AudioProcessor.spectro_gram(np.array([waveform]), sample_rate, n_mels=self.config["n_mels"], n_fft=self.config["n_fft"], hop_len=None)
        # specgram = AudioProcessor.spectro_augment(specgram, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1)
        
        if not self.is_test:        # Train or val set
            y = self.y[self.y["Filename"] == filename.split('.')[0]][self.label_name].to_numpy()
            sample = {"x": specgram, "y": y}
            return sample
        else:                       # Test set
            return {"x": specgram, "filename": filename.split('.')[0]}
