import pyroomacoustics as pra
import numpy as np
from typing import Tuple
from torch import Tensor
import torchaudio
import torch
import random
import os

class LibriSpeechLocations():
    '''
    Class of LibriSpeech recordings. Each recording is annotated with a speaker location.
    '''

    def __init__(self, source_locs, data_root, split):
        self.source_locs = source_locs
        self.data_root = data_root
        self.split = split

        self.filepaths = []
        self.metadata = []

        split_dir = os.path.join(data_root, split)
        for speaker_id in os.listdir(split_dir):
            speaker_dir = os.path.join(split_dir, speaker_id)
            for chapter_id in os.listdir(speaker_dir):
                chapter_dir = os.path.join(speaker_dir, chapter_id)
                for filename in os.listdir(chapter_dir):
                    if filename.endswith(".flac"):
                        full_path = os.path.join(chapter_dir, filename)
                        self.filepaths.append(full_path)
                        self.metadata.append(int(speaker_id))
        
        assert len(self.filepaths) == len(source_locs), \
            f"Number of source locations ({len(source_locs)}) does not match number of audio files ({len(self.filepaths)})"

    def __getitem__(self, index: int) -> Tuple[Tensor, int, int, float, int]:
        filepath = self.filepaths[index]
        speaker_id = self.metadata[index]
        source_loc = self.source_locs[index]
        seed = index

        waveform, sample_rate = torchaudio.load(filepath)
        return (waveform, sample_rate, speaker_id), source_loc, seed

    def __len__(self):
        return len(self.filepaths)


def one_random_delay(room_dim, fs, t60, mic_locs, signal, xyz_min, xyz_max, snr, anechoic=False):
    '''
    Simulate signal propagation using pyroomacoustics using random source location.
    '''

    if anechoic:
        e_absorption = 1.0
        max_order = 0
    else:
        e_absorption, max_order = pra.inverse_sabine(t60, room_dim)

    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(
        e_absorption), max_order=max_order)

    source_loc = np.random.uniform(low=xyz_min, high=xyz_max, size=(3))
    room.add_source(source_loc, signal=signal.squeeze())
    room.add_microphone(mic_locs)
    c = room.c
    d = np.sqrt(np.sum((mic_locs[:, 0] - source_loc)**2)) - \
        np.sqrt(np.sum((mic_locs[:, 1] - source_loc)**2))
    delay = d * fs / c
    room.simulate(reference_mic=0, snr=snr)
    x1 = room.mic_array.signals[0, :]
    x2 = room.mic_array.signals[1, :]

    return x1, x2, delay, room


def one_delay(room_dim, fs, t60, mic_locs, signal, source_loc, snr=1000, anechoic=False):
    '''
    Simulate signal propagation using pyroomacoustics for a given source location.
    '''

    if anechoic:
        e_absorption = 1.0
        max_order = 0
    else:
        e_absorption, max_order = pra.inverse_sabine(t60, room_dim)

    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(
        e_absorption), max_order=max_order)

    room.add_source(source_loc, signal=signal.squeeze())
    room.add_microphone(mic_locs)
    c = room.c
    d = np.sqrt(np.sum((mic_locs[:, 0] - source_loc)**2)) - \
        np.sqrt(np.sum((mic_locs[:, 1] - source_loc)**2))
    delay = d * fs / c
    room.simulate(reference_mic=0, snr=snr)
    x1 = room.mic_array.signals[0, :]
    x2 = room.mic_array.signals[1, :]

    return x1, x2, delay, room


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(
        batch, batch_first=True, padding_value=0.)
    return batch


class DelaySimulator(object):
    '''
    Given a batch of LibrispeechLocation samples, simulate signal
    propagation from source to the microphone locations.
    '''

    def __init__(self, room_dim, fs, N, t60, mic_locs, max_tau, anechoic, train=True, snr=1000, lower_bound=16000, upper_bound=48000):

        self.room_dim = room_dim
        self.fs = fs
        self.N = N
        self.mic_locs = mic_locs
        self.max_tau = max_tau
        self.snr = snr
        self.t60 = t60
        self.anechoic = anechoic
        self.train = train

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __call__(self, batch):
        # A data tuple has the form:
        # waveform, sample_rate, label, speaker_id, utterance_number

        tensors1, tensors2, targets = [], [], []

        # Gather in lists, and encode labels as indices
        with torch.no_grad():
            for (waveform, sample_rate, _), source_loc, seed in batch:

                waveform = waveform.squeeze()
                signal = waveform

                # use random seed for training, fixed for val/test
                # this controls the randomness in sound propagation when simulating the room
                if not self.train:
                    torch.manual_seed(seed)
                    random.seed(seed)
                    np.random.seed(seed)

                # sample random reverberation time and SNR
                this_t60 = np.random.uniform(low=self.t60[0], high=self.t60[1])
                this_snr = np.random.uniform(low=self.snr[0], high=self.snr[1])

                x1, x2, delay, _ = one_delay(room_dim=self.room_dim, fs=self.fs, t60=this_t60,
                                             mic_locs=self.mic_locs, signal=signal,
                                             source_loc=source_loc, snr=this_snr,
                                             anechoic=self.anechoic)

                if self.train:
                    start_idx = torch.randint(
                        self.lower_bound, self.upper_bound - self.N - 1, (1,))
                else:
                    start_idx = self.lower_bound

                end_idx = start_idx + self.N
                x1 = x1[start_idx:end_idx]
                x2 = x2[start_idx:end_idx]

                tensors1 += [torch.as_tensor(x1, dtype=torch.float)]
                tensors2 += [torch.as_tensor(x2, dtype=torch.float)]
                targets += [delay+self.max_tau]

        # Group the list of tensors into a batched tensor
        tensors1 = pad_sequence(tensors1).unsqueeze(1)
        tensors2 = pad_sequence(tensors2).unsqueeze(1)
        targets = torch.Tensor(targets)

        return tensors1, tensors2, targets