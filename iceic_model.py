import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.fft

class GCC(nn.Module):
    def __init__(self, max_tau=None, dim=2, filt='phat', epsilon=0.001, beta=None):
        super().__init__()

        ''' GCC implementation based on Knapp and Carter,
        "The Generalized Correlation Method for Estimation of Time Delay",
        IEEE Trans. Acoust., Speech, Signal Processing, August, 1976 '''

        self.max_tau = max_tau
        self.dim = dim
        self.filt = filt
        self.epsilon = epsilon
        self.beta = beta

    def forward(self, x, y):

        n = x.shape[-1] + y.shape[-1]

        # Generalized Cross Correlation Phase Transform
        X = torch.fft.rfft(x, n=n)
        Y = torch.fft.rfft(y, n=n)
        Gxy = X * torch.conj(Y)

        if self.filt == 'phat':
            phi = 1 / (torch.abs(Gxy) + self.epsilon)

        elif self.filt == 'roth':
            phi = 1 / (X * torch.conj(X) + self.epsilon)

        elif self.filt == 'scot':
            Gxx = X * torch.conj(X)
            Gyy = Y * torch.conj(Y)
            phi = 1 / (torch.sqrt(X * Y) + self.epsilon)

        elif self.filt == 'ht': 
            Gxx = X * torch.conj(X)
            Gyy = Y * torch.conj(Y)
            gamma = Gxy / torch.sqrt(Gxx * Gxy)
            phi = torch.abs(gamma)**2 / (torch.abs(Gxy)
                                         * (1 - gamma)**2 + self.epsilon)

        elif self.filt == 'cc':
            phi = 1.0

        else:
            raise ValueError('Unsupported filter function')

        if self.beta is not None:
            cc = []
            for i in range(self.beta.shape[0]):
                cc.append(torch.fft.irfft(
                    Gxy * torch.pow(phi, self.beta[i]), n))

            cc = torch.cat(cc, dim=1)

        else:
            cc = torch.fft.irfft(Gxy * phi, n)

        max_shift = int(n / 2)
        if self.max_tau:
            max_shift = np.minimum(self.max_tau, int(max_shift))

        if self.dim == 2:
            cc = torch.cat((cc[:, -max_shift:], cc[:, :max_shift+1]), dim=-1)
        elif self.dim == 3:
            cc = torch.cat(
                (cc[:, :, -max_shift:], cc[:, :, :max_shift+1]), dim=-1)

        return cc

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

class DiffusenessMask(nn.Module):
    def __init__(self, threshold=0.2, n_fft=512, hop_length=256):
        super(DiffusenessMask, self).__init__()
        self.d = 0.5
        self.c = 343.0
        self.threshold = threshold
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, x1, x2):
        # STFT to get time-frequency representation
        x1 = x1.squeeze(1)
        x2 = x2.squeeze(1)
        stft_x1 = torch.stft(x1, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)
        stft_x2 = torch.stft(x2, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=True)

        # Compute cross-power spectral density (CPSD)
        cross_power = stft_x1 * torch.conj(stft_x2)  # (batch_size, num_frequencies, num_time_frames)
        auto_power_x1 = stft_x1 * torch.conj(stft_x1)  # (batch_size, num_frequencies, num_time_frames)
        auto_power_x2 = stft_x2 * torch.conj(stft_x2)  # (batch_size, num_frequencies, num_time_frames)

        # Get the frequency bins from STFT result
        num_freq_bins = stft_x1.shape[1]
        fs = 2 * (num_freq_bins - 1)  # Sampling frequency estimate based on the number of frequency bins
        f_k = torch.linspace(0, fs / 2, num_freq_bins, device=stft_x1.device)

        # Noise coherence calculation based on equation (5)
        noise_coherence = torch.sin(2 * np.pi * f_k * self.d / self.c) / (2 * np.pi * f_k * self.d / self.c + 1e-10)
        noise_coherence = torch.abs(noise_coherence)
        noise_coherence = noise_coherence.unsqueeze(0).unsqueeze(-1)

        # Spatial coherence function
        spatial_coherence = cross_power / (torch.sqrt(auto_power_x1 * auto_power_x2) + 1e-10)

        # Calculate components for CDR estimation
        real_spatial_coherence = torch.real(spatial_coherence)  # \(\Re\{ \hat{\Gamma}_X(m, k) \}\)
        real_spatial_coherence_sq = real_spatial_coherence ** 2  # \(\Re\{\hat{\Gamma}_X\}^2\)
        abs_spatial_coherence_sq = torch.abs(spatial_coherence) ** 2  # \(|\hat{\Gamma}_X(m, k)|^2\)

        noise_coherence_sq = noise_coherence ** 2  # \(\tilde{\Gamma}_N^2\)

        # correction term 계산: 수식 (7)에 해당하는 부분
        correction_term = torch.sqrt(
            noise_coherence_sq * real_spatial_coherence_sq
            - noise_coherence_sq * abs_spatial_coherence_sq
            + noise_coherence_sq
            - 2 * noise_coherence * real_spatial_coherence
            + abs_spatial_coherence_sq
        )

        # CDR estimation using equation (6)
        numerator = noise_coherence * real_spatial_coherence - abs_spatial_coherence_sq - correction_term
        denominator = abs_spatial_coherence_sq - 1
        cdr = numerator / (denominator + 1e-10)  # Avoid division by zero

        # Diffuseness estimation
        diffuseness = 1 / (cdr + 1)
        
        # Binary mask with a rigorous threshold
        mask = (diffuseness < self.threshold).float()  # (batch_size, num_frequencies, num_time_frames)

        # Adjust mask shape to match STFT result shape if necessary
        if mask.dim() < stft_x1.dim():
            mask = mask.unsqueeze(-1)

        # Apply mask in the STFT domain
        masked_stft_x1 = stft_x1 * mask
        masked_stft_x2 = stft_x2 * mask

        # Inverse STFT to return to time domain
        x1_filtered = torch.istft(masked_stft_x1, n_fft=self.n_fft, hop_length=self.hop_length)
        x2_filtered = torch.istft(masked_stft_x2, n_fft=self.n_fft, hop_length=self.hop_length)

        return x1_filtered, x2_filtered



class mPGCCPHAT(nn.Module):
    def __init__(self, beta=np.arange(0, 1.1, 0.1), max_tau=42, head='regression'):
        super().__init__()

        '''
        Implementation of CNN-Based Parametrized GCC-PHAT by Salvati et al.
        https://www.isca-speech.org/archive/pdfs/interspeech_2021/salvati21_interspeech.pdf
        '''

        self.beta = beta
        self.gcc = GCC(max_tau=max_tau, dim=3, filt='phat', beta=beta)
        self.head = head
        self.max_tau = max_tau

        if head == 'regression':
            n_out = 1
        else:
            n_out = 2 * self.max_tau + 1

        # Diffuseness Mask Layer
        self.diffuseness_mask = DiffusenessMask()

        # CNN Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3))
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3))
        self.bn5 = nn.BatchNorm2d(512)

        # MLP Layers
        self.mlp = nn.Sequential(
            nn.Linear(512 * (2 * max_tau + 1 - 10), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_out)
        )

    def forward(self, x1, x2):
        batch_size = x1.shape[0]

        # Apply diffuseness mask to both input signals
        x1, x2 = self.diffuseness_mask(x1, x2)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)

        # Compute GCC-PHAT between the masked signals
        x = self.gcc(x1, x2).unsqueeze(1)
       
        # Pass through CNN layers
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
     
        x = self.conv2(x)
        x = F.relu(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = self.conv4(x)
        x = F.relu(self.bn4(x))

        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        
        # Flatten and pass through MLP layers
        x = self.mlp(x.reshape([batch_size, -1])).squeeze()
        
        return x