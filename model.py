import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.fft

class GCC_freq(nn.Module):
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

    def forward(self, X1, X2): #X1 shape (batch:32, block:16, freq bins: 257)
        n = (X1.shape[-1]-1)*2
        # Generalized Cross Correlation Phase Transform
        Gxy = X1 * torch.conj(X2)

        if self.filt == 'phat':
            phi = 1 / (torch.abs(Gxy) + self.epsilon)
        
        elif self.filt == 'roth':
            phi = 1 / (X1 * torch.conj(X1) + self.epsilon)

        elif self.filt == 'scot':
            Gxx = X1 * torch.conj(X1)
            Gyy = X2 * torch.conj(X2)
            phi = 1 / (torch.sqrt(Gxx * Gyy) + self.epsilon)

        elif self.filt == 'ht':
            Gxx = X1 * torch.conj(X1)
            Gyy = X2 * torch.conj(X2)
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

            cc = torch.stack(cc, dim=1)

        else:
            cc = torch.fft.irfft(Gxy * phi, n)

        max_shift = int(n / 2)

        if self.max_tau:
            max_shift = min(self.max_tau, int(max_shift))

        if self.dim == 2:
            cc = torch.cat((cc[:,:, -max_shift:], cc[:,:, :max_shift+1]), dim=-1)
        elif self.dim == 3:
            cc = torch.cat(
                (cc[:, :, :, -max_shift:], cc[:, :, :, :max_shift+1]), dim=-1)

        return cc

class GCC_time(nn.Module):
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
        if x.shape[-1] == 0:
            raise ValueError("Input has zero length!")
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
            phi = 1 / (torch.sqrt(Gxx * Gyy) + self.epsilon)

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


class PGCCPHAT(nn.Module):
    def __init__(self, beta=np.arange(0, 1.1, 0.1), max_tau=42, head='regression', input_shape = 'freq'):
        super().__init__()

        '''
        Implementation of CNN-Based Parametrized GCC-PHAT by Salvati et al.
        https://www.isca-speech.org/archive/pdfs/interspeech_2021/salvati21_interspeech.pdf
        '''
        self.mode = input_shape
        self.beta = beta
        if self.mode == 'freq':
            self.gcc = GCC_freq(max_tau=max_tau, dim=3, filt='phat', beta=beta)
            self.num_class = 11
        else:
            self.gcc = GCC_time(max_tau=max_tau, dim=3, filt='phat', beta=beta)
            self.num_class = 1
        self.head = head
        self.max_tau = max_tau

        if head == 'regression':
            n_out = 1
        else:
            n_out = 2 * self.max_tau + 1

        self.conv1 = nn.Conv2d(self.num_class, 32, kernel_size=(3, 3))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3))
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=(3, 3))
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=(3, 3))
        self.bn5 = nn.BatchNorm2d(512)

        self.mlp = nn.Sequential(
            nn.LazyLinear(512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_out)
        )

    def forward(self, X1, X2):

        batch_size = X1.shape[0]
        
        x = self.gcc(X1, X2) # x shape: (batch, beta:11, block:16, delay:1024)
        if self.mode != "freq":
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = F.relu(self.bn1(x)) # output shape: (batch, 32, 14, )

        x = self.conv2(x)
        x = F.relu(self.bn2(x)) # output shape: (batch, 64, 12, )

        x = self.conv3(x)
        x = F.relu(self.bn3(x)) # output shape: (batch, 128, 10, )

        x = self.conv4(x)
        x = F.relu(self.bn4(x)) # output shape: (batch, 256, 8, )

        x = self.conv5(x) 
        x = F.relu(self.bn5(x)) # output shape: (batch, 512, 6, 1014)

        x = self.mlp(x.reshape([batch_size, -1])).squeeze() # batch, 512x6x1014

        return x    