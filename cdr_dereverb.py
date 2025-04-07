import cfg_dereverb as cfgs
import librosa
import filterbank
import lib
import torch
import estimator

def cdr_robust(x1,x2):
    batch_size = x1.shape[0]
    mic1_list = []
    mic2_list = []
    device = x1.device
    for i in range(batch_size):
        x = torch.stack([x1[i], x2[i]], axis = 0)
        p = torch.hann_window(512, periodic = True, device = device)

        ''' Signal processing
        % The algorithm itself is real-time capable, i.e., no processing of the entire
        % utterance is necessary. Here however, for efficiency of the MATLAB implementation,
        % the entire signal is processed at once.
        '''

        # analysis filterbank
        X=filterbank.DFTAnaRealEntireSignal(x,cfgs.K,cfgs.N,p)

        # estimate PSD and coherence
        Pxx = lib.estimate_psd(X,cfgs.nr["lambda"])
        Cxx = lib.estimate_cpsd(X[0,:,:],X[1,:,:],cfgs.nr["lambda"])/torch.sqrt(Pxx[0,:,:]*Pxx[1,:,:])

        frequency = torch.linspace(0,cfgs.fs/2,int(cfgs.K/2+1), device = device) # frequency axis

        # define coherence models
        Css = torch.exp(1j * 2 * torch.pi * frequency * cfgs.TDOA)              # target signal coherence; not required for estimate_cdr_nodoa
        Cnn = torch.sinc(2 * frequency * cfgs.d_mic/cfgs.c).to(device) # diffuse noise coherence; not required for estimate_cdr_nodiffuse
        Cnn = Cnn.reshape(1,-1)

        # apply CDR estimator (=SNR)
        SNR = estimator.estimate_cdr_nodoa(Cxx, Cnn)
        SNR = torch.clamp(torch.real(SNR),min = 0.0)

        weights = lib.spectral_subtraction(SNR,cfgs.nr["alpha"],cfgs.nr["beta"],cfgs.nr["mu"])
        weights = torch.clamp(weights,min = cfgs.nr["floor"])
        weights = torch.clamp(weights,max = 1.0)

        # postfilter input is computed from averaged PSDs of both microphones
        Postfilter_mic1 = torch.sqrt(torch.mean(torch.abs(X)**2,0))*torch.exp(1j*torch.angle(X[0,:,:]))
        Postfilter_mic2 = torch.sqrt(torch.mean(torch.abs(X)**2,0))*torch.exp(1j*torch.angle(X[1,:,:]))

        # apply postfilter
        Processed_mic1 = weights*Postfilter_mic1 #(block수, 주파수 길이)
        Processed_mic2 = weights*Postfilter_mic2

        mic1_list.append(Processed_mic1.unsqueeze(0))  # (1, frames, freqs)
        mic2_list.append(Processed_mic2.unsqueeze(0))
    
    return torch.cat(mic1_list, dim = 0), torch.cat(mic2_list, dim = 0)

# import time
# import soundfile as sf
# import numpy as np
# import matplotlib.pyplot as plt

# # 타이머 시작
# start_time = time.time()

# # (여기에 신호 처리 코드 수행)

# # 처리 완료 메시지
# print(f"done ({time.time() - start_time:.2f}s)")

# # 오디오 저장
# y = np.real(y).astype(np.float32)
# sf.write("D:/multi_source_localization/ICEIC-2025/out.wav", y, cfgs.fs)

# # 시각화
# plt.figure(figsize=(10, 6))

# # Estimated CDR (=SNR) [dB]
# plt.subplot(2, 1, 1)
# plt.imshow(10 * np.log10(SNR), aspect='auto', origin='lower', cmap='jet')
# plt.colorbar(label="dB")
# plt.clim(-15, 15)
# plt.title("Estimated CDR (=SNR) [dB]")
# plt.xlabel("Frame Index")
# plt.ylabel("Subband Index")

# # Filter Gain
# plt.subplot(2, 1, 2)
# plt.imshow(weights, aspect='auto', origin='lower', cmap='jet')
# plt.colorbar(label="Gain")
# plt.clim(0, 1)
# plt.title("Filter Gain")
# plt.xlabel("Frame Index")
# plt.ylabel("Subband Index")

# plt.tight_layout()
# plt.show()

