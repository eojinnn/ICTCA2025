import cfg_dereverb as cfgs
import librosa
import numpy as np
import filterbank
import lib
import estimator

# preparation
x1, fs_in = librosa.load('D:/datas/20250203_정보섬/20250203_ssu/PROJECT/PRJ000/AUDIO/MONO-000.wav')
x1 = librosa.resample(y = x1,orig_sr = fs_in, target_sr = cfgs.fs)
x2, fs_in = librosa.load('D:/datas/20250203_정보섬/20250203_ssu/PROJECT/PRJ000/AUDIO/MONO-003.wav')
x2 = librosa.resample(y = x2,orig_sr = fs_in, target_sr = cfgs.fs)

x = np.stack([x1, x2], axis = 0)
p = np.hanning(512)

''' Signal processing
% The algorithm itself is real-time capable, i.e., no processing of the entire
% utterance is necessary. Here however, for efficiency of the MATLAB implementation,
% the entire signal is processed at once.
'''

# analysis filterbank
X=filterbank.DFTAnaRealEntireSignal(x,cfgs.K,cfgs.N,p)

# estimate PSD and coherence
Pxx = lib.estimate_psd(X,cfgs.nr["lambda"])
Cxx = lib.estimate_cpsd(X[0,:,:],X[1,:,:],cfgs.nr["lambda"])/np.sqrt(Pxx[0,:,:]*Pxx[1,:,:])

frequency = np.linspace(0,cfgs.fs/2,int(cfgs.K/2+1)) # frequency axis
print('freq', frequency.shape)
# define coherence models
Css = np.exp(1j * 2 * np.pi * frequency * cfgs.TDOA)              # target signal coherence; not required for estimate_cdr_nodoa
Cnn = np.sinc(2 * frequency * cfgs.d_mic/cfgs.c) # diffuse noise coherence; not required for estimate_cdr_nodiffuse
Cnn = Cnn.reshape(1,-1)
print(Cxx.shape)
print(Css.shape)
print(Cnn.shape)

# apply CDR estimator (=SNR)
SNR = estimator.estimate_cdr_nodoa(Cxx, Cnn)
SNR = np.maximum(np.real(SNR),0)

print(SNR)
weights = lib.spectral_subtraction(SNR,cfgs.nr["alpha"],cfgs.nr["beta"],cfgs.nr["mu"])
weights = np.maximum(weights,cfgs.nr["floor"])
weights = np.minimum(weights,1)

# postfilter input is computed from averaged PSDs of both microphones
Postfilter_input = np.sqrt(np.mean(np.abs(X)**2,0))*np.exp(1j*np.angle(X[0,:,:]))

# apply postfilter
Processed = weights*Postfilter_input
print('processed', Processed.shape)
# synthesis filterbank
y = filterbank.DFTSynRealEntireSignal(Processed,cfgs.K,cfgs.N,p)

import time
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

# 타이머 시작
start_time = time.time()

# (여기에 신호 처리 코드 수행)

# 처리 완료 메시지
print(f"done ({time.time() - start_time:.2f}s)")

# 오디오 저장
y = np.real(y).astype(np.float32)
sf.write("D:/multi_source_localization/ICEIC-2025/out.wav", y, cfgs.fs)

# 시각화
plt.figure(figsize=(10, 6))

# Estimated CDR (=SNR) [dB]
plt.subplot(2, 1, 1)
plt.imshow(10 * np.log10(SNR), aspect='auto', origin='lower', cmap='jet')
plt.colorbar(label="dB")
plt.clim(-15, 15)
plt.title("Estimated CDR (=SNR) [dB]")
plt.xlabel("Frame Index")
plt.ylabel("Subband Index")

# Filter Gain
plt.subplot(2, 1, 2)
plt.imshow(weights, aspect='auto', origin='lower', cmap='jet')
plt.colorbar(label="Gain")
plt.clim(0, 1)
plt.title("Filter Gain")
plt.xlabel("Frame Index")
plt.ylabel("Subband Index")

plt.tight_layout()
plt.show()

