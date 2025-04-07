import torch
import scipy.signal as signal

def DFTAnaRealEntireSignal(x_in, K, N, p):
    """
    K : FFT size, N : frame shift, p : filter
    """
    Lp = len(p)  # 필터 길이
    Lx = x_in.shape[1]  # 입력 신호 길이
    n_ch = x_in.shape[0] if x_in.ndim > 1 else 1  # 채널 수
    n_blocks = Lx//N
    device = x_in.device
    X_out = torch.zeros((n_ch, n_blocks, K//2 + 1), dtype=torch.cfloat, device = device)  # 복소수형 FFT 결과 저장

    for ch_ix in range(n_ch):
        x_tmp = x_in[ch_ix,:] if n_ch > 1 else x_in
        x_tmp = x_tmp.flatten()
        # 패딩 추가 (MATLAB의 `buffer([zeros(1,N-1) x_tmp], Lp, Lp-N);`과 동일)
        x_padded = torch.cat((torch.zeros(Lp - N,device = device), x_tmp))  # 앞쪽에 Lp-N 개의 0 추가

        # 슬라이딩 윈도우 적용 (MATLAB의 buffer() 대체)
        x_buffer = x_padded.unfold(0,Lp,N) #shape = (n_blocks, Lp)

        # 프레임을 역순으로 정렬 (MATLAB `Lp:-1:1`과 동일)
        x_buffer = torch.flip(x_buffer, dims=[1])  # 열 방향으로만 뒤집음 shape = (n_blocks, Lp)

        # 필터 적용 (브로드캐스팅을 위해 p를 (1, Lp)로 변환)
        p = p.to(device)
        p = p.reshape(1, -1)
        U = (x_buffer * p).reshape(n_blocks, Lp // K, K)  # 크기 변환

        V = torch.fft.fft(U.sum(dim=1), dim=1)

        X_out[ch_ix, :, :] = V[:, :K//2+1]

    return X_out


#inverse stft

def DFTSynRealEntireSignal(X_in, K, N, p):
    """
    Real-valued STFT Synthesis (GPU-compatible)
    X_in: [n_frames, K//2 + 1]
    K: FFT size
    N: hop size
    p: filter (1D tensor of length Lp)
    """
    device = X_in.device
    Lp = len(p)
    n_frames = X_in.shape[0]
    x = torch.zeros(N * n_frames, dtype=torch.float32, device=device)

    if torch.allclose(X_in, torch.zeros_like(X_in)):
        return x

    # Hermitian symmetry to get full FFT spectrum
    X_sym = torch.cat([X_in, torch.conj(X_in[:, 1:-1].flip(dims=[1]))], dim=1)

    Y = torch.fft.ifft(X_sym, dim=1).real  # IFFT to time domain

    # Repeat & apply filter
    tmp = torch.tile(Y, (Lp // K, 1)) * p.to(device).unsqueeze(0)  # shape: (n_frames, Lp)

    tdl = torch.zeros(Lp, dtype=torch.float32, device=device)
    Nzeros = torch.zeros(N, dtype=torch.float32, device=device)

    for i in range(n_frames):
        k = N * i
        tdl = torch.cat((Nzeros, tdl[:Lp - N])) + tmp[i]
        x[k:k+N] = (K * N) * tdl[-N:].flip(0)  # overlap-add

    return x