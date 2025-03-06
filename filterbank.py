import numpy as np
import scipy.signal as signal

def DFTAnaRealEntireSignal(x_in, K, N, p):
    """
    K : FFT size, N : frame shift, p : filter
    """
    Lp = len(p)  # 필터 길이
    Lx = x_in.shape[1]  # 입력 신호 길이
    n_ch = x_in.shape[0] if x_in.ndim > 1 else 1  # 채널 수
    n_blocks = int(np.floor(Lx / N))  # 블록 개수
    X_out = np.zeros((n_ch, n_blocks, K//2 + 1), dtype=np.complex64)  # 복소수형 FFT 결과 저장

    for ch_ix in range(n_ch):
        x_tmp = x_in[ch_ix,:] if n_ch > 1 else x_in
        x_tmp = x_tmp.flatten()
        # ✅ 패딩 추가 (MATLAB의 `buffer([zeros(1,N-1) x_tmp], Lp, Lp-N);`과 동일)
        x_padded = np.concatenate((np.zeros(Lp - N), x_tmp))  # 앞쪽에 Lp-N 개의 0 추가

        # ✅ 슬라이딩 윈도우 적용 (MATLAB의 buffer() 대체)
        x_buffer = np.lib.stride_tricks.sliding_window_view(x_padded, window_shape=(Lp,))[::N]
 
        # ✅ 프레임을 역순으로 정렬 (MATLAB `Lp:-1:1`과 동일)
        x_buffer = np.flip(x_buffer, axis=1)  # 열 방향으로만 뒤집음

        # ✅ 필터 적용 (브로드캐스팅을 위해 p를 (1, Lp)로 변환)
        p = p.reshape(1, -1)
        U = (x_buffer * p).reshape(n_blocks, Lp // K, K)  # 크기 변환

        # ✅ FFT 수행
        V = np.fft.fft(U.sum(axis=1), axis=1)

        X_out[ch_ix, :, :] = V[:, :K//2+1]

    return X_out


#inverse stft
import numpy as np

def DFTSynRealEntireSignal(X_in, K, N, p):
    Lp = len(p)  # 필터 길이
    x = np.zeros(N * X_in.shape[0], dtype=X_in.dtype)  # 출력 신호 초기화

    if np.all(X_in == 0):
        return x

    tdl = np.zeros(Lp, dtype=X_in.dtype)  # 지연선 필터 초기화

    # 역 FFT 수행
    X_sym = np.hstack([X_in, np.conj(X_in[:,-2:0:-1])])  # 대칭 복원
    Y = np.real(np.fft.ifft(X_sym, axis=1))  # 시간 도메인 변환

    # 윈도우 적용
    tmp = np.tile(Y.T, (Lp // K, 1)).T * p
    tdl = np.zeros(Lp, dtype = X_in.dtype)
    # 프레임별 합성
    Nzeros = np.zeros(N, dtype=X_in.dtype)

    for i in range(X_in.shape[0]):
        k = N * i  # MATLAB (i-1) 보정 불필요

        tdl = np.concatenate((Nzeros, tdl[:Lp-N])) + tmp[i,:]
        x[k:k+N] = (K * N) * tdl[-N:][::-1]  # 마지막 N개를 뒤집어 저장

    return x
