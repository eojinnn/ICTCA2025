import torch

#estimate cross power spectral density
def estimate_cpsd(X1, X2, lambda_):
    """
    matlab의 filter(1-lambda, [1 -lambda], X1.*conj(X2), [], 2)를 python으로 변환.
    X1, X2 : 입력 STFT 신호
    lambda_ : 지수 이동 평균 필터 계수
    """
    Sx1x2 = torch.zeros_like(X1, dtype=torch.cfloat)  # 초기화
    Sx1x2[0, :] = (1 - lambda_) * (X1[0, :] * torch.conj(X2[0, :]))  # 첫 프레임 처리

    for t in range(1, X1.shape[0]):  # 시간 축(프레임) 반복
        Sx1x2[t, :] = (1 - lambda_) * (X1[t, :] * torch.conj(X2[t, :])) + lambda_ * Sx1x2[t-1, :]

    return Sx1x2

#estimate power spectral density
def estimate_psd(X, lambda_):
    """
    MATLAB의 filter(1-lambda, [1 -lambda], abs(X).^2, [], 2)를 Python으로 변환.
    """
    Sxx = torch.zeros_like(X, dtype=torch.float32)  # 결과 배열 초기화
    Sxx[:, 0, :] = (1 - lambda_) * torch.abs(X[:, 0, :]) ** 2  # 첫 프레임 초기값 설정
    
    for t in range(1, X.shape[1]):  # 시간 축을 따라 반복
        Sxx[:, t, :] = (1 - lambda_) * torch.abs(X[:, t, :]) ** 2 + lambda_ * Sxx[:, t - 1, :]

    return Sxx

#estimate spectral subtraction
def spectral_subtraction(SNR, alpha=2, beta=0.5, mu=1, Gmin=0.1):
    """
    Perform spectral subtraction-based weighting.

    Parameters:
    - SNR: numpy array or scalar, signal-to-noise ratio
    - alpha: exponent parameter (default: 2)
    - beta: exponent parameter (default: 0.5)
    - mu: scaling factor (default: 1)
    - Gmin: minimum gain value (default: 0.1)

    Returns:
    - weights: numpy array or scalar, computed spectral subtraction weights
    """
    SNR = torch.clamp(SNR, min = 0)  # Ensure SNR is non-negative
    weights = torch.clamp(1 - (mu / (SNR + 1)) ** beta, min = 0) ** alpha
    weights = torch.clamp(weights, min = 0)
    weights = torch.clamp(torch.sqrt(weights), min = Gmin)

    return weights
