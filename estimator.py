import numpy as np

def estimate_cdr_robust_unbiased(Cxx, Cnn, Css):
    """
    Coherent-to-Diffuse Ratio (CDR) 계산 함수 (MATLAB 코드 변환)
    Cxx: 관찰된 신호의 크로스 PSD (Cross Power Spectral Density)
    Cnn: 확산 잡음의 PSD (Diffuse Noise PSD)
    Css: 목표 신호(직접 도달 음)의 PSD
    """
    # `Css`와 `Cnn`을 `Cxx` 크기로 브로드캐스트
    Css = np.broadcast_to(Css, Cxx.shape)
    Cnn = np.broadcast_to(Cnn, Cxx.shape)

    # 수치적 안정성을 위해 `Cxx` 크기 제한
    magnitude_threshold = 1 - 1e-10
    Cxx = np.where(np.abs(Cxx) > magnitude_threshold,
                   magnitude_threshold * Cxx / np.abs(Cxx),
                   Cxx)

    # CDR 계산
    theta_Css = np.angle(Css)  # Css의 위상

    CDR = 1.0 / (-np.abs(Cnn - np.exp(1j * theta_Css)) / (Cnn * np.cos(theta_Css) - 1)) * \
          np.abs((np.exp(-1j * theta_Css) * Cnn - np.exp(-1j * theta_Css) * Cxx) /
                 (np.real(np.exp(-1j * theta_Css) * Cxx) - 1))

    # 실수부만 유지하고, 음수 값 방지
    CDR = np.maximum(np.real(CDR), 0)

    return CDR

import numpy as np

def estimate_cdr_nodoa(Cxx, Cnn):
    """
    DOA-independent CDR estimation (TDOA 없이 CDR 추정)
    
    Parameters:
        Cxx: np.array, 복소수 코히어런스 (관찰된 신호)
        Cnn: np.array, 노이즈 코히어런스 (실수값)
    
    Returns:
        CDR: np.array, Coherent-to-Diffuse Ratio (CDR)
    """
    # ✅ Cnn을 Cxx와 같은 크기로 브로드캐스트
    Cnn = np.broadcast_to(Cnn, Cxx.shape)

    # ✅ 수치적 안정성을 위해 Cxx 크기 제한
    magnitude_threshold = 1 - 1e-10
    Cxx = np.where(np.abs(Cxx) > magnitude_threshold,
                   magnitude_threshold * Cxx / np.abs(Cxx),
                   Cxx) ##코히런시 0에서 1사이 1이상 튀는 값들 정규화

    # ✅ CDR 계산
    CDR = (-np.sqrt(np.abs(Cxx)**2 + Cnn**2 * np.real(Cxx)**2 - Cnn**2 * np.abs(Cxx)**2 
            - 2 * Cnn * np.real(Cxx) + Cnn**2) 
            - np.abs(Cxx)**2 + Cnn * np.real(Cxx)) / (np.abs(Cxx)**2 - 1)
    print('cdr' , CDR.shape)
    # ✅ 실수부만 유지하고, 음수 값 방지
    CDR = np.maximum(np.real(CDR), 0)

    return CDR
