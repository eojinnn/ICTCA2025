import torch

def estimate_cdr_robust_unbiased(Cxx, Cnn, Css):
    """
    Coherent-to-Diffuse Ratio (CDR) 계산 함수 (MATLAB 코드 변환)
    Cxx: 관찰된 신호의 크로스 PSD (Cross Power Spectral Density)
    Cnn: 확산 잡음의 PSD (Diffuse Noise PSD)
    Css: 목표 신호(직접 도달 음)의 PSD
    """
    # `Css`와 `Cnn`을 `Cxx` 크기로 브로드캐스트
    Css = Css.expand_as(Cxx)
    Cnn = Cnn.expand_as(Cxx)

    # 수치적 안정성을 위해 `Cxx` 크기 제한
    magnitude_threshold = 1 - 1e-10
    Cxx = torch.where(torch.abs(Cxx) > magnitude_threshold,
                   magnitude_threshold * Cxx / torch.abs(Cxx),
                   Cxx)

    # CDR 계산
    theta_Css = torch.angle(Css)  # Css의 위상

    CDR = 1.0 / (-torch.abs(Cnn - torch.exp(1j * theta_Css)) / (Cnn * torch.cos(theta_Css) - 1)) * \
          torch.abs((torch.exp(-1j * theta_Css) * Cnn - torch.exp(-1j * theta_Css) * Cxx) /
                 (torch.real(torch.exp(-1j * theta_Css) * Cxx) - 1))

    # 실수부만 유지하고, 음수 값 방지
    CDR = torch.maximum(torch.real(CDR), 0)

    return CDR

def estimate_cdr_nodoa(Cxx, Cnn):
    """
    DOA-independent CDR estimation (TDOA 없이 CDR 추정)
    
    Parameters:
        Cxx: np.array, 복소수 코히어런스 (관찰된 신호)
        Cnn: np.array, 노이즈 코히어런스 (실수값)
    
    Returns:
        CDR: np.array, Coherent-to-Diffuse Ratio (CDR)
    """
    Cnn = Cnn.expand_as(Cxx)

    # ✅ 수치적 안정성을 위해 Cxx 크기 제한
    magnitude_threshold = 1 - 1e-10
    Cxx = torch.where(torch.abs(Cxx) > magnitude_threshold,
                   magnitude_threshold * Cxx / torch.abs(Cxx),
                   Cxx) ##코히런시 0에서 1사이 1이상 튀는 값들 정규화
    
    denominator = torch.abs(Cxx)**2 - 1
    denominator = torch.where(denominator == 0, torch.tensor(1e-8, device = denominator.device), denominator)  # 0 방지

    # ✅ CDR 계산
    CDR = (-torch.sqrt(torch.abs(Cxx)**2 + Cnn**2 * torch.real(Cxx)**2 - Cnn**2 * torch.abs(Cxx)**2 
            - 2 * Cnn * torch.real(Cxx) + Cnn**2) 
            - torch.abs(Cxx)**2 + Cnn * torch.real(Cxx)) / denominator

    # ✅ 실수부만 유지하고, 음수 값 방지
    CDR = torch.clamp(torch.real(CDR), min = 0.0)

    return CDR
