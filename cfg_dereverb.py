#filterbank initialization
K = 512; # FFT size
N = 128; # frame shift
Lp = 512; # prototype filter length
#p=IterLSDesign(cfg.Lp,cfg.K,cfg.N);
#load('lib/filterbank/prototype_K512_N128_Lp1024.mat');

# algorithm and scenario configuration
fs = 22050      # sampling rate [Hz]
c = 342         # speed of sound [m/s]
d_mic = 9    # mic spacing [m]

# all estimators except estimate_cdr_nodoa require the TDOA of the signal; make sure
# to adapt this when loading another wave file
TDOA = 2.15e-04 # ground truth for wav/roomC-2m-75deg.wav

nr = {
    "lambda": 0.68,  # smoothing factor for psd estimation
    "mu": 1.3,  # noise overestimation factor
    "floor": 0.1,  # minimum gain
    "alpha": 2,  
    "beta": 0.5, #magnitude subtraction
}
'''
alpha = 1, beta = 1 for power subtraction
alpha = 2, beta = 1 for wiener filter
'''


#estimator = estimate_cdr_unbiased;           # unbiased estimator (CDRprop1)
estimator = "estimate_cdr_robust_unbiased"    #unbiased, "robust" estimator (CDRprop2)
#estimator = estimate_cdr_nodoa;              # DOA-independent estimator (CDRprop3)
#estimator = @estimate_cdr_nodiffuse;          # noise coherence-independent estimator (CDRprop4; does not work for TDOA -> 0!)