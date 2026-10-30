# Robust Time Delay Estimation via Probabilistic Weighting of Parameterized GCC-PHAT on Enhanced Signals

**Eojin Kim, Sungbin Im** | Soongsil University | ICTCA 2025

[Code](https://github.com/eojinnn/ICTCA2025) | [ICEIC 2025](https://github.com/eojinnn/ICEIC-2025) | [Research Portfolio](https://github.com/eojinnn/Research-portfolio)

A follow-up to our ICEIC 2025 work on CDR-enhanced GCC-PHAT and CNN-based time-delay estimation.

Compared with ICEIC, this implementation makes two main changes:
- **CDR-based enhancement:** Replaces binary diffuseness masking with continuous spectral weighting.
- **Probabilistic weighting:** Applies Softmax to GCC-PHAT correlation features and uses the resulting weights to modulate CNN features **before** the final regression head.

## Environment

Python with PyTorch, torchaudio, NumPy, librosa, Pyroomacoustics, and other dependencies imported by the scripts. Exact package versions are not pinned.

## Data Preparation

1. Download [LibriSpeech](https://www.openslr.org/12/) (the experiment code uses the `test-clean` split).
2. Configure microphone geometry, SNR/T60 ranges, and model options in [`cfg.py`](cfg.py).
3. Specify the local LibriSpeech root using `--data_loc`.

## Training

```bash
python main.py --exp_name proposed --data_loc /path/to/LibriSpeech
```

## Evaluation

```bash
python main.py --exp_name proposed --data_loc /path/to/LibriSpeech --evaluate
```

The default model is `Probabilistic_PGCCPHAT` in [`model.py`](model.py). CDR-based enhancement is implemented in [`cdr_dereverb.py`](cdr_dereverb.py). Evaluation requires a compatible checkpoint at `experiments/proposed/model.pth`.

**Note:** These are research scripts and have not been independently validated as a clean-environment reproduction. Local paths and dependencies may need adjustment.

## Acknowledgements

This work extends our [ICEIC 2025 implementation](https://github.com/eojinnn/ICEIC-2025). Its room simulation and training/evaluation pipeline are based on [Neural Generalized Cross Correlations (NGCC)](https://github.com/axeber01/ngcc). The original [MIT license notice](LICENSES/NGCC-MIT.txt) is retained.

## Citation

```bibtex
@inproceedings{kim2025ictca,
  title={Robust Time Delay Estimation via Probabilistic Weighting of Parameterized GCC-PHAT on Enhanced Signals},
  author={Kim, Eojin and Im, Sungbin},
  booktitle={Proceedings of the 16th International Conference on Theoretical and Computational Acoustics (ICTCA)},
  year={2025}
}
```
