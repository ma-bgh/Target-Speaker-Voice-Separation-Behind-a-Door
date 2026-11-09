# Target-Speaker Voice Separation Behind a Door

Extracting one known speaker's voice from a noisy "party" mixture when the recording is made from behind a closed door.

The repository contains a runnable single-channel demo (`voice_separation_Party.py`) and the design of a complete multi-microphone pipeline (AR/LPC speaker fingerprint → harmonic tracking → speaker verification → TDOA localization → MVDR beamforming). It was developed as a coursework project for *Speech and Speaker Recognition*.

---
## Problem Statement

A person stands behind a door and wants to hear **one specific speaker** inside a room full of party noise and other voices. The door acts as a low-pass filter, so the target voice arrives muffled and mixed with interfering speakers and background noise.

The following five acquisition scenarios are considered in the project:

| # | Speaker | Microphones |
|---|---------|-------------|
| 1 | Stationary | None inside the room |
| 2 | Stationary | One stationary microphone inside the room |
| 3 | Stationary | One moving microphone inside the room |
| 4 | Moving | One moving microphone |
| 5 | Moving | Two moving microphones inside the room |

## Method Overview

The approach has three stages:

1. **Learn the target's voice** from a short reference sample.
2. **Find where the target speaks** in the mixture.
3. **Separate** the voice using spatial information and filtering.



## What the Demo Implements

`voice_separation_Party.py` is a **self-contained, single-channel demonstration** that needs no external audio files. It generates a synthetic scene and extracts the target with a spectral-prior mask:

1. **Synthetic voices.** Three voiced signals are synthesized from a harmonic source (with slow pitch vibrato) shaped by Gaussian formant envelopes.
2. **Door effect.** Every source is passed through a 4th-order Butterworth low-pass filter at 2300 Hz.
3. **Mixture.** The target is mixed with two amplitude-modulated interfering speakers and white Gaussian noise.
4. **Reference fingerprint.** From the reference signal, LPC coefficients (order 16, autocorrelation / Yule–Walker via `scipy.linalg.solve_toeplitz`) are computed on 25 ms Hamming-windowed frames (10 ms hop). The LPC spectral envelopes are averaged into a single reference envelope.
5. **Time–frequency mask.** The mixture STFT (512-point window, 75 % overlap) is weighted by a mask combining
   - a **spectral prior** from the averaged reference envelope, and
   - a **harmonic comb** at the target `f0` (145 Hz, 17 harmonics).

   ```
   mask = 0.20 + 0.80 * (0.55 * prior + 0.45 * harmonic)
   ```
6. **Noise-floor gain.** A Wiener-style gain is computed against a per-frequency noise floor (35th magnitude percentile) and clipped to [0.12, 1.0].
7. **Resynthesis.** The inverse STFT produces the extracted voice, which is peak-normalized.

---

## Installation

Python 3.8+ is recommended.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install numpy scipy soundfile
```

`soundfile` relies on `libsndfile`; on most platforms it is installed automatically with the pip wheel.

---

## Usage

```bash
python voice_separation_Party.py
```

The script runs in a few seconds and prints:

```
It’s done. Three WAV files have been created in this folder::
1) demo_party_mix.wav
2) demo_reference_voice.wav
3) demo_extracted_voice.wav
```

### Tunable parameters

Parameters are set at the top of the file or inline in `main()`:

| Parameter | Location | Meaning |
|-----------|----------|---------|
| `SR` | top of file | Sampling rate (16 kHz) |
| `DURATION` | top of file | Length of the synthetic scene in seconds (7.0) |
| `f0`, formants | `voiced_synth(...)` calls | Pitch and formant structure of each synthetic speaker |
| mixing gains | `mix = ...` | Relative level of target, interferers and noise |
| LPC order | `lpc_coeff(..., 16)` | Order of the AR model |
| `frame`, `hop` | `main()` | Analysis frame (400 samples = 25 ms) and hop (160 samples = 10 ms) |
| mask weights | `mask = ...` | Balance between spectral prior and harmonic comb |
| `gain` clip range | `np.clip(..., 0.12, 1.0)` | Maximum attenuation / amplification of the gain |

---

## Simulated Scenario

| Source | f0 (Hz) | Role | Level |
|--------|---------|------|-------|
| Target | 145 | Speaker to extract | 0.70 |
| Interferer 1 | 205 | Competing voice (slow amplitude modulation) | 0.48 |
| Interferer 2 | 112 | Competing voice (slow amplitude modulation) | 0.38 |
| Noise | – | White Gaussian, σ = 0.10 | – |

All voices pass through the same door low-pass filter (2300 Hz cutoff) before mixing.

---

## Outputs

| File | Description |
|------|-------------|
| `demo_party_mix.wav` | The noisy mixture, as heard behind the door |
| `demo_reference_voice.wav` | The clean (door-filtered) target used to learn the fingerprint |
| `demo_extracted_voice.wav` | The target voice recovered from the mixture |

Listen to the mixture first, then the extracted result, and compare with the reference.

---

## Project Status and Roadmap

The report describes the full multi-microphone pipeline. The current code implements the **single-channel part** of that design.

| Component | Status |
|-----------|--------|
| AR / LPC fingerprint (Yule–Walker, 25 ms frames) | Implemented in the demo |
| Harmonic mask from `f0` | Implemented with a **fixed** `f0` (145 Hz) |
| Spectral-prior masking with noise-floor gain | Implemented |
| Model-order selection with AIC / BIC | Described in the report; not in the demo script |
| `f0` estimation by autocorrelation and DP `f0` tracking | Planned |
| Frame-level target verification (Itakura distance + `f0` range) | Planned |
| TDOA direction finding (GCC-PHAT) | Planned |
| MVDR beamforming with two microphones | Planned |
| Moving speaker / moving microphone scenarios | Planned |

---

## Repository Structure

```
.
├── voice_separation_Party.py   # Runnable single-channel demo
├── README.md
└── docs/
    └── report.docx             # Project report (optional, Persian)
```

---

## References

1. M. Viswanathan, "Yule Walker Estimation and simulation in Matlab," *GaussianWaves*, 2014.
2. M. Viswanathan, "AR, MA and ARMA Models," *GaussianWaves*, 2014.
3. L. Rabiner and R. Schafer, *Digital Processing of Speech Signals*, 1978.
4. F. Itakura, "Minimum prediction residual principle applied to speech recognition," *IEEE Trans. ASSP*, 23(1), 1975.
5. C. Knapp and G. Carter, "The generalized correlation method for estimation of time delay," *IEEE Trans. ASSP*, 24(4), 1976.
6. J. Capon, "High-resolution frequency-wavenumber spectrum analysis," *Proc. IEEE*, 57(8), 1969.
7. P. Loizou, *Speech Enhancement: Theory and Practice*, 2007.
8. A. Jourjine, S. Rickard and O. Yilmaz, "Blind separation of disjoint orthogonal signals," *ICASSP*, 2000.
