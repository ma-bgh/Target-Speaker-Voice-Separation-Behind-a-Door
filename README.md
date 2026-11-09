# Target-Speaker Voice Separation Behind a Door

Extracting one known speaker's voice from a noisy "party" mixture when the recording is made from behind a closed door.

The repository contains a runnable single-channel demo (`voice_separation_Party.py`) and the design of a complete multi-microphone pipeline (AR/LPC speaker fingerprint → harmonic tracking → speaker verification → TDOA localization → MVDR beamforming). It was developed as a coursework project for *Speech and Speaker Recognition*.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Method Overview](#method-overview)
3. [What the Demo Implements](#what-the-demo-implements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Simulated Scenario](#simulated-scenario)
7. [Outputs](#outputs)
8. [Project Status and Roadmap](#project-status-and-roadmap)
9. [Repository Structure](#repository-structure)
10. [References](#references)

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

The observed signal at microphone *m* in the STFT domain is modeled as:

$$
X_m(\tau, f) \approx \sum_{i=1}^{I} H_{m,i}(f)\, S_i(\tau, f) + V_m(\tau, f)
$$

where $S_i$ are the speaker sources, $H_{m,i}$ the acoustic paths from source *i* to microphone *m*, and $V_m$ the noise. The goal is to recover $S_1$, the clean voice of the target speaker.

---

## Method Overview

The approach has three stages:

1. **Learn the target's voice** from a short reference sample.
2. **Find where the target speaks** in the mixture.
3. **Separate** the voice using spatial information and filtering.

### 1. Speaker fingerprint with an AR model

Speech is modeled as the output of an all-pole (autoregressive) filter driven by white noise. The poles of this filter correspond to the **formants**.

$$
s[n] = -\sum_{k=1}^{p} a_k\, s[n-k] + e[n], \qquad
P(\omega) = \frac{\sigma^2}{\left|1 + \sum_{k=1}^{p} a_k e^{-j\omega k}\right|^2}
$$

- Coefficients are estimated with the **Yule–Walker equations**, solved efficiently with the **Levinson–Durbin** recursion:

  $$
  \mathbf{a} = -\mathbf{R}^{-1}\mathbf{r}, \qquad \sigma^2 = r_{xx}(0) + \sum_{k=1}^{p} a_k r_{xx}(k)
  $$

- Analysis is done on **25 ms frames**, since speech is only quasi-stationary over very short intervals.
- The model order *p* is selected with **AIC** and **BIC**:

  $$
  \mathrm{AIC}_p = N\ln\sigma_p^2 + 2p, \qquad \mathrm{BIC}_p = N\ln\sigma_p^2 + p\ln N
  $$

  For 8 kHz speech *p* is typically about 10. The procedure was validated on the AR(3) example from the reference article using 1000 samples; both criteria were minimized at *p* = 3, matching the published result.

Two quantities are stored from a few seconds of reference speech:

- **LPC vector (`a_S`)** – the spectral envelope, i.e. the speaker's timbre.
- **Fundamental frequency (`f0`)** – the pitch, estimated via autocorrelation:

  $$
  f_0 = \frac{f_s}{\tau^*}, \qquad \tau^* = \arg\max_{\tau}\frac{r_{xx}(\tau)}{r_{xx}(0)}
  $$

### 2. Detecting the target in the mixture

1. **Harmonic tracking.** The mixture is transformed with the STFT. Human voiced speech has energy at integer multiples of `f0`, so for each candidate `f0` in the target's pitch range the harmonic energy is summed, and the `f0` trajectory over time is followed with **dynamic programming**. The trajectory yields a **harmonic mask** that is large on harmonic lines and small between them.
2. **Identity verification.** The spectral shape of each frame is compared with the reference spectrum using the **Itakura distance**:

   $$
   d_I = \ln\frac{\mathbf{a}_S^{T}\mathbf{R}_x\mathbf{a}_S}{\mathbf{a}_x^{T}\mathbf{R}_x\mathbf{a}_x} \;\ge\; 0
   $$

   A frame is labeled as the target if the distance is below a threshold ε **and** its `f0` lies in the target's pitch range. Requiring both conditions avoids accepting another speaker with a similar spectrum.

### 3. Spatial separation and cleanup

Only for frames labeled as the target:

1. **Direction finding (TDOA).** The arrival-time difference between two microphones is found with generalized cross-correlation (phase transform), then converted to an angle:

   $$
   \tau = \arg\max_{\tau}\int \frac{X_1 X_2^{*}}{|X_1 X_2^{*}|}\, e^{j2\pi f\tau}\,df, \qquad \theta = \arccos\frac{c\,\tau}{d}
   $$

2. **MVDR beamforming.** Using the estimated direction, microphone signals are combined so that sound from that direction passes undistorted while other directions are attenuated. Noise and interferer statistics are estimated from frames where the target is silent:

   $$
   \mathbf{w} = \frac{\Phi_{nn}^{-1}\mathbf{d}}{\mathbf{d}^{H}\Phi_{nn}^{-1}\mathbf{d}}, \qquad Y = \mathbf{w}^{H}\mathbf{x}
   $$

---

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
