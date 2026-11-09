import numpy as np
import soundfile as sf
from scipy import signal
from scipy.signal import stft, istft
from scipy.linalg import solve_toeplitz

SR = 16000
DURATION = 7.0

def voiced_synth(f0, formants, dur, seed=0):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    tt = np.arange(n) / SR
    f_inst = f0 * (1 + 0.025*np.sin(2*np.pi*2.1*tt))
    phase = 2*np.pi*np.cumsum(f_inst) / SR
    x = np.zeros(n)
    for k in range(1, 35):
        x += np.sin(k*phase + rng.uniform(-0.04, 0.04)) / k
    freqs = np.fft.rfftfreq(n, 1/SR)
    env = np.zeros_like(freqs)
    for fc, bw, gain in formants:
        env += gain*np.exp(-0.5*((freqs-fc)/bw)**2)
    env = np.maximum(env, 0.015)
    y = np.fft.irfft(np.fft.rfft(x)*env, n=n)
    amp = np.minimum(1, tt/0.12)*np.minimum(1, (dur-tt)/0.18)
    y *= np.clip(amp, 0, 1)
    return y / (np.max(np.abs(y)) + 1e-9)

def lpc_coeff(x, order=16):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    r = np.correlate(x, x, mode="full")[len(x)-1:len(x)+order]
    if len(r) < order+1 or r[0] < 1e-10:
        return np.r_[1.0, np.zeros(order)]
    try:
        tail = solve_toeplitz((r[:order], r[:order]), -r[1:order+1])
        return np.r_[1.0, tail]
    except Exception:
        return np.r_[1.0, np.zeros(order)]

def lpc_envelope(a):
    _, h = signal.freqz([1.0], a, worN=257, fs=SR)
    return np.abs(h)

def main():
    t = np.arange(int(SR*DURATION))/SR
    
    target = voiced_synth(145, [(520,150,1.0),(1450,240,0.65),(2500,400,0.22)], DURATION, 3)
    other1 = voiced_synth(205, [(850,180,1.0),(1900,300,0.7),(3000,400,0.2)], DURATION, 4)
    other2 = voiced_synth(112, [(330,120,0.9),(1050,220,0.8),(2200,350,0.25)], DURATION, 5)

    rng = np.random.default_rng(12)
    mod1 = 0.55 + 0.35*np.sin(2*np.pi*0.32*t)**2
    mod2 = 0.45 + 0.40*np.sin(2*np.pi*0.21*t+1.3)**2
    noise = rng.normal(0, 0.10, len(t))

    def door_filter(x):
        sos = signal.butter(4, 2300, fs=SR, output="sos")
        return signal.sosfilt(sos, x)

    ref = door_filter(target)
    mix = (0.70*ref + 0.48*mod1*door_filter(other1)
           + 0.38*mod2*door_filter(other2) + noise)
    mix = mix/(np.max(np.abs(mix))+1e-9)*0.92
    ref = ref/(np.max(np.abs(ref))+1e-9)*0.85

    frame, hop = 400, 160
    envs = []
    for start in range(0, len(ref)-frame, hop):
        a = lpc_coeff(ref[start:start+frame]*np.hamming(frame), 16)
        envs.append(lpc_envelope(a))
    ref_env = np.mean(envs, axis=0)
    ref_env /= np.max(ref_env)+1e-9

    f, _, Z = stft(mix, fs=SR, nperseg=512, noverlap=384)
    prior = np.interp(f, np.linspace(0, SR/2, len(ref_env)), ref_env)
    prior = (prior/(np.max(prior)+1e-9))**1.4
    harmonic = np.zeros_like(f)
    for k in range(1, 18):
        harmonic += np.exp(-0.5*((f-k*145)/18)**2)/k
    harmonic /= np.max(harmonic)+1e-9

    mask = 0.20 + 0.80*(0.55*prior + 0.45*harmonic)
    mag = np.abs(Z)
    floor = np.percentile(mag, 35, axis=1, keepdims=True)
    target_power = (mask[:, None]*mag)**2
    noise_power = (floor+1e-6)**2
    gain = np.clip(target_power/(target_power+noise_power+1e-9), 0.12, 1.0)

    _, result = istft(Z*gain, fs=SR, nperseg=512, noverlap=384)
    result = result[:len(mix)]
    result = result/(np.max(np.abs(result))+1e-9)*0.92

    sf.write("demo_party_mix.wav", mix, SR)
    sf.write("demo_reference_voice.wav", ref, SR)
    sf.write("demo_extracted_voice.wav", result, SR)
    print("It’s done. Three WAV files have been created in this folder::")
    print("1) demo_party_mix.wav")
    print("2) demo_reference_voice.wav")
    print("3) demo_extracted_voice.wav")

if __name__ == "__main__":
    main()
