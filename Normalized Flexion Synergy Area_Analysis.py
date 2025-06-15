
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit

# --- EMG Processing Functions ---

def butter_bandpass(lowcut, highcut, fs, order=4):
    return butter(order, [lowcut / (0.5 * fs), highcut / (0.5 * fs)], btype='band')

def butter_lowpass(cutoff, fs, order=4):
    return butter(order, cutoff / (0.5 * fs), btype='low')

def filter_signal(signal, lowcut=20, highcut=450, fs=1000):
    b, a = butter_bandpass(lowcut, highcut, fs)
    return filtfilt(b, a, signal)

def extract_envelope(signal, lowpass_cutoff=5, fs=1000):
    rectified = np.abs(signal)
    b, a = butter_lowpass(lowpass_cutoff, fs)
    return filtfilt(b, a, rectified)

# --- Utility Functions ---

def find_liftup_segments(trigger, fs=1000):
    rising = np.where((trigger[:-1] < 4) & (trigger[1:] >= 4))[0] + 1
    falling = np.where((trigger[:-1] > 4) & (trigger[1:] <= 4))[0] + 1
    if len(rising) < 12 or len(falling) < 12:
        return [], None
    tasks = [(rising[i], falling[i]) for i in [1, 2, 3]]
    baseline = (rising[11], falling[11])
    return tasks, baseline

def normalize(signal, max_val):
    return (signal / max_val) * 100 if max_val else signal

def exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def compute_avg_slope(x_vals, y_vals):
    dy_dx = np.gradient(y_vals, x_vals)
    return np.mean(dy_dx)

# --- Main Analysis ---

base_dir = os.path.join("path", "to", "your", "emg_data")
processed_folder = "processed_data"
fs = 1000

subjects = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

for subject in subjects:
    subject_path = os.path.join(base_dir, subject, processed_folder)
    liftup_file = next((f for f in os.listdir(subject_path) if "liftup" in f and f.endswith(".csv")), None)
    if not liftup_file:
        continue

    data = pd.read_csv(os.path.join(subject_path, liftup_file))
    trigger = data["Trigger"].values
    deltoid_raw = data["Deltoid"].values
    biceps_raw = data["Biceps"].values

    tasks, _ = find_liftup_segments(trigger, fs)
    if not tasks:
        continue

    indices = np.concatenate([np.arange(start, end) for start, end in tasks if end > start + 1000])
    deltoid_filtered = filter_signal(deltoid_raw[indices], fs=fs)
    biceps_filtered = filter_signal(biceps_raw[indices], fs=fs)

    deltoid_env = extract_envelope(deltoid_filtered, fs=fs)
    biceps_env = extract_envelope(biceps_filtered, fs=fs)

    deltoid_max = np.max(deltoid_env)
    pullin_file = next((f for f in os.listdir(subject_path) if "pullin" in f and "result" in f), None)
    if not pullin_file:
        continue
    biceps_max = pd.read_csv(os.path.join(subject_path, pullin_file)).iloc[0, 1]

    d_norm = normalize(deltoid_env, deltoid_max)
    b_norm = normalize(biceps_env, biceps_max)

    valid = (d_norm > 0) & (d_norm < 100) & (b_norm > 0)
    d_norm, b_norm = d_norm[valid], b_norm[valid]

    try:
        popt_exp, _ = curve_fit(exponential, d_norm, b_norm, p0=[1, 0.01, 0], maxfev=10000)
    except RuntimeError:
        continue

    x_fit = np.linspace(0, 100, 100)
    y_fit = exponential(x_fit, *popt_exp)
    r2 = r_squared(b_norm, exponential(d_norm, *popt_exp))

    # --- NFSA Area and Average Slope Calculation ---
    x_shade = np.linspace(10, 90, 100)
    y_shade = exponential(x_shade, *popt_exp)

    shaded_area = np.trapz(y_shade, x_shade)
    total_area = 100 * 100
    percent_area = (shaded_area / total_area) * 100
    avg_slope = compute_avg_slope(x_shade, y_shade)

    y_20 = exponential(20, *popt_exp)
    y_40 = exponential(40, *popt_exp)

    print(f"{subject} - Exponential R²: {r2:.2f}")
    print(f"{subject} - NFSA Area (10–90): {percent_area:.2f}%")
    print(f"{subject} - Avg Slope (10–90): {avg_slope:.3f}")
    print(f"{subject} - y(20): {y_20:.2f}, y(40): {y_40:.2f}")

    # --- Plot Exponential Fit ---
    plt.figure(figsize=(10, 6))
    plt.scatter(d_norm, b_norm, color='blue', label="Data")
    plt.plot(x_fit, y_fit, color='purple', linewidth=3, linestyle='dotted', label=f"Exponential Fit (R²={r2:.2f})")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel("SABD (% MVC)", fontsize=16)
    plt.ylabel("Synergistic BIC (% MVC)", fontsize=16)
    plt.title(f"NFSA - Exponential Fit", fontsize=16)
    plt.legend(fontsize=14)
    plt.grid()
    plt.show()

    # --- Final Plot with NFSA Shading ---
    plt.figure(figsize=(10, 6))
    plt.scatter(d_norm, b_norm, color='blue')
    plt.plot(x_fit, y_fit, color='red', linewidth=3, label="Exponential Fit")
    plt.axvline(x=10, color='black', linestyle='--')
    plt.axvline(x=90, color='black', linestyle='--')
    plt.fill_between(x_shade, y_shade, color='lightgreen', alpha=0.5, label="NFSA Area")
    plt.legend(fontsize=14)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel("SABD (% MVC)", fontsize=16)
    plt.ylabel("Synergistic BIC (% MVC)", fontsize=16)
    plt.title(f"NFSA Shading - {subject}", fontsize=16)
    plt.grid()
    plt.show()
