'''This code takes the samples of 4000 audio (~10% of total audio samples from 8 languages) from 
each language and resamples them to 16KHz for mathematical comparison.  It chops the audio into 30ms frame and calculates 8 distinct acoustic 
metrics.  Use Parallel processing to handle thousands of files rapidly.  

6 Analytical plots - Boxplots, Histogram, Retention Curves, Heatmaps, IQR spread and Radar chart 
are made to help me decide between Global vs Per-language quality thresholds.

Metrics Used - ZCR, Clipping Rate, Silence Ratio, Waveform Kurtosis, Spectral Flatness, SNR, 
ELR(Early-to-Late Ratio, C50_db), VAD Ratio.

The plots is saved to as PNG files in the `./plots` directory and CSV Data is saved to 
`./metrics_sample.csv` .

It dynamically scales to use `multiprocessing.cpu_count() - 1` cores, meaning it will run safely 
and efficiently on everything from a 4-core laptop to a 64-core cloud VM without freezing the OS.

The insights are saved in Readme.md'''



ROOT      = "./data/audios"  
OUTPUT    = "./Decision_plots_and_csv"
SAVE_CSV  = "./Decision_plots_and_csv/metrics_sample.csv"
LANGUAGES = ["hindi", "tamil", "telugu", "kannada", "malayalam", "bengali", "gujarati", "marathi"]
N_SAMPLES = 400
N_WORKERS = max(1, multiprocessing.cpu_count() - 1)
print(f"Workers: {N_WORKERS}")

import json
import logging
import random
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import kurtosis as scipy_kurtosis
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("analysis")

LANG_PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
]
plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor":   "#F5F5F5",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.4,
    "font.family":      "monospace",
    "font.size":        10,
})

METRICS = [
    "zcr", "clipping_rate", "silence_ratio", "kurtosis",
    "spectral_flatness", "snr_db", "c50_db", "vad_ratio",
]
METRIC_LABELS = {
    "zcr":              "ZCR (zero-crossing rate)",
    "clipping_rate":    "Clipping Rate",
    "silence_ratio":    "Silence Ratio",
    "kurtosis":         "Waveform Kurtosis",
    "spectral_flatness":"Spectral Flatness",
    "snr_db":           "SNR — WADA proxy (dB)",
    "c50_db":           "ELR — Early-to-Late Ratio (dB)",
    "vad_ratio":        "VAD Ratio — energy+ZCR (voiced fraction)",
}
HIGHER_IS_BETTER = {
    "zcr":              False,
    "clipping_rate":    False,
    "silence_ratio":    False,
    "kurtosis":         False,
    "spectral_flatness":False,
    "snr_db":           True,
    "c50_db":           True,
    "vad_ratio":        True,
}

EPS         = 1e-10
FRAME_MS    = 30
SILENCE_DB  = -40.0
AUDIO_EXT   = {".flac"}
MIN_DURATION_S=0.3
TARGET_SR = 16000

def compute_snr_wada(signal, sr):
    """
    WADA-SNR proxy: sort frames by RMS power, treat bottom 20% as noise floor.
    SNR = 10 * log10(mean_signal_power / mean_noise_power).
    Reliable for read speech with inter-word pauses acting as noise samples.
    """
    frames      = frame_signal(signal, sr)
    power       = np.mean(frames ** 2, axis=1) + EPS
    sorted_pow  = np.sort(power)
    n_noise     = max(int(len(sorted_pow) * 0.20), 1)
    noise_power = np.mean(sorted_pow[:n_noise])
    sig_power   = np.mean(power)
    return float(np.clip(10 * np.log10(sig_power / noise_power), -5, 60))


def compute_elr(signal, sr):
    """
    Early-to-Late energy Ratio — model-free proxy for C50.
    ELR = 10 * log10(energy[0:50ms] / energy[50ms:])
    High ELR = dry/clear room. Low ELR = reverberant environment.
    """
    t50   = int(0.050 * sr)
    if len(signal) <= t50:
        return 0.0
    early = np.sum(signal[:t50] ** 2) + EPS
    late  = np.sum(signal[t50:] ** 2) + EPS
    return float(10 * np.log10(early / late))


def compute_vad_ratio(signal, sr):
    """
    Fraction of 30ms frames classified as voiced.
    Voiced = RMS above adaptive floor AND ZCR below 0.3 (not pure noise).
    """
    frames    = frame_signal(signal, sr)
    rms_db    = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1) + EPS))
    zcr_per   = np.mean(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / 2
    energy_th = np.percentile(rms_db, 20)  
    voiced    = (rms_db > energy_th) & (rms_db > SILENCE_DB) & (zcr_per < 0.3)
    return float(np.mean(voiced))

    
def load_mono(path: str, target_sr: int = TARGET_SR):
    signal, sr = sf.read(path, dtype="float32", always_2d=False)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if sr != target_sr:
        n_out  = int(len(signal) * target_sr / sr)
        signal = np.interp(
            np.linspace(0, len(signal) - 1, n_out),
            np.arange(len(signal)), signal,
        )
    return signal.astype(np.float32), target_sr


def frame_signal(signal, sr):
    frame_len = int(sr * FRAME_MS / 1000)
    n_frames  = len(signal) // frame_len
    if n_frames == 0:
        return signal[np.newaxis, :]
    return signal[:n_frames * frame_len].reshape(n_frames, frame_len)


def compute_zcr(signal, sr):
    frames = frame_signal(signal, sr)
    return float(np.mean(np.mean(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / 2))

def compute_clipping_rate(signal, threshold=0.9999):
    return float(np.mean(np.abs(signal) >= threshold))

def compute_silence_ratio(signal, sr):
    frames = frame_signal(signal, sr)
    rms_db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1) + EPS))
    return float(np.mean(rms_db < SILENCE_DB))

def compute_kurtosis(signal):
    return float(np.clip(scipy_kurtosis(signal, fisher=False), 0, 50))

def compute_spectral_flatness(signal, sr, n_fft=512):
    hop    = n_fft // 2
    window = np.hanning(n_fft)
    n_frames = max(1, (len(signal) - n_fft) // hop)
    vals = []
    for i in range(n_frames):
        frame = signal[i*hop : i*hop + n_fft] * window
        power = np.abs(np.fft.rfft(frame)) ** 2 + EPS
        vals.append(np.exp(np.mean(np.log(power))) / np.mean(power))
    return float(np.clip(np.mean(vals), 0, 1))


def compute_metrics(path: str) -> dict | None:
    try:
        signal, sr = load_mono(path)
    except Exception as e:
        log.debug("Cannot load %s: %s", path, e)
        return None
    if len(signal) / sr < MIN_DURATION_S:
        return None
    result = {
        "path":             path,
        "duration_s":       round(len(signal) / sr, 3),
        "zcr":              compute_zcr(signal, sr),
        "clipping_rate":    compute_clipping_rate(signal),
        "silence_ratio":    compute_silence_ratio(signal, sr),
        "kurtosis":         compute_kurtosis(signal),
        "spectral_flatness":compute_spectral_flatness(signal, sr),
    }
    result["snr_db"]    = compute_snr_wada(signal, sr)
    result["c50_db"]    = compute_elr(signal, sr)
    result["vad_ratio"] = compute_vad_ratio(signal, sr)
    return result


def collect_files(root: Path, language: str) -> list:
    files = []
    
    lang_dir = root / language
    if lang_dir.exists():
        files = [str(p) for p in lang_dir.rglob("*") if p.suffix.lower() in AUDIO_EXT]
    
    if not files:
        files = [
            str(p) for p in root.rglob("*")
            if p.suffix.lower() in AUDIO_EXT and p.parts[-3] == language
        ]
    
    return files

def sample_and_compute(root: Path, languages: list, n: int) -> pd.DataFrame:
    lang_files = {}
    for lang in languages:
        all_files = collect_files(root, lang)
        if not all_files:
            log.warning("No files for '%s'", lang)
            continue
        random.seed(42)
        sampled = random.sample(all_files, min(n, len(all_files)))
        lang_files[lang] = sampled
        log.info("[%s] sampling %d / %d files", lang, len(sampled), len(all_files))

    rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(compute_metrics, f): (f, lang)
            for lang, files in lang_files.items()
            for f in files
        }
        with tqdm(total=len(futures), desc="Computing metrics", unit="file") as pbar:
            for future in as_completed(futures):
                filepath, lang = futures[future]
                try:
                    result = future.result()
                    if result is not None:
                        result["language"] = lang
                        rows.append(result)
                except Exception as e:
                    log.debug("Worker error %s: %s", filepath, e)
                finally:
                    pbar.update(1)
    log.info("Total rows collected: %d", len(rows))
    return pd.DataFrame(rows)


def color_map(languages):
    return {l: LANG_PALETTE[i % len(LANG_PALETTE)] for i, l in enumerate(languages)}


def _save(fig, out: Path, name: str):
    p = out / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", p)
    

def plot_boxplots(df, languages, out):
    cmap = color_map(languages)
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.flatten()
    for i, metric in enumerate(METRICS):
        ax   = axes[i]
        data = [df[df["language"] == l][metric].dropna().values for l in languages]
        bp   = ax.boxplot(
            data, patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker=".", markersize=3, alpha=0.4),
        )
        for patch, lang in zip(bp["boxes"], languages):
            patch.set_facecolor(cmap[lang]); patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(languages) + 1))
        ax.set_xticklabels(languages, fontsize=9)
        ax.set_title(METRIC_LABELS[metric], fontsize=10, fontweight="bold")
        for j, (lang, d) in enumerate(zip(languages, data)):
            if len(d) > 4:
                q25, q75 = np.percentile(d, [25, 75])
                ax.text(j + 1, q75, f" {q75-q25:.2f}",
                        fontsize=6, va="bottom", color=cmap[lang], alpha=0.9)
    fig.suptitle("Boxplots per Language  (spread → per-lang | overlap → universal ok)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, out, "01_boxplots_per_language.png")


def plot_histograms(df, languages, out):
    cmap = color_map(languages)
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.flatten()
    for i, metric in enumerate(METRICS):
        ax    = axes[i]
        all_v = df[metric].dropna()
        bins  = np.linspace(all_v.quantile(0.01), all_v.quantile(0.99), 40)
        for lang in languages:
            vals = df[df["language"] == lang][metric].dropna()
            if len(vals) < 2: continue
            ax.hist(vals, bins=bins, alpha=0.45, color=cmap[lang], label=lang, density=True)
        ax.set_title(METRIC_LABELS[metric], fontsize=10, fontweight="bold")
        ax.set_ylabel("Density", fontsize=8)
        if i == 0: ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Metric Histograms — All Languages Overlaid",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, out, "02_histograms_all_languages.png")


def plot_retention_curves(df, languages, out):
    cmap  = color_map(languages)
    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes  = axes.flatten()
    for i, metric in enumerate(METRICS):
        ax  = axes[i]
        hib = HIGHER_IS_BETTER[metric]
        all_v      = df[metric].dropna()
        thresholds = np.linspace(all_v.quantile(0.01), all_v.quantile(0.99), 100)
        for lang in languages:
            vals = df[df["language"] == lang][metric].dropna().values
            if len(vals) < 5: continue
            ret = [np.mean(vals >= t) if hib else np.mean(vals <= t) for t in thresholds]
            ax.plot(thresholds, ret, color=cmap[lang], label=lang, linewidth=1.8)
        ax.axhline(0.7, color="gray",  linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axhline(0.9, color="green", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_title(METRIC_LABELS[metric], fontsize=10, fontweight="bold")
        ax.set_xlabel("Threshold", fontsize=8)
        ax.set_ylabel("Retention", fontsize=8)
        ax.set_ylim(0, 1.05)
        if i == 0: ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Retention Curves  (dashed: 70% and 90% marks)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, out, "03_retention_curves.png")


def plot_correlation_heatmap(df, out):
    corr    = df[METRICS].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im      = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    labels  = [METRIC_LABELS[m].split("(")[0].strip() for m in METRICS]
    ax.set_xticks(range(len(METRICS))); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(METRICS))); ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(METRICS)):
        for j in range(len(METRICS)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if abs(v) > 0.5 else "black")
    ax.set_title("Metric Correlation Heatmap  (|r| > 0.8 → likely redundant pair)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, out, "04_correlation_heatmap.png")


def plot_iqr_spread(df, languages, out):
    records = []
    for metric in METRICS:
        for lang in languages:
            vals = df[df["language"] == lang][metric].dropna().values
            if len(vals) < 4: continue
            q25, q75 = np.percentile(vals, [25, 75])
            records.append({"metric": metric, "language": lang, "iqr": q75 - q25})
    sdf     = pd.DataFrame(records)
    summary = sdf.groupby("metric")["iqr"].agg(["mean", "std"]).reset_index()
    summary["cv"] = summary["std"] / (summary["mean"] + EPS)
    summary = summary.sort_values("cv", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(summary["metric"], summary["cv"], color=LANG_PALETTE[0], alpha=0.8)
    ax.axvline(0.25, color="orange", linestyle="--", linewidth=1, label="CV=0.25 → consider per-lang")
    ax.axvline(0.50, color="red",    linestyle="--", linewidth=1, label="CV=0.50 → per-lang strongly advised")
    ax.set_xlabel("CV of IQR across languages", fontsize=10)
    ax.set_title("Cross-Language IQR Spread per Metric\n(High CV → per-language threshold needed)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, out, "05_iqr_spread_per_metric.png")


def plot_radar(df, languages, out):
    norm = df.copy()
    for m in METRICS:
        lo, hi  = norm[m].quantile(0.05), norm[m].quantile(0.95)
        rng     = hi - lo + EPS
        norm[m] = ((norm[m] - lo) / rng) if HIGHER_IS_BETTER[m] else (1 - (norm[m] - lo) / rng)
        norm[m] = norm[m].clip(0, 1)
    cats   = [METRIC_LABELS[m].split("(")[0].strip() for m in METRICS]
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    angles += angles[:1]
    cmap   = color_map(languages)
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for lang in languages:
        vals = norm[norm["language"] == lang][METRICS].median().values.tolist()
        vals += vals[:1]
        ax.plot(angles, vals, color=cmap[lang], linewidth=2, label=lang)
        ax.fill(angles, vals, color=cmap[lang], alpha=0.12)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, size=9)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
    ax.set_title("Language Quality Fingerprints\n(Outer = better quality)",
                 fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    _save(fig, out, "06_radar_language_profiles.png")

if __name__ == "__main__":
    out = Path(OUTPUT)
    out.mkdir(parents=True, exist_ok=True)

    df = sample_and_compute(Path(ROOT), LANGUAGES, N_SAMPLES)
    df.to_csv(SAVE_CSV, index=False)
    log.info("Metrics CSV saved: %s", SAVE_CSV)

    languages = [l for l in LANGUAGES if l in df["language"].unique()]
    
    plot_boxplots(df, languages, out)
    plot_histograms(df, languages, out)
    plot_retention_curves(df, languages, out)
    plot_correlation_heatmap(df, out)
    plot_iqr_spread(df, languages, out)
    plot_radar(df, languages, out)

    print(f"\nPlots → {out}")
    print(f"CSV   → {SAVE_CSV}")
