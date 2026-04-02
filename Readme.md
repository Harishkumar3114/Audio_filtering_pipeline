# Assignment: Audio Filtering Pipeline for Indic Speech 

* This assignment involves building a scalable audio filtering pipeline to prepare large multilingual Indic speech datasets for high-quality model training.

* It evaluates your engineering design, selection of audio quality metrics, and strategies for efficiently processing data at scale.

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Key Features](#key-features)
* [Prerequisites and Installation](#prerequisites-and-installation)
* [Phase 1: Loading the dataset](#phase-1-loading-the-dataset)
* [Phase 2: Initial Data Assessment & Manual Verification](#Phase-2-Initial-Data-Assessment-&-Manual-Verification)
* [Phase 3: Soft Scoring(Core part of assignment)](#Phase-3-Soft-Scoring)
* [Phase 4: Bonus Phase (ASR done using Whisper)](#Phase-4-Bonus-Phase-ASR-done-using-Whisper)
* [Key Insights and Final Inference](#Key-Insights-and-Final-Inference)



---
## Overview

This project implements a highly scalable, automated curation pipeline that is designed to filter out bad quality audio from the dataset. It processes raw audio in three primary stages:

1. **High-Speed Data Ingestion**: Efficiently downloading, parsing, and extracting raw audio bytes directly from Hugging Face parquet files into structured `.flac` directories.
2. **Context-Aware Acoustic Filtration**: Evaluating audio quality using a custom, per-language normalized "Soft Score" to surgically remove noise, dead air, and distortion without introducing linguistic bias.
3. **Semantic Alignment & Verification**: Using a high-throughput, multi-stage `faster-whisper` pipeline to compare audio against ground-truth text and drop severe transcript mismatches.
---
## Dataset

- **Source**: [IndicVoices](https://huggingface.co/datasets/ai4bharat/IndicVoices) by AI4Bharat
- **Languages Evaluated**: Hindi, Bengali, Marathi, Gujarati, Tamil, Telugu, Kannada, Malayalam
- **Audio Format**: Extracted and standardized to 16 kHz `.flac` Mono
- **Metadata**: Managed via highly structured `.jsonl` manifests containing original text, normalized text, and demographics

--- 

## Key Features

### Evaluation & Filtration Framework

- **Dynamic Hardware Profiling**: Auto-detects available CPU cores and GPU VRAM to dynamically scale batch sizes and compute types (float16 vs int8), preventing OOM crashes.
- **"Zero-Idle" Semantic Pipeline**: A custom 3-stage concurrent architecture (I/O Prefetching → GPU Transcription → CPU Math) that ensures the GPU never stalls while computing Character Error Rates (CER).
- **Hybrid Normalization Soft-Score**: Balances 7 distinct audio metrics using:
  - *Per-Language Normalization*: For highly variable metrics (VAD Ratio, SNR, ZCR).
  - *Global Normalization*: For universal structural flaws (Clipping Rate, Waveform Kurtosis, ELR, Spectral Flatness).
- **Acoustic Profiling Suite**: Automatically generates diagnostic visualizations (Boxplots, Histograms, Radar Charts, Retention Curves) to mathematically prove the validity of rejection thresholds.
- **Streaming Data Extraction**: Bypasses memory bottlenecks by using hybrid Process/Thread pools to parse massive parquet files and write binary audio data directly to disk.

---
## Prerequisites and Installation

```bash
# Python 3.10+
python --version

# CUDA 12.x+ (Recommended for faster-whisper GPU acceleration)
nvcc --version

# 1. Clone the repository
git clone <your-repo-link>
cd <your-repo-folder>

# 2. Recommended: Create and activate the Conda environment
conda env create -f environment.yaml
conda activate indic_speech_filter

# OR: Install manually via pip (if not using Conda)

# Install PyTorch (Update the CUDA URL to match your system's nvcc version)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)

# Install ASR & evaluation models
pip install faster-whisper editdistance

# Install audio processing and math libraries
pip install librosa soundfile scipy numpy

# Install data manipulation, downloading, and UI utilities
pip install pandas polars huggingface-hub tqdm matplotlib

```

## Phase 1: Loading the dataset
* **The Source:** The raw audio is ingested directly from the `ai4bharat/IndicVoices` repository on Hugging Face, which is a comprehensive, crowdsourced database of natural Indian speech recorded in diverse field conditions.
* **8 Languages loaded:**
  * **Indo-Aryan:** Hindi, Bengali, Marathi, and Gujarati.
  * **Dravidian:** Tamil, Telugu, Kannada, and Malayalam.
* **Direct Byte Extraction:** The raw audio is stored as bytes inside `.parquet` files. The script extracts these bytes and reconstructs the actual audio files (`.flac`) on your local drive.
* **Master Manifest Creation:** As it extracts the audio, it builds a `combined_manifest.jsonl` file containing the physical file paths, original transcripts, and demographic metadata for every sample.
* **High-Speed Concurrency:** It uses a hybrid approach—`ProcessPoolExecutor` for parsing the massive parquet files and `ThreadPoolExecutor` for the I/O-heavy task of writing thousands of audio files to disk.

### How to Run
Execute the script from your terminal. You can specify a custom save directory, or let it default to `./data`.

```bash
python dataset_download.py --save_dir ./data
```

### Folder Structure:

data/
├── setup.log                          # Execution logs and error tracking
├── hf/                                # Raw Hugging Face dataset cache
├── audios/                            # Extracted, ready-to-use audio files
│   ├── tamil/
│   │   ├── train-00000-of-XXXXX/      # Grouped by the original parquet index
│   │   │   ├── sample_1.flac
│   │   │   └── sample_2.flac
│   ├── hindi/
│   └── ... (other languages)
└── manifests/                         # Metadata tracking
    ├── tamil_manifests/
    │   └── train-00000-of-XXXXX.jsonl # Metadata for this specific parquet
    ├── hindi_manifests/
    └── combined_manifest.jsonl        # The MASTER manifest used for Phase 1




---



## Phase 2: Initial Data Assessment & Manual Verification

Before designing the automated filtration pipeline, I conducted a manual acoustic audit, listening to 20–30 random samples from each of the 8 languages. This qualitative analysis was critical for understanding the real-world conditions of the dataset. 

I identified six recurring quality issues that were severely degrading the data:

* **Ultra-Short & Silent Clips:** Many recordings were under 1 second in duration, often containing abrupt cut-offs or absolute dead air with no human speech.
* **Heavy Background Noise:** A significant portion of the data suffered from severe environmental noise (traffic, crowds, sustained AC hums) that completely overpowered the primary speaker.
* **Proximity Effect & Clipping:** Speakers frequently held the microphone too close to their mouths. This resulted in heavy plosives (mic pops), breathiness, and permanent digital distortion (clipping).
* **Telephony Glitches:** Audio recorded via cell phone calls often exhibited voice breaks, network dropouts, and heavy digital compression artifacts.
* **Reverberation (Echo):** Several samples were recorded in highly reflective, empty rooms, causing smeared phonemes and heavy room echo that would confuse an ASR model.

**Next Steps:** These manual findings directly informed the selection of the 8 mathematical acoustic metrics (SNR, VAD Ratio, Clipping Rate, ELR, etc.) used in Phase 1 to automatically detect and purge these exact flaws.

---

## Phase 3: Soft Scoring

After manual verification of samples from all 8 languages and the problems figured out, I took 8 metrics to evaluate our data and these metrics capture the majority of bad samples from the data-

* **SNR (Signal-to-Noise Ratio)**: Measures constant background hums (like AC or traffic) to prevent the model from hallucinating words from noise.
* **VAD Ratio (Voice Activity)**: Calculates the percentage of actual human speech, ensuring the model aligns text to voice rather than dead air.
* **ELR (Early-to-Late Ratio)**: Detects room echo and reverberation, preventing the model from confusing smeared word boundaries.
* **Clipping Rate**: Identifies digital distortion from audio being recorded too loud, catching "blown-out" files where speech data is permanently destroyed.
* **Waveform Kurtosis**: Detects sudden, extreme audio spikes, preventing the model from mistaking mic pops or table bumps for hard consonants.
* **Spectral Flatness**: Differentiates structured human speech from pure white noise, filtering out files that are mostly drone sounds or static.
* **ZCR (Zero-Crossing Rate)**: Measures sound wave vibration frequency to catch excessive hissing or high-pitched electronic whines from faulty equipment.
* **Silence Ratio**: Calculates the percentage of absolute dead air to flag recordings with excessively long pauses.

This [threshold_decision.py](./threshold_decision.py) takes the samples of 4000 audio (~10% of total audio samples from 8 languages) from each language and resamples them to 16KHz for mathematical comparison.  It chops the audio into 30ms frame and calculates 8 distinct acoustic 
metrics.  Use Parallel processing to handle thousands of files rapidly.

It uses 6 Analytical plots - Boxplots, Histogram, Retention Curves, Heatmaps, IQR spread and Radar chart are applied on each of the 8 metrics and for each language are plotted and plots are saved.  It are made to help me decide between Global vs Per-language quality thresholds.


### Key Inferences from Acoustic Visualizations

#### A. Global vs. Per-Language Behavior
* **Visual Evidence:** [Boxplots per Language](./Decision_plots_and_csv/01_boxplots_per_language.png) & [Histograms](./Decision_plots_and_csv/02_histograms_all_languages.png)
* **Insight:** Metrics like `Waveform Kurtosis`, `Spectral Flatness`, and `ELR` are structurally stable across all languages (CV < 0.25). However, **VAD Ratio** (Voice Activity) and **SNR** (Signal-to-Noise) vary wildly. For instance, applying a global VAD threshold of >0.6 would preserve almost all Tamil data but delete over 40% of Hindi data.
* **Decision:**  I used **Per-Language** thresholds and normalization for VAD Ratio and SNR, while relying on **Global** thresholds for the remaining metrics.

#### B. Redundant Metrics & Hard Failures
* **Visual Evidence:** [Metric Correlation Heatmap](./Decision_plots_and_csv/04_correlation_heatmap.png)
* **Insight:** `VAD Ratio` and `Silence Ratio` are almost perfectly inversely correlated ($r = -0.91). Using both over-penalizes the exact same audio flaw. Additionally, `Clipping Rate` has roughly zero correlation with acoustic metrics because it is a hardware-level digital error.
* **Decision:** Completely drop `Silence Ratio` from the evaluation score. Treat `Clipping Rate` as a binary "Hard Reject" rather than a weighted metric.

#### C. Language-Specific Quirks
* **Visual Evidence:** [Language Quality Fingerprints (Radar)](./Decision_plots_and_csv/06_radar_language_profiles.png) & [Retention Curves](./Decision_plots_and_csv/03_retention_curves.png)
* **Insight:** Bengali exhibits a uniquely high Zero-Crossing Rate (ZCR) compared to the median dataset shape. 
* **Decision:** ZCR must be moved to the **Per-Language** normalization group so Bengali audio is only evaluated against its own linguistic baseline.

### The Final Curation Pipeline (Soft Score)

Based on the inferences above, I processed and filtered the audio in three distinct steps:

#### Step 1: The Hard Filters (Immediate Rejection)
Before calculating any complex scores, drop audio:
**Drop if** `duration > 0.3`
* **Drop if** `clipping_rate > 0.05` (Heavy digital distortion / blown-out audio).
* **Drop if** `vad_ratio < lang_median_vad × 0.4` (Essentially dead air; negligible human speech).

#### Step 2: Hybrid Normalization (0.0 to 1.0)
To combine different units (dB, percentages, raw floats) into a single score, normalize all metrics to a `0.0` (Worst) to `1.0` (Best) scale. 
* **Per-Language Normalization:** `vad_ratio`, `snr_db`, and `zcr`. Required because these metrics have high natural variance across languages (CV > 0.25). A global scale would unfairly penalize and disproportionately delete data from outlier languages like Hindi and Bengali.

* **Global Normalization:** `c50_db` (ELR), `spectral_flatness`, and `kurtosis`.  These metrics showed structural stability across all languages (CV < 0.25), meaning a reverberant recording sounds equally degraded regardless of the language being spoken. Normalizing these globally is both correct and efficient.

#### Step 3: The Weighted Soft Score(Intially, not tuned)
Calculating final Audio Quality Soft Score using the following weights:

| Metric | Weight | Importance | Normalization Type |
| :--- | :--- | :--- | :--- |
| **SNR (WADA)** | **40%** | Background noise is the #1 cause of model hallucinations. | **Per-Language** |
| **VAD Ratio** | **30%** | Ensures the model trains on actual phonemes, not silence. | **Per-Language** |
| **ELR (c50_db)**| **10%** | Rejects heavily reverberant (echoey) room recordings. | Global |
| **Spectral Flatness** | **10%** | Identifies pure white-noise files. | Global |
| **ZCR** | **5%** | Minor penalty for excessive hissing/static. | **Per-Language** |
| **Kurtosis** | **5%** | Minor penalty for sudden mic pops/clicks. | Global |


#### Step 4: The Weighted Soft Score(Tuned - Finalised values)
Running above configuration on 400 samples per language across all 8 languages (3,200 files
total) produced the following result:

- **Hard rejected:** ~12 files (clipping or near-silent)
- **Soft rejected:** ~41 files (below soft score threshold)
- **Total removed:** ~53 files (2.2% rejection rate)

This rejection rate is not accpetable and might not be true as when I manually verified samples and through plots around 10% of the audio samples were bad. Insights from plot from [plot_after_remove](plot_after_remove.py) is derived: 

- **[Plot 1 (Score Distribution)](./Phase1_filter/Validation_plot/plot1_score_distribution.png):** Confirmed the initial 0.40 threshold was safe, as the vast majority of files comfortably clustered between 0.65 and 0.90.

- **[Plot 2 (Passed vs. Rejected Boxplots)](./Phase1_filter/Validation_plot/plot2_metric_boxplot_pass_vs_reject.png):** Proved that SNR and VAD were the strongest discriminators for filtering bad audio, while metrics like spectral flatness had minimal impact.

- **[Plot 3 (Per-Language Rejection)](./Phase1_filter/Validation_plot/plot3_per_language_rejection.png):** Validated that per-language normalization worked perfectly, ensuring no single language was disproportionately penalized.

- **[Plot 4 (Soft Score CDF)](./Phase1_filter/Validation_plot/plot4_score_cdf.png):** Revealed a natural quality cliff around 0.50, pinpointing the exact boundary where acoustic quality genuinely drops.

And these are the final configuration:

```python
SOFT_SCORE_THRESHOLD = 0.50

CLIP_HARD_LIMIT      = 0.05
VAD_FLOOR_MULTIPLIER = 0.40

WEIGHTS = {
    "snr_db":           0.45,
    "vad_ratio":        0.30,
    "c50_db":           0.10,
    "spectral_flatness":0.05,
    "zcr":              0.05,
    "kurtosis":         0.05,
}
```



### Validation

The final configuration was validated against three criteria before being accepted:

**1. No language imbalance.** [Plot 3](./Phase1_filter/Validation_plot/plot3_per_language_rejection.png) confirmed that all 8 languages had rejection rates
within a narrow band. No language lost more than 2× the average rejection rate of the
corpus, confirming that per-language normalisation of VAD, SNR, and ZCR was working as
intended.

**2. Threshold at a natural quality boundary.** [Plot 4](./Phase1_filter//Validation_plot//plot4_score_cdf.png) confirmed the threshold of 0.50
sits at the inflection point of the score CDF — at a genuine quality cliff in the data
rather than an arbitrary cutoff.

**3. Rejected files are genuinely worse.** Manual inspection of 15 files just below the
threshold (scores 0.40–0.50) and 15 files just above it (scores 0.50–0.60) confirmed that
files below the threshold had audibly worse quality — background noise, low speech density,
or both — compared to files above it.

**4. Rejection Rate:** Currently we have the rejection rate ~5%, we need to pass the remaining data that passed this phase to our Whisper model and SpeechBrain model.

**5. The output is stored in [Filter_output](./Phase1_filter/) that contains [Rejected audio](./Phase1_filter/audio_rejected/), the [Hard Rejected audio](./Phase1_filter/hard_rejected.csv) and [Soft Rejected Audio](./Phase1_filter/soft_rejected.csv) and [Summary](./Phase1_filter/summary.json) how many samples accepted or rejected and the plots that shows the insights about data rejection inference per language - [Validation Plots](./Phase1_filter/Validation_plot/)

---


## Phase 4: Bonus Phase (ASR done using Whisper):

Phase 3 successfully filtered out audio with structural and acoustic flaws (noise, dead air, distortion), it could not verify if the audio actually matched the provided text. A file might have perfect studio quality but contain the wrong speaker, hallucinated text, or a completely mismatched transcript.  This phase somewhat resolves it, since it is Indic data, the model performance is mid.

This phase is a scalable, multi-stage audio filtering pipeline that uses Whisper transcription and Character Error Rate (CER) to evaluate and filter audio-text pairs.

This pipeline processes a dataset of audio samples which comes as output of phase 3(filtered_manifest.json) and determines whether each sample is valid or noisy/misaligned based on transcription quality.

`Audio File → Transcription (Whisper) → CER Computation → Filtering`

### Threshold tuning and inference:
* Initially the **CER Threshold was set to 0.3**, and less than **1%** of the audio is accepted.  This is because the model can't capture fast spoken or incomplete words from the audio, this rejects lot of samples.

* Then I tuned the value and listened to the accepted and rejected audio and finalised **CER threshold to 0.7.**


### Reason for High CER Leniency:

* Model vs. Language Complexity: I am using faster-whisper, a general-purpose multilingual model, on highly diverse Indic language datasets. Whisper's baseline accuracy for Indic scripts is naturally lower than for English due to complex tokenization, regional dialects, and frequent code-mixing (e.g. using English loan words in Hindi or Tamil).

* Acoustic Validity: Manual inspection and visual plot confirmed that files scoring between 0.30 and 0.70 CER were actually perfectly valid, high-quality human speech. Whisper was simply making spelling errors, dropping fast phonemes, or struggling with localized pronunciations. 

### Rejection rate:
* Leniency of 0.7 CER is set, the model **rejects 40%** of the bad audio samples and keeps **60%** back.  
* Through manual verification of removed samples it is working perfectly that transcripts don't accurately match for noisy and unclear audio.
* The Whisper model did a good job in filtering out bad samples with a decent CER threshold.


### Scalability Approach:

### 1. Hybrid Concurrency for Data Ingestion
* **Optimized I/O and Compute:** The dataset downloader utilizes a hybrid approach, combining a `ProcessPoolExecutor` to parse large `.parquet` files and a nested `ThreadPoolExecutor` to handle the I/O-bound extraction of audio bytes to disk.

### 2. OS-Safe Acoustic Parallelization (Phase 3)
* **Maximized CPU Math:** The acoustic feature extraction scripts (Soft Score and Threshold Profiling) utilize `ProcessPoolExecutor` to run heavy signal processing (FFTs, RMS energy) across thousands of files simultaneously.

### 3. The "Zero-Idle" 3-Stage Whisper Pipeline (Phase 4)
The semantic filtration script resolves the classic GPU-starvation bottleneck by breaking transcription into three isolated, asynchronous stages:
* **Stage 1 (I/O Prefetching):** A dedicated thread pool continuously reads and resamples `.flac` files into RAM, bypassing the Global Interpreter Lock (GIL).
* **Stage 2 (GPU Transcription):** The main thread pulls pre-loaded numpy arrays directly from RAM to VRAM, ensuring the GPU is never stalled waiting for a disk read.
* **Stage 3 (CPU Post-Processing):** While the GPU processes the next batch, a separate process pool calculates the CPU-heavy Character Error Rate (Levenshtein distance) for the previous batch.

---

## Key Insights and Final Inference:

* The Indic dataset consist of lot of low-mid samples which is found using manual listening to sample of audio and found out most prominent problems in audio samples.
* Developed a hybrid Soft Score pipeline, applying per-language normalization to highly variable metrics (VAD, SNR) and global normalization to structural flaws.
* Set the rejection threshold to a inflection point (0.50), safely pruning dead air and toxic noise without erasing naturally slower or hissier languages.
* Applying a global Voice Activity (VAD) threshold of > 0.60 would preserve almost all Tamil data but unfairly delete over 40% of Hindi data, proving the absolute necessity of per-language normalization.
* Shifting the baseline Soft Score rejection threshold from 0.40 to the CDF's natural quality cliff at 0.50 corrected an artificially low 2.2% rejection rate to a more accurate ~5% rejection rate.
* Calibrated the Character Error Rate (CER) threshold to 0.70, purging catastrophic transcript mismatches while forgiving Whisper's baseline struggles with complex Indic dialects.
* **The Final Impact:** The two-phase pipeline guarantees a sanitized, densely voiced, and accurately transcribed dataset that is strictly balanced and primed for robust ASR training.

---