import argparse
import json
import logging
import multiprocessing
import random
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import kurtosis as scipy_kurtosis
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("soft_score_filter")



ROOT              = "./data/audios"
OUTPUT_DIR        = "./Phase1_filter"
LANGUAGES         = ["hindi", "tamil", "telugu", "kannada",
                     "malayalam", "bengali", "gujarati", "marathi"]
N_SAMPLES         = None     
SOFT_SCORE_THRESHOLD = 0.5 
RANDOM_SEED       = 42

CLIP_HARD_LIMIT       = 0.05     
VAD_FLOOR_MULTIPLIER  = 0.40     

TARGET_SR    = 16000
FRAME_MS     = 30
SILENCE_DB   = -40.0
MIN_DURATION = 0.3
EPS          = 1e-10

WEIGHTS = {
    "snr_db":           0.45,
    "vad_ratio":        0.30,
    "c50_db":           0.10,
    "spectral_flatness":0.05,
    "zcr":              0.05,
    "kurtosis":         0.05,
}

PER_LANGUAGE = {
    "snr_db":           True,
    "vad_ratio":        True,
    "zcr":              True,
    "c50_db":           False,
    "spectral_flatness":False,
    "kurtosis":         False,
}

AUDIO_EXT = {".flac"}



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


def copy_files(df: pd.DataFrame, dest_dir: Path, label: str):

    dest_dir.mkdir(parents=True, exist_ok=True)
    copied, failed = 0, 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {label}", unit="file"):
        src  = Path(row["path"])
        lang = row.get("language", "unknown")
        dst  = dest_dir / lang / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
            copied += 1
        except Exception as e:
            log.debug("Copy failed %s: %s", src, e)
            failed += 1
    log.info("%s — copied: %d  failed: %d → %s", label, copied, failed, dest_dir)
    
    
def frame_signal(signal, sr):
    frame_len = int(sr * FRAME_MS / 1000)
    n_frames  = len(signal) // frame_len
    if n_frames == 0:
        return signal[np.newaxis, :]
    return signal[:n_frames * frame_len].reshape(n_frames, frame_len)


def compute_zcr(signal, sr):
    frames = frame_signal(signal, sr)
    return float(np.mean(
        np.mean(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / 2
    ))


def compute_clipping_rate(signal, threshold=0.9999):
    return float(np.mean(np.abs(signal) >= threshold))


def compute_kurtosis(signal):
    return float(np.clip(scipy_kurtosis(signal, fisher=False), 0, 50))


def compute_spectral_flatness(signal, sr, n_fft=512):
    hop      = n_fft // 2
    window   = np.hanning(n_fft)
    n_frames = max(1, (len(signal) - n_fft) // hop)
    vals = []
    for i in range(n_frames):
        frame = signal[i*hop : i*hop + n_fft] * window
        power = np.abs(np.fft.rfft(frame)) ** 2 + EPS
        vals.append(np.exp(np.mean(np.log(power))) / np.mean(power))
    return float(np.clip(np.mean(vals), 0, 1))


def compute_snr_wada(signal, sr):
    frames      = frame_signal(signal, sr)
    power       = np.mean(frames ** 2, axis=1) + EPS
    sorted_pow  = np.sort(power)
    n_noise     = max(int(len(sorted_pow) * 0.20), 1)
    noise_power = np.mean(sorted_pow[:n_noise])
    sig_power   = np.mean(power)
    return float(np.clip(10 * np.log10(sig_power / noise_power), -5, 60))


def compute_elr(signal, sr):
    t50 = int(0.050 * sr)
    if len(signal) <= t50:
        return 0.0
    early = np.sum(signal[:t50] ** 2) + EPS
    late  = np.sum(signal[t50:] ** 2) + EPS
    return float(10 * np.log10(early / late))


def compute_vad_ratio(signal, sr):
    frames    = frame_signal(signal, sr)
    rms_db    = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1) + EPS))
    zcr_per   = np.mean(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) / 2
    energy_th = np.percentile(rms_db, 20)
    voiced    = (rms_db > energy_th) & (rms_db > SILENCE_DB) & (zcr_per < 0.3)
    return float(np.mean(voiced))


def compute_metrics(path: str) -> dict | None:
    try:
        signal, sr = load_mono(path)
    except Exception:
        return None

    if len(signal) / sr < MIN_DURATION:
        return None

    return {
        "path":             path,
        "duration_s":       round(len(signal) / sr, 3),
        "zcr":              compute_zcr(signal, sr),
        "clipping_rate":    compute_clipping_rate(signal),
        "kurtosis":         compute_kurtosis(signal),
        "spectral_flatness":compute_spectral_flatness(signal, sr),
        "snr_db":           compute_snr_wada(signal, sr),
        "c50_db":           compute_elr(signal, sr),
        "vad_ratio":        compute_vad_ratio(signal, sr),
    }



def collect_files(root: Path, language: str) -> list:
    lang_dir = root / language
    if lang_dir.exists():
        return [str(p) for p in lang_dir.rglob("*")
                if p.suffix.lower() in AUDIO_EXT]
    return [str(p) for p in root.rglob("*")
            if p.suffix.lower() in AUDIO_EXT and p.parts[-3] == language]



def collect_metrics(root: Path, languages: list, n_samples, n_workers: int) -> pd.DataFrame:
    lang_files = {}
    for lang in languages:
        all_files = collect_files(root, lang)
        if not all_files:
            log.warning("No files found for language: %s", lang)
            continue
        random.seed(RANDOM_SEED)
        sampled = (random.sample(all_files, min(n_samples, len(all_files)))
           if n_samples is not None else all_files)
        lang_files[lang] = sampled
        log.info("[%s] %d / %d files queued", lang, len(sampled), len(all_files))

    total = sum(len(v) for v in lang_files.values())
    log.info("Total files to process: %d  |  Workers: %d", total, n_workers)

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
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
                except Exception:
                    pass
                finally:
                    pbar.update(1)

    log.info("Metrics computed: %d rows", len(rows))
    return pd.DataFrame(rows)



def apply_hard_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rejected_rows = []

    clip_mask = df["clipping_rate"] > CLIP_HARD_LIMIT
    clip_rej  = df[clip_mask].copy()
    clip_rej["hard_reject_reason"] = (
        "clipping_rate=" + clip_rej["clipping_rate"].round(4).astype(str)
        + " > " + str(CLIP_HARD_LIMIT)
    )
    rejected_rows.append(clip_rej)
    df = df[~clip_mask].copy()

    lang_median_vad = df.groupby("language")["vad_ratio"].median()
    df["_vad_floor"] = df["language"].map(lang_median_vad) * VAD_FLOOR_MULTIPLIER
    vad_mask = df["vad_ratio"] < df["_vad_floor"]
    vad_rej  = df[vad_mask].copy()
    vad_rej["hard_reject_reason"] = (
        "vad_ratio=" + vad_rej["vad_ratio"].round(4).astype(str)
        + " < adaptive_floor=" + vad_rej["_vad_floor"].round(4).astype(str)
    )
    rejected_rows.append(vad_rej)
    df = df[~vad_mask].drop(columns=["_vad_floor"]).copy()

    rejected = pd.concat(rejected_rows, ignore_index=True)
    rejected = rejected.drop(columns=["_vad_floor"], errors="ignore")
    log.info("Hard filter — passed: %d  |  rejected: %d", len(df), len(rejected))
    return df, rejected



def _minmax(series: pd.Series, higher_is_better: bool) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < EPS:
        return pd.Series(1.0, index=series.index)
    norm = (series - lo) / (hi - lo)
    return norm if higher_is_better else (1 - norm)


HIGHER_IS_BETTER = {
    "snr_db":           True,
    "vad_ratio":        True,
    "c50_db":           True,
    "spectral_flatness":False,
    "zcr":              False,
    "kurtosis":         False,
}


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    norm_df = df.copy()
    score_metrics = list(WEIGHTS.keys())

    for metric in score_metrics:
        hib = HIGHER_IS_BETTER[metric]
        col = f"norm_{metric}"

        if PER_LANGUAGE[metric]:
            norm_df[col] = (
                norm_df.groupby("language")[metric]
                       .transform(lambda s: _minmax(s, hib))
            )
        else:
            norm_df[col] = _minmax(norm_df[metric], hib)

    return norm_df



def compute_soft_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["soft_score"] = sum(
        df[f"norm_{metric}"] * weight
        for metric, weight in WEIGHTS.items()
    )
    return df


def apply_soft_threshold(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    passed   = df[df["soft_score"] >= SOFT_SCORE_THRESHOLD].copy()
    rejected = df[df["soft_score"] <  SOFT_SCORE_THRESHOLD].copy()
    rejected["soft_reject_reason"] = (
        "soft_score=" + rejected["soft_score"].round(4).astype(str)
        + " < threshold=" + str(SOFT_SCORE_THRESHOLD)
    )

    log.info("Soft threshold — passed: %d  |  rejected: %d", len(passed), len(rejected))

    for lang in sorted(df["language"].unique()):
        p = len(passed[passed["language"] == lang])
        r = len(rejected[rejected["language"] == lang])
        t = p + r
        log.info("  [%s]  kept: %d / %d  (%.1f%%)", lang, p, t, 100*p/max(t,1))

    return passed, rejected


METRIC_COLS = [
    "path", "language", "duration_s",
    "clipping_rate", "vad_ratio", "snr_db", "c50_db",
    "spectral_flatness", "zcr", "kurtosis",
    "soft_score",
]

def save_outputs(passed: pd.DataFrame, hard_rej: pd.DataFrame,
                 soft_rej: pd.DataFrame, out: Path):
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = out / "filtered_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for _, row in passed.iterrows():
            f.write(json.dumps({"path": row["path"],
                                "language": row["language"],
                                "normalized": row["unsanitized_normalized"],
                                "soft_score": round(row["soft_score"], 4)},
                               ensure_ascii=False) + "\n")
    log.info("Filtered manifest → %s  (%d files)", manifest_path, len(passed))

    metrics_cols = [c for c in METRIC_COLS if c in passed.columns]
    all_scored = pd.concat([passed, soft_rej], ignore_index=True)
    all_scored[metrics_cols + ["soft_score"]].to_csv(
        out / "all_metrics.csv", index=False
    )
    log.info("All metrics CSV → %s", out / "all_metrics.csv")

    hard_out_cols = ["path", "language", "duration_s",
                     "clipping_rate", "vad_ratio", "hard_reject_reason"]
    hard_out_cols = [c for c in hard_out_cols if c in hard_rej.columns]
    hard_rej[hard_out_cols].to_csv(out / "hard_rejected.csv", index=False)
    log.info("Hard rejected CSV → %s  (%d files)", out / "hard_rejected.csv", len(hard_rej))

    soft_out_cols = metrics_cols + ["soft_score", "soft_reject_reason"]
    soft_out_cols = [c for c in soft_out_cols if c in soft_rej.columns]
    soft_rej[soft_out_cols].to_csv(out / "soft_rejected.csv", index=False)
    log.info("Soft rejected CSV → %s  (%d files)", out / "soft_rejected.csv", len(soft_rej))

    all_rejected = pd.concat([hard_rej, soft_rej], ignore_index=True)
    copy_files(all_rejected, out / "audio_rejected",  "rejected")
    summary = {
        "total_processed":   len(passed) + len(hard_rej) + len(soft_rej),
        "hard_rejected":     len(hard_rej),
        "soft_rejected":     len(soft_rej),
        "passed":            len(passed),
        "soft_threshold":    SOFT_SCORE_THRESHOLD,
        "weights":           WEIGHTS,
        "per_language": {
            lang: {
                "passed":  int((passed["language"] == lang).sum()),
                "rejected": int(
                    ((hard_rej["language"] == lang).sum() if "language" in hard_rej.columns else 0)
                    + (soft_rej["language"] == lang).sum()
                ),
            }
            for lang in sorted(passed["language"].unique())
        }
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary JSON → %s", out / "summary.json")

def merge_transcripts_from_manifests(df: pd.DataFrame, root_dir: Path) -> pd.DataFrame:
   
    log.info("Dynamically fetching transcripts from individual manifest files...")
    
  
    data_dir = root_dir.parent
    manifests_dir = data_dir / "manifests"
    
    path_to_text = {}
    
    df["_batch_name"] = df["path"].apply(lambda p: Path(p).parent.name)
    
    grouped = df.groupby(["language", "_batch_name"])
    
    with tqdm(total=len(grouped), desc="Fetching Transcripts", unit="batch") as pbar:
        for (lang, batch_name), group_df in grouped:
            manifest_file = manifests_dir / f"{lang}_manifests" / f"{batch_name}.jsonl"
            
            if not manifest_file.exists():
                log.warning("Manifest not found: %s", manifest_file)
                pbar.update(1)
                continue
                
            batch_lookup = {}
            with open(manifest_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        a_path = str(Path(data["audio_filepath"])).replace("\\", "/")
                        batch_lookup[a_path] = data.get("unsanitized_normalized", "")
                    except Exception:
                        pass
            
            for _, row in group_df.iterrows():
                norm_p = str(Path(row["path"])).replace("\\", "/")
                path_to_text[norm_p] = batch_lookup.get(norm_p, "")
                
            pbar.update(1)

    df["path_norm"] = df["path"].apply(lambda p: str(Path(p)).replace("\\", "/"))
    df["unsanitized_normalized"] = df["path_norm"].map(path_to_text).fillna("")
    
    df = df.drop(columns=["_batch_name", "path_norm"])
    
    return df


def main(args):

    root      = Path(args.root)
    out       = Path(args.output)
    n_workers = args.workers
    threshold = args.threshold

    global SOFT_SCORE_THRESHOLD
    SOFT_SCORE_THRESHOLD = threshold

    log.info("Root      : %s", root)
    log.info("Output    : %s", out)
    log.info("Languages : %s", LANGUAGES)
    log.info("Samples   : %s", args.samples or "ALL")
    log.info("Workers   : %d", n_workers)
    log.info("Threshold : %.2f", threshold)

    df = collect_metrics(root, LANGUAGES, args.samples, n_workers)
    if df.empty:
        log.error("No files found. Check --root path.")
        return
    
    
    df = merge_transcripts_from_manifests(df, root)
    
    df_passed, hard_rejected = apply_hard_filters(df)

    df_norm = normalise(df_passed)

    df_scored = compute_soft_score(df_norm)

    passed, soft_rejected = apply_soft_threshold(df_scored)

    save_outputs(passed, hard_rejected, soft_rejected, out)
    
    log.info("Done.")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="Soft-score audio quality filter for Indic speech datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root",      default=ROOT,
                        help="Root directory containing language subdirs")
    parser.add_argument("--output",    default=OUTPUT_DIR,
                        help="Output directory for CSVs and manifest")
    parser.add_argument("--samples", type=int, default=None,
                    help="Files to sample per language. Omit or set to 0 for all files.")
    parser.add_argument("--threshold", type=float, default=SOFT_SCORE_THRESHOLD,
                        help="Soft score threshold — files below this are rejected")
    parser.add_argument("--workers",   type=int,
                        default=max(1, multiprocessing.cpu_count() - 1),
                        help="Number of parallel worker processes")
    parser.add_argument("--manifest", type=str, default="./combined_manifest.jsonl",
                        help="Path to the original manifest to fetch the text column")
    args = parser.parse_args()
    main(args)