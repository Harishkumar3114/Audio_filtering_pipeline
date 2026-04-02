import os
import logging
import argparse
import polars as pl
from tqdm import tqdm
from huggingface_hub import snapshot_download
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing


_CORES        = multiprocessing.cpu_count()
N_PROCS       = max(1, _CORES - 1)
N_THREADS     = max(2, _CORES // 4)
N_DL_WORKERS  = max(4, _CORES * 2)
TARGET_LANGUAGES = ['tamil', 'telugu', 'hindi', 'bengali', 'kannada', 'malayalam', 'marathi', 'gujarati']

def setup_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger("IndicvoicesSetup")
    logger.setLevel(log_level)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}\n")

    return logger


def process_row(row, dest_dir):
    try:
        audio_data = row['audio_filepath']['bytes']
        audio_name = row['audio_filepath']['path']

        del row['audio_filepath']

        save_path = os.path.join(dest_dir, audio_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "wb") as f:
            f.write(audio_data)

        entry = {
            'audio_filepath': save_path,
            **row
        }
        return True, entry

    except Exception as e:
        return False, f"Error processing sample: {e}"


def process_parquet(parquet_file, lang, audio_save_dir, manifest_save_dir, log_file):
 
    logger = logging.getLogger(f"IndicvoicesSetup.{lang}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    index    = os.path.basename(parquet_file).split('.')[0]
    dest_dir = os.path.join(audio_save_dir, lang, index)
    os.makedirs(dest_dir, exist_ok=True)

    def log(msg):       logger.info(f"{lang} - {os.path.basename(parquet_file)} - {msg}")
    def log_err(msg):   logger.error(f"{lang} - {os.path.basename(parquet_file)} - {msg}")

    log("Processing...")

    try:
        df = pl.read_parquet(parquet_file)
    except Exception as e:
        log_err(f"Failed to read parquet: {e}")
        return []

    manifest = []
    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = [
            executor.submit(process_row, row, dest_dir)
            for row in df.iter_rows(named=True)
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"{lang}/{os.path.basename(parquet_file)}",
            leave=False
        ):
            success, result = future.result()
            if success:
                manifest.append(result)
            else:
                log_err(result)

    log(f"Processed {len(manifest)}/{len(df)} rows successfully")

    if manifest:
        manifest_df = pl.DataFrame(manifest)
        save_path = os.path.join(
            manifest_save_dir, f'{lang}_manifests', f"{index}.jsonl"
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        manifest_df.write_ndjson(save_path)
        log(f"Manifest saved → {save_path}")

    return manifest


def build_allow_patterns(languages):

    patterns = []
    for lang in languages:
        patterns.append(f"{lang}/valid*.parquet")
        patterns.append(f"{lang}/train-0000[0-5]*.parquet")
    return patterns


def main(save_dir: str):
    HF_SAVE_DIR       = os.path.join(save_dir, 'hf')
    AUDIO_SAVE_DIR    = os.path.join(save_dir, 'audios')
    MANIFEST_SAVE_DIR = os.path.join(save_dir, 'manifests')

    for d in [save_dir, HF_SAVE_DIR, AUDIO_SAVE_DIR, MANIFEST_SAVE_DIR]:
        os.makedirs(d, exist_ok=True)

    log_file = os.path.join(save_dir, 'setup.log')
    if os.path.exists(log_file):
        os.remove(log_file)

    logger = setup_logger(log_file=log_file)
    logger.info(f"Target languages : {TARGET_LANGUAGES}")
    logger.info(f"CPU config       : {N_PROCS} processes × {N_THREADS} threads")
    logger.info(f"Directories      : {save_dir}")

    patterns = build_allow_patterns(TARGET_LANGUAGES)
    logger.info(f"Downloading {len(TARGET_LANGUAGES)} languages from HuggingFace...")
    logger.info(f"Patterns: {patterns}")

    snapshot_download(
        repo_id="ai4bharat/IndicVoices",
        repo_type="dataset",
        local_dir=HF_SAVE_DIR,
        local_dir_use_symlinks=False,
        max_workers=N_DL_WORKERS,  
        resume_download=True,
        allow_patterns=patterns
    )
    logger.info(f"Download complete → {HF_SAVE_DIR}\n")

    all_tasks = []   

    for lang in TARGET_LANGUAGES:
        lang_dir = os.path.join(HF_SAVE_DIR, "data", lang)
        if not os.path.isdir(lang_dir):
            lang_dir = os.path.join(HF_SAVE_DIR, lang)

        if not os.path.isdir(lang_dir):
            logger.warning(f"Directory not found for lang={lang}, skipping")
            continue

        parquet_files = [
            os.path.join(lang_dir, f)
            for f in os.listdir(lang_dir)
            if f.endswith(".parquet")
        ]

        if not parquet_files:
            logger.warning(f"No parquet files found for lang={lang}")
            continue

        logger.info(f"Found {len(parquet_files):>3} parquet files for [{lang}]")
        for pf in parquet_files:
            all_tasks.append((pf, lang))

    logger.info(f"Total parquet files to process: {len(all_tasks)}")


    combined_manifest = []

    with ProcessPoolExecutor(max_workers=N_PROCS) as executor:
        futures = {
            executor.submit(
                process_parquet,
                pf, lang,
                AUDIO_SAVE_DIR,
                MANIFEST_SAVE_DIR,
                log_file          
            ): (pf, lang)
            for pf, lang in all_tasks
        }

        with tqdm(total=len(futures), desc="Overall progress") as pbar:
            for future in as_completed(futures):
                pf, lang = futures[future]
                try:
                    result = future.result()
                    combined_manifest.extend(result)
                    pbar.set_postfix({"last": f"{lang}/{os.path.basename(pf)}"})
                except Exception as e:
                    logger.error(f"Failed: {lang}/{os.path.basename(pf)} — {e}")
                finally:
                    pbar.update(1)

    logger.info(f"Total samples collected: {len(combined_manifest)}")

    if combined_manifest:
        combined_df = pl.DataFrame(combined_manifest)
        save_path   = os.path.join(MANIFEST_SAVE_DIR, "combined_manifest.jsonl")
        combined_df.write_ndjson(save_path)
        logger.info(f"Combined manifest saved → {save_path}")

        print("\n── Per-language summary ──────────────────────────")
        lang_col = "lang" if "lang" in combined_df.columns else None
        if lang_col:
            summary = combined_df.group_by(lang_col).len().sort(lang_col)
            print(summary)
        print(f"Total: {len(combined_df)} samples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndicVoices dataset setup")
    parser.add_argument("--save_dir", type=str, default="./data",
                        help="Root directory to save audios and manifests")
    args = parser.parse_args()
    main(save_dir=args.save_dir)