import json, logging, os, re, warnings, argparse, multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import List, Tuple, Optional
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import soundfile as sf
import librosa
from faster_whisper import WhisperModel

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("Phase2_filter")

MANIFEST_PATH  = "./Phase1_filter/filtered_manifest.jsonl"
OUTPUT_DIR     = "./Phase2_filter"
CER_THRESHOLD  = 0.7
MIN_SOFT_SCORE = 0.50
SAMPLE_RATE    = 16000   

LANG_MAP = {
    "hindi": "hi", "tamil": "ta", "telugu": "te", "kannada": "kn",
    "malayalam": "ml", "bengali": "bn", "gujarati": "gu", "marathi": "mr",
}

MODEL_VRAM_GB = {
    "tiny": 1.0, "base": 1.0, "small": 2.0,
    "medium": 5.0, "large-v2": 10.0, "large-v3": 10.0,
}

_DONE = object()


def build_hw_profile(model_size: str) -> dict:

    cpu_cores    = max(1, (os.cpu_count() or 4))
    is_windows   = os.name == "nt"
    mp_context   = "spawn" if is_windows else "fork"

    profile = {
        "cpu_cores":    cpu_cores,
        "mp_context":   mp_context,
        "device":       "cpu",
        "compute_type": "int8",
        "batch_size":   4,
        "io_workers":   min(4, max(1, cpu_cores // 4)),  
        "cer_workers":  max(1, cpu_cores // 2),            
        "io_q_depth":   8,    
        "cer_q_depth":  8,   
    }

    if not torch.cuda.is_available():
        log.warning("No GPU — CPU mode. Using int8, small batches.")
        profile["io_workers"] = min(2, cpu_cores)
        return profile

    props       = torch.cuda.get_device_properties(0)
    total_vram  = props.total_memory / (1024 ** 3)
    usable_vram = total_vram * 0.85
    model_need  = MODEL_VRAM_GB.get(model_size.split(".")[0], 10.0)
    free_vram   = usable_vram - model_need

    profile["device"] = "cuda"

    if model_need > usable_vram:
        profile["compute_type"] = "int8_float16"
        free_vram = usable_vram - (model_need / 2)
        log.warning("VRAM tight: using int8_float16 (model ~%.1fGB, have %.1fGB)",
                    model_need, usable_vram)
    else:
        profile["compute_type"] = "float16"

    if free_vram >= 8:
        profile["batch_size"] = 32
    elif free_vram >= 4:
        profile["batch_size"] = 16
    elif free_vram >= 2:
        profile["batch_size"] = 8
    else:
        profile["batch_size"] = 4

    profile["io_workers"] = min(profile["batch_size"], max(2, cpu_cores // 2))

    profile["io_q_depth"]  = profile["batch_size"] * 2
    profile["cer_q_depth"] = profile["batch_size"] * 2

    log.info(
        "HW Profile | GPU: %s (%.1fGB) | compute: %s | "
        "batch: %d | io_workers: %d | cer_workers: %d | mp: %s",
        props.name, total_vram,
        profile["compute_type"],
        profile["batch_size"],
        profile["io_workers"],
        profile["cer_workers"],
        mp_context,
    )
    return profile


def _load_one(row: dict) -> Optional[dict]:
    
    try:
        try:
            audio, sr = sf.read(row["path"], dtype="float32", always_2d=False)
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        except Exception:
            audio, _ = librosa.load(row["path"], sr=SAMPLE_RATE, mono=True)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        return {**row, "_audio": audio.astype(np.float32)}
    except Exception as e:
        log.warning("I/O error on %s: %s", row["path"], e)
        return None


def io_stage(rows: List[dict], hw: dict, audio_q: Queue):

    batch_size = hw["batch_size"]
    n_workers  = hw["io_workers"]

    def make_batches(data, size):
        for i in range(0, len(data), size):
            yield data[i:i + size]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for batch in make_batches(rows, batch_size):
            futures = {pool.submit(_load_one, row): row for row in batch}
            loaded  = []
            for fut in as_completed(futures):
                result = fut.result()
                if result is not None:
                    loaded.append(result)

            if loaded:
                loaded.sort(key=lambda r: r.get("_orig_idx", 0))
                audio_q.put(loaded)   

    audio_q.put(_DONE)   


def gpu_stage(model: WhisperModel, audio_q: Queue, cer_q: Queue):
    
    while True:
        try:
            batch = audio_q.get(timeout=30)
        except Empty:
            log.warning("GPU stage timed out waiting for audio. Shutting down.")
            break

        if batch is _DONE:
            break

        for row in batch:
            audio = row.pop("_audio")  
            lang  = LANG_MAP.get(row.get("language", ""), "hi")
            try:
                segments, _ = model.transcribe(
                    audio,                         
                    language=lang,
                    beam_size=1,                    
                    without_timestamps=True,
                    condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters=dict(
                        min_silence_duration_ms=300,
                        speech_pad_ms=100,
                    ),
                )
                row["whisper_hyp"] = " ".join(s.text for s in segments).strip()
            except Exception as e:
                log.warning("Transcription error on %s: %s", row.get("path", "?"), e)
                row["whisper_hyp"] = ""

        cer_q.put(batch)
        audio_q.task_done()

    cer_q.put(_DONE)  


def _cer_worker(args: Tuple[str, str]) -> float:
    ref, hyp = _normalise(args[0]), _normalise(args[1])
    if not ref:
        return 0.0   
    try:
        import editdistance
        dist = editdistance.eval(ref, hyp)
    except ImportError:
        r, h   = list(ref), list(hyp)
        n, m   = len(r), len(h)
        prev   = list(range(m + 1))
        for i in range(1, n + 1):
            curr = [i] + [0] * m
            for j in range(1, m + 1):
                curr[j] = (prev[j-1] if r[i-1] == h[j-1]
                           else 1 + min(prev[j], curr[j-1], prev[j-1]))
            prev = curr
        dist = prev[m]
    return dist / max(len(ref), 1)


def _normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def cer_stage(cer_q: Queue, hw: dict, cer_threshold: float,
              passed: list, rejected: list):
    
    n_workers = hw["cer_workers"]
    ctx       = mp.get_context(hw["mp_context"])

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        while True:
            try:
                batch = cer_q.get(timeout=60)
            except Empty:
                log.warning("CER stage timed out. Shutting down.")
                break

            if batch is _DONE:
                break

            pairs   = [(r.get("normalized", ""), r.get("whisper_hyp", ""))
                       for r in batch]
            futures = {pool.submit(_cer_worker, p): r
                       for p, r in zip(pairs, batch)}

            for fut, row in futures.items():
                cer      = fut.result()
                row["cer"] = round(cer, 4)

                if not row.get("normalized"):
                    passed.append(row)
                elif cer > cer_threshold:
                    row["cer_reject_reason"] = f"CER={cer:.4f} > {cer_threshold}"
                    rejected.append(row)
                else:
                    passed.append(row)

            cer_q.task_done()


def run_pipeline(rows: List[dict], model: WhisperModel,
                 hw: dict, cer_threshold: float) -> Tuple[List[dict], List[dict]]:
    

    for i, row in enumerate(rows):
        row["_orig_idx"] = i

    io_q  = Queue(maxsize=hw["io_q_depth"])
    cer_q = Queue(maxsize=hw["cer_q_depth"])  

    passed:   List[dict] = []
    rejected: List[dict] = []

    cer_thread = Thread(
        target=cer_stage,
        args=(cer_q, hw, cer_threshold, passed, rejected),
        daemon=True,
        name="CER-Stage",
    )
    cer_thread.start()

    io_thread = Thread(
        target=io_stage,
        args=(rows, hw, io_q),
        daemon=True,
        name="IO-Stage",
    )
    io_thread.start()

    total = len(rows)
    with tqdm(total=total, desc="Transcribing", unit="file",
              dynamic_ncols=True) as pbar:

        def gpu_stage_with_progress():
            while True:
                try:
                    batch = io_q.get(timeout=30)
                except Empty:
                    log.warning("GPU stage timed out waiting for audio.")
                    break

                if batch is _DONE:
                    break

                for row in batch:
                    audio = row.pop("_audio")
                    lang  = LANG_MAP.get(row.get("language", ""), "hi")
                    try:
                        segments, _ = model.transcribe(
                            audio,
                            language=lang,
                            beam_size=1,
                            without_timestamps=True,
                            condition_on_previous_text=False,
                            vad_filter=True,
                            vad_parameters=dict(
                                min_silence_duration_ms=300,
                                speech_pad_ms=100,
                            ),
                        )
                        row["whisper_hyp"] = " ".join(s.text for s in segments).strip()
                    except Exception as e:
                        log.warning("Transcription error %s: %s", row.get("path", "?"), e)
                        row["whisper_hyp"] = ""

                cer_q.put(batch)
                pbar.update(len(batch))
                io_q.task_done()

            cer_q.put(_DONE)

        gpu_stage_with_progress()

    io_thread.join()
    cer_thread.join()

    passed.sort(key=lambda r: r.pop("_orig_idx", 0))
    rejected.sort(key=lambda r: r.pop("_orig_idx", 0))

    return passed, rejected


def read_manifest(path: str) -> List[dict]:
    rows, dropped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if d.get("soft_score", 0) > MIN_SOFT_SCORE:
                    rows.append(d)
                else:
                    dropped += 1
            except json.JSONDecodeError:
                pass
    log.info("Manifest: %d passed soft_score gate | %d dropped", len(rows), dropped)
    return rows


def save_outputs(passed: List[dict], rejected: List[dict],
                 out: Path, model_size: str):
    out.mkdir(parents=True, exist_ok=True)
    skip_keys = {"cer_reject_reason", "_orig_idx"}

    passed_file = out / "passed.jsonl"
    with open(passed_file, "w", encoding="utf-8") as f:
        for r in passed:
            f.write(json.dumps(
                {k: v for k, v in r.items() if k not in skip_keys},
                ensure_ascii=False) + "\n")
    log.info("Passed  → %s (%d)", passed_file, len(passed))

    if rejected:
        rej_file = out / "cer_rejected.csv"
        cols = ["path", "language", "soft_score", "normalized",
                "whisper_hyp", "cer", "cer_reject_reason"]
        df   = pd.DataFrame(rejected)
        df[[c for c in cols if c in df.columns]].to_csv(rej_file, index=False)
        log.info("Rejected → %s (%d)", rej_file, len(rejected))

    all_rows = passed + rejected
    summary  = {
        "phase2_input":    len(all_rows),
        "passed":          len(passed),
        "cer_rejected":    len(rejected),
        "cer_threshold":   CER_THRESHOLD,
        "whisper_model":   model_size,
        "backend":         "faster-whisper (3-stage pipeline)",
        "per_language": {
            lang: {
                "total":    len([r for r in all_rows if r.get("language") == lang]),
                "passed":   sum(1 for r in passed   if r.get("language") == lang),
                "rejected": sum(1 for r in rejected if r.get("language") == lang),
            }
            for lang in sorted(set(r.get("language", "unknown") for r in all_rows))
        },
    }
    with open(out / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary → %s", out / "phase2_summary.json")


def main(args):
    global CER_THRESHOLD
    CER_THRESHOLD = args.cer_threshold

    log.info("Model: %s | CER threshold: %.2f", args.model, CER_THRESHOLD)

    rows = read_manifest(args.manifest)
    if not rows:
        log.error("No qualifying rows in manifest.")
        return

    hw = build_hw_profile(args.model)

    log.info("Loading Whisper model...")
    t0    = time.perf_counter()
    model = WhisperModel(
        args.model,
        device=hw["device"],
        compute_type=hw["compute_type"],
        num_workers=max(1, hw["cpu_cores"] // 2), 
    )
    log.info("Model loaded in %.1fs", time.perf_counter() - t0)

    t1 = time.perf_counter()
    passed, rejected = run_pipeline(rows, model, hw, CER_THRESHOLD)
    elapsed = time.perf_counter() - t1

    save_outputs(passed, rejected, Path(args.output), args.model)

    total = len(rows)
    log.info(
        "Done in %.1fs (%.2f files/sec) | "
        "Passed: %d (%.1f%%) | Rejected: %d (%.1f%%)",
        elapsed, total / max(elapsed, 0.001),
        len(passed),   100 * len(passed)   / max(total, 1),
        len(rejected), 100 * len(rejected) / max(total, 1),
    )


if __name__ == "__main__":
    mp.freeze_support()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--manifest",      default=MANIFEST_PATH)
    parser.add_argument("--output",        default=OUTPUT_DIR)
    parser.add_argument("--cer-threshold", type=float, default=CER_THRESHOLD)
    parser.add_argument("--model",         default="large-v3")
    main(parser.parse_args())