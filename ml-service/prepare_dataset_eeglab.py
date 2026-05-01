"""
prepare_dataset_eeglab.py  —  ml-service/

Prepara dataset de entrenamiento para estados hipnóticos (awake/induction/trance)
a partir de archivos EEGLAB .set/.fdt organizados por sugestionabilidad:

  data/ICApruned_highs/
  data/ICApruned_lows/

Salida:
  NPZ con arrays:
    X                  float32 (n_samples, 15)
    y                  int32   (n_samples,)   0=awake, 1=induction, 2=trance
    subject_id         str     (n_samples,)
    suggestibility     str     (n_samples,)   high|low
    split              str     (n_samples,)   train|test

El split es por sujeto completo para evitar fuga de informacion.
"""

import argparse
import os
import re
import sys
import warnings
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")

import numpy as np
from scipy.signal import butter, coherence, filtfilt, hilbert, welch

sys.path.insert(0, os.path.dirname(__file__))
from src.classifier import CHANNELS_ORDERED, N_FEATURES

# Constantes
THETA = (4.0, 8.0)
BETA = (13.0, 30.0)
WIN_SEC = 2.048
HOP_SEC = 1.0
SKIP_SEC = 2.0

STATE_LABELS = {
    0: "awake",
    1: "induction",
    2: "trance",
}


def _bandpower(x: np.ndarray, fs: int, fmin: float, fmax: float) -> float:
    nperseg = min(len(x), fs)
    freqs, psd = welch(x, fs=fs, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


def _plv(x: np.ndarray, y: np.ndarray, fs: int, fmin: float, fmax: float) -> float:
    nyq = fs / 2.0
    b, a = butter(4, [fmin / nyq, fmax / nyq], btype="bandpass")
    xf = filtfilt(b, a, x)
    yf = filtfilt(b, a, y)
    phi = np.angle(hilbert(xf)) - np.angle(hilbert(yf))
    return float(np.abs(np.mean(np.exp(1j * phi))))


def _coherence_band(x: np.ndarray, y: np.ndarray, fs: int, fmin: float, fmax: float) -> float:
    nperseg = min(len(x), fs)
    freqs, coh = coherence(x, y, fs=fs, nperseg=nperseg)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.mean(coh[mask])) if mask.any() else 0.0


def extract_features(window: np.ndarray, ch_names: List[str], fs: int) -> np.ndarray:
    idx = {ch: i for i, ch in enumerate(ch_names)}

    tbr_features = []
    for ch in CHANNELS_ORDERED:
        x = window[idx[ch]]
        t = _bandpower(x, fs, *THETA)
        b = _bandpower(x, fs, *BETA)
        tbr_features.append(min(t / max(b, 1e-9), 20.0))

    fz = window[idx["Fz"]]
    pz = window[idx["Pz"]]
    f3 = window[idx["F3"]]
    f4 = window[idx["F4"]]
    c3 = window[idx["C3"]]
    c4 = window[idx["C4"]]

    coh_fz_pz = _coherence_band(fz, pz, fs, *THETA)
    plv_f3_f4 = _plv(f3, f4, fs, *THETA)
    plv_c3_c4 = _plv(c3, c4, fs, *THETA)

    th_fz = _bandpower(fz, fs, *THETA)
    th_f3 = _bandpower(f3, fs, *THETA)
    th_f4 = _bandpower(f4, fs, *THETA)
    fsi = min(th_fz / max((th_f3 + th_f4) / 2.0, 1e-9), 5.0)

    return np.array(
        tbr_features + [coh_fz_pz, plv_f3_f4, plv_c3_c4, fsi],
        dtype=np.float32,
    )


def _channel_aliases() -> Dict[str, List[str]]:
    return {
        "Fz": ["Fz", "FZ"],
        "Fp1": ["Fp1", "FP1", "Fpz", "FPZ", "F7"],
        "F3": ["F3"],
        "C3": ["C3"],
        "Pz": ["Pz", "PZ"],
        "O1": ["O1", "OZ", "Oz", "PO3"],
        "F4": ["F4"],
        "C4": ["C4"],
        "P4": ["P4"],
        "O2": ["O2", "OZ", "Oz", "PO4"],
        "Cz": ["Cz", "CZ"],
    }


def _resolve_channel_mapping(ch_names: List[str]) -> Dict[str, str]:
    aliases = _channel_aliases()
    available = {ch.lower(): ch for ch in ch_names}
    used_sources = set()
    mapping: Dict[str, str] = {}

    for target in CHANNELS_ORDERED:
        found = None
        for cand in aliases.get(target, [target]):
            src = available.get(cand.lower())
            if src is None:
                continue
            if src in used_sources and src.upper() not in {"OZ", "FPZ", "FP1"}:
                continue
            found = src
            break
        if found is None:
            raise ValueError(f"No se pudo mapear canal requerido '{target}' desde {ch_names}")
        mapping[target] = found
        used_sources.add(found)

    return mapping


def _parse_numeric_markers(raw_annotations) -> Dict[int, float]:
    markers: Dict[int, float] = {}
    for ann in raw_annotations:
        desc = str(ann["description"]).strip()
        if not re.fullmatch(r"\d+", desc):
            continue
        marker = int(desc)
        onset = float(ann["onset"])
        if marker not in markers:
            markers[marker] = onset
    return markers


def _subject_segments(markers: Dict[int, float], duration_s: float, group: str) -> List[Tuple[float, float, int]]:
    segments: List[Tuple[float, float, int]] = []

    m1 = markers.get(1)
    m2 = markers.get(2)
    m3 = markers.get(3)
    m4 = markers.get(4)
    m5 = markers.get(5)

    # Awake: 1 -> 2
    if m1 is not None and m2 is not None and m2 > m1:
        segments.append((m1, m2, 0))

    # Induction: 2 -> 3 (siempre)
    if m2 is not None and m3 is not None and m3 > m2:
        segments.append((m2, m3, 1))

    # Trance/Induction segun grupo
    if group == "high":
        if m3 is not None and m4 is not None and m4 > m3:
            segments.append((m3, m4, 2))
        if m4 is not None and m5 is not None and m5 > m4:
            segments.append((m4, m5, 2))
    else:
        if m3 is not None and m4 is not None and m4 > m3:
            segments.append((m3, m4, 1))

    # Recuperacion 5->fin no se usa por defecto
    _ = duration_s
    return segments


def _extract_windows_for_segments(
    data: np.ndarray,
    fs: int,
    segments: List[Tuple[float, float, int]],
) -> Tuple[np.ndarray, np.ndarray]:
    win_n = int(WIN_SEC * fs)
    hop_n = int(HOP_SEC * fs)
    skip_n = int(SKIP_SEC * fs)

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for onset_s, offset_s, label in segments:
        start = int(onset_s * fs) + skip_n
        end = int(offset_s * fs) - skip_n
        if end <= start + win_n:
            continue

        i = start
        while i + win_n <= end:
            window = data[:, i:i + win_n]
            feat = extract_features(window, CHANNELS_ORDERED, fs)
            if feat.shape[0] != N_FEATURES:
                raise ValueError("Vector de features invalido")
            X_list.append(feat)
            y_list.append(label)
            i += hop_n

    if not X_list:
        return np.empty((0, N_FEATURES), dtype=np.float32), np.empty((0,), dtype=np.int32)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


def _list_subject_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".set")
    )


def _split_subjects(high_subjects: List[str], low_subjects: List[str], train_ratio: float, seed: int) -> Dict[str, str]:
    rng = np.random.RandomState(seed)

    high = high_subjects.copy()
    low = low_subjects.copy()
    rng.shuffle(high)
    rng.shuffle(low)

    n_high_train = int(round(len(high) * train_ratio))
    n_low_train = int(round(len(low) * train_ratio))

    train_set = set(high[:n_high_train] + low[:n_low_train])

    split_map: Dict[str, str] = {}
    for sid in high + low:
        split_map[sid] = "train" if sid in train_set else "test"
    return split_map


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae features desde .set/.fdt y genera split por sujeto (train/test)."
    )
    parser.add_argument("--high-dir", default="data/ICApruned_highs")
    parser.add_argument("--low-dir", default="data/ICApruned_lows")
    parser.add_argument("--output", default="data/sessions_icapruned.npz")
    parser.add_argument("--train-ratio", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fs-target", type=int, default=250)
    parser.add_argument("--max-windows-per-class-per-split", type=int, default=1200)
    args = parser.parse_args()

    import mne

    high_sets = _list_subject_files(args.high_dir)
    low_sets = _list_subject_files(args.low_dir)

    if not high_sets or not low_sets:
        print("[error] No se encontraron archivos .set en high-dir o low-dir")
        sys.exit(1)

    high_subjects = [os.path.splitext(os.path.basename(p))[0] for p in high_sets]
    low_subjects = [os.path.splitext(os.path.basename(p))[0] for p in low_sets]
    split_map = _split_subjects(high_subjects, low_subjects, args.train_ratio, args.seed)

    print("=" * 72)
    print("  PREPARACION DATASET EEGLAB — Estados hipnoticos")
    print("=" * 72)
    print(f"  Highs: {len(high_sets)} | Lows: {len(low_sets)}")
    print(f"  Split train/test por sujeto: {args.train_ratio:.2f}/{1.0 - args.train_ratio:.2f}")

    all_X: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    all_subject: List[np.ndarray] = []
    all_group: List[np.ndarray] = []
    all_split: List[np.ndarray] = []

    for group, files in (("high", high_sets), ("low", low_sets)):
        for set_path in files:
            subject_id = os.path.splitext(os.path.basename(set_path))[0]
            split = split_map[subject_id]

            try:
                raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
                mapping = _resolve_channel_mapping(raw.ch_names)

                src_channels = [mapping[ch] for ch in CHANNELS_ORDERED]
                data = raw.get_data(picks=src_channels)

                if args.fs_target > 0 and int(raw.info["sfreq"]) != args.fs_target:
                    raw.resample(args.fs_target, verbose=False)
                    data = raw.get_data(picks=src_channels)

                raw_fs = int(raw.info["sfreq"])
                raw.notch_filter(50.0, verbose=False)
                raw.filter(1.0, 40.0, verbose=False)
                data = raw.get_data(picks=src_channels)

                markers = _parse_numeric_markers(raw.annotations)
                segments = _subject_segments(markers, float(raw.times[-1]), group)
                X_sub, y_sub = _extract_windows_for_segments(data, raw_fs, segments)

                if len(X_sub) == 0:
                    print(f"  [skip] {subject_id}: sin ventanas validas")
                    continue

                all_X.append(X_sub)
                all_y.append(y_sub)
                all_subject.append(np.array([subject_id] * len(y_sub), dtype=object))
                all_group.append(np.array([group] * len(y_sub), dtype=object))
                all_split.append(np.array([split] * len(y_sub), dtype=object))

                counts = {k: int((y_sub == k).sum()) for k in (0, 1, 2)}
                print(
                    f"  {subject_id:>10s} [{group:4s}/{split:5s}]  "
                    f"awake={counts[0]:4d} induction={counts[1]:4d} trance={counts[2]:4d}"
                )

            except Exception as exc:
                print(f"  [skip] {subject_id}: {exc}")

    if not all_X:
        print("[error] No se pudieron extraer ventanas del dataset.")
        sys.exit(1)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subject_id = np.concatenate(all_subject, axis=0)
    suggestibility = np.concatenate(all_group, axis=0)
    split = np.concatenate(all_split, axis=0)

    # Balance por clase dentro de cada split
    rng = np.random.RandomState(args.seed)
    keep_idx = []
    for split_name in ("train", "test"):
        for cls in (0, 1, 2):
            idx = np.where((split == split_name) & (y == cls))[0]
            if len(idx) == 0:
                continue
            n = min(len(idx), args.max_windows_per_class_per_split)
            chosen = rng.choice(idx, size=n, replace=False)
            keep_idx.extend(chosen.tolist())

    keep_idx = np.array(keep_idx, dtype=np.int64)
    rng.shuffle(keep_idx)

    X = X[keep_idx]
    y = y[keep_idx]
    subject_id = subject_id[keep_idx]
    suggestibility = suggestibility[keep_idx]
    split = split[keep_idx]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(
        args.output,
        X=X.astype(np.float32),
        y=y.astype(np.int32),
        subject_id=subject_id.astype(str),
        suggestibility=suggestibility.astype(str),
        split=split.astype(str),
    )

    print("\n" + "=" * 72)
    print("  RESUMEN")
    print("=" * 72)
    print(f"  Muestras totales: {len(X)}")
    for split_name in ("train", "test"):
        idx = split == split_name
        print(f"  {split_name.upper():5s}: {int(idx.sum())} ventanas | sujetos={len(np.unique(subject_id[idx]))}")
        for cls in (0, 1, 2):
            n = int(((y == cls) & idx).sum())
            print(f"    - {STATE_LABELS[cls]:10s}: {n}")

    print(f"\n  Guardado: {args.output}")
    print("  Entrenar con:")
    print(f"    python ml-service/train_classifier.py --real {args.output}")


if __name__ == "__main__":
    main()
