"""
prepare_dataset.py  —  ml-service/

Lee los datos BIDS del dataset EEGMeditation (BrainVision .eeg),
extrae el vector de 15 features por ventana de 2 s y guarda el
resultado como data/sessions.npz listo para train_classifier.py.

Mapeo de etiquetas (EEGMeditation → clases del clasificador):
  OPEN_EYES              → 0  (awake      — ojos abiertos, alerta)
  CLOSE_EYES             → 1  (induction  — ojos cerrados, relajación)
  INTERNAL_CONCENTRATION → 2  (trance     — meditación profunda)

Uso:
  python ml-service/prepare_dataset.py
  python ml-service/prepare_dataset.py --data-dir data/yoga_eeg --output data/sessions.npz
  python ml-service/prepare_dataset.py --max-windows-per-class 600
"""

import argparse
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from scipy.signal import butter, coherence, filtfilt, hilbert, welch

sys.path.insert(0, os.path.dirname(__file__))
from src.classifier import CHANNELS_ORDERED, N_FEATURES

# ─── Constantes ──────────────────────────────────────────────────────────────
FS_TARGET = 250          # Hz — frecuencia objetivo tras resampleo
WIN_SEC   = 2.0          # s  — tamaño de ventana
HOP_SEC   = 1.0          # s  — paso entre ventanas (50 % solapamiento)
THETA     = (4.0, 8.0)   # Hz
BETA      = (13.0, 30.0) # Hz
SKIP_SEC  = 2.0          # s  — margen a descartar al inicio/fin de segmento

# Mapeo keyword del TSV → clase
EVENT_LABEL = {
    "OPEN_EYES":              0,   # awake
    "CLOSE_EYES":             1,   # induction
    "INTERNAL_CONCENTRATION": 2,   # trance
}


# ─── Extracción de features ───────────────────────────────────────────────────

def _bandpower(x: np.ndarray, fs: int, fmin: float, fmax: float) -> float:
    nperseg = min(len(x), fs)  # ventana Welch de 1 s
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


def extract_features(window: np.ndarray, ch_names: list, fs: int) -> np.ndarray:
    """
    Extrae el vector de 15 features desde una ventana EEG.

    window   : shape (n_channels, n_samples)
    ch_names : lista de nombres de canal en el mismo orden que window
    fs       : frecuencia de muestreo en Hz
    """
    idx = {ch: i for i, ch in enumerate(ch_names)}

    # ── Features 0–10: TBR por canal ─────────────────────────────────────
    tbr_features = []
    for ch in CHANNELS_ORDERED:
        x = window[idx[ch]]
        t = _bandpower(x, fs, *THETA)
        b = _bandpower(x, fs, *BETA)
        tbr_features.append(min(t / max(b, 1e-9), 20.0))

    # ── Features 11–13: conectividad ─────────────────────────────────────
    fz = window[idx["Fz"]]
    pz = window[idx["Pz"]]
    f3 = window[idx["F3"]]
    f4 = window[idx["F4"]]
    c3 = window[idx["C3"]]
    c4 = window[idx["C4"]]

    coh_fz_pz = _coherence_band(fz, pz, fs, *THETA)
    plv_f3_f4 = _plv(f3, f4, fs, *THETA)
    plv_c3_c4 = _plv(c3, c4, fs, *THETA)

    # ── Feature 14: FSI ───────────────────────────────────────────────────
    th_fz = _bandpower(fz, fs, *THETA)
    th_f3 = _bandpower(f3, fs, *THETA)
    th_f4 = _bandpower(f4, fs, *THETA)
    fsi = min(th_fz / max((th_f3 + th_f4) / 2.0, 1e-9), 5.0)

    return np.array(tbr_features + [coh_fz_pz, plv_f3_f4, plv_c3_c4, fsi], dtype=np.float32)


# ─── Parseo de eventos ────────────────────────────────────────────────────────

def parse_segments(tsv_path: str) -> list:
    """
    Lee el events.tsv y retorna lista de (onset_s, offset_s, label).
    La columna 'value' contiene el nombre del evento (ej. OPEN_EYES_START).
    """
    segments = []
    active   = {}  # keyword → onset_s

    with open(tsv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            value = row.get("value", "").strip()
            onset = float(row.get("onset", 0.0))

            for keyword, label in EVENT_LABEL.items():
                if value == f"{keyword}_START":
                    active[keyword] = onset
                elif value == f"{keyword}_END" and keyword in active:
                    segments.append((active.pop(keyword), onset, label))

    return segments


# ─── Procesamiento por sujeto ─────────────────────────────────────────────────

def process_subject(vhdr_path: str, tsv_path: str) -> tuple:
    """
    Carga, filtra, segmenta y extrae features de un sujeto.
    Retorna (X, y) con los features y etiquetas de todas las ventanas.
    """
    import mne

    print(f"    Cargando : {os.path.basename(vhdr_path)}")
    raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)

    # Seleccionar solo los 11 canales necesarios por el clasificador
    raw.pick_channels(CHANNELS_ORDERED)

    # Resampleo primero (aplica filtro anti-aliasing interno de MNE)
    print(f"    Resampleando 2048 Hz → {FS_TARGET} Hz ...")
    raw.resample(FS_TARGET, verbose=False)

    # Filtrado a la frecuencia reducida (más rápido que filtrar a 2048 Hz)
    raw.notch_filter(50.0, verbose=False)
    raw.filter(1.0, 40.0, verbose=False)

    data     = raw.get_data()        # shape (11, n_samples)
    ch_names = raw.ch_names          # ["Fz", "Fp1", ...]
    fs       = int(raw.info["sfreq"])

    win_n  = int(WIN_SEC * fs)
    hop_n  = int(HOP_SEC * fs)
    skip_n = int(SKIP_SEC * fs)

    segments = parse_segments(tsv_path)
    print(f"    Segmentos mapeados : {len(segments)}")

    cls_names = {0: "awake", 1: "induction", 2: "trance"}
    X_list, y_list = [], []

    for onset_s, offset_s, label in segments:
        start = int(onset_s  * fs) + skip_n
        end   = int(offset_s * fs) - skip_n

        n_windows = 0
        i = start
        while i + win_n <= end:
            window = data[:, i: i + win_n]
            feat   = extract_features(window, ch_names, fs)
            X_list.append(feat)
            y_list.append(label)
            i += hop_n
            n_windows += 1

        print(f"      [{cls_names[label]:12s}] {onset_s:7.0f}–{offset_s:7.0f} s → {n_windows} ventanas")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrae features EEG desde datos BIDS para entrenar el clasificador.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", default="data/yoga_eeg",
        help="Directorio raíz del dataset BIDS (default: data/yoga_eeg)",
    )
    parser.add_argument(
        "--output", default="data/sessions.npz",
        help="Ruta de salida del .npz (default: data/sessions.npz)",
    )
    parser.add_argument(
        "--max-windows-per-class", type=int, default=800, metavar="N",
        help="Máximo de ventanas por clase para balancear dataset (default: 800)",
    )
    args = parser.parse_args()

    sep = "=" * 62
    print(f"\n{sep}")
    print("  EXTRACCIÓN DE FEATURES — Dataset EEGMeditation")
    print(sep)
    print(f"\n  Directorio : {args.data_dir}")
    print(f"  Salida     : {args.output}")
    print(f"  Fs target  : {FS_TARGET} Hz | Ventana: {WIN_SEC} s | Hop: {HOP_SEC} s")
    print(f"  Max ventanas/clase : {args.max_windows_per_class}")

    subjects = sorted([
        d for d in os.listdir(args.data_dir)
        if d.startswith("sub-") and os.path.isdir(os.path.join(args.data_dir, d))
    ])
    print(f"\n  Sujetos : {subjects}\n")

    all_X, all_y = [], []

    for subj in subjects:
        # Buscar dinámicamente la carpeta de sesión y los archivos
        # (el prefijo del archivo puede diferir de la carpeta, ej. ses-02 dentro de ses-01)
        vhdr, tsv = None, None
        subj_root = os.path.join(args.data_dir, subj)
        for ses in sorted(os.listdir(subj_root)):
            eeg_dir = os.path.join(subj_root, ses, "eeg")
            if not os.path.isdir(eeg_dir):
                continue
            for fname in os.listdir(eeg_dir):
                if fname.endswith("_eeg.vhdr"):
                    vhdr = os.path.join(eeg_dir, fname)
                    tsv_name = fname.replace("_eeg.vhdr", "_events.tsv")
                    tsv = os.path.join(eeg_dir, tsv_name)
                    break
            if vhdr:
                break

        if not vhdr or not tsv or not os.path.exists(vhdr) or not os.path.exists(tsv):
            print(f"  [skip] {subj} — archivos no encontrados")
            continue

        print(f"  ── {subj} {'─'*44}")
        X_sub, y_sub = process_subject(vhdr, tsv)
        all_X.append(X_sub)
        all_y.append(y_sub)
        print(f"    Total extraído : {len(X_sub)} ventanas\n")

    if not all_X:
        print("\n[error] No se encontraron datos. Verifica --data-dir.")
        sys.exit(1)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # ── Balance de clases ─────────────────────────────────────────────────
    max_per = args.max_windows_per_class
    rng     = np.random.RandomState(42)
    idxs    = []
    for cls in [0, 1, 2]:
        cls_idx = np.where(y == cls)[0]
        n       = min(len(cls_idx), max_per)
        chosen  = rng.choice(cls_idx, size=n, replace=False)
        idxs.extend(chosen.tolist())
    idxs = np.array(idxs)
    rng.shuffle(idxs)
    X, y = X[idxs], y[idxs]

    # ── Resumen ───────────────────────────────────────────────────────────
    print(f"{sep}")
    print("  RESUMEN FINAL")
    print(f"{sep}")
    print(f"\n  Total muestras : {len(X)}")
    for cls, name in {0: "awake", 1: "induction", 2: "trance"}.items():
        n = int((y == cls).sum())
        print(f"    {cls} ({name:12s}) : {n:4d} ventanas")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, X=X, y=y)
    print(f"\n  Guardado en : {args.output}")
    print(f"\n  Siguiente paso:")
    print(f"    python ml-service/train_classifier.py --real {args.output}")
    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
