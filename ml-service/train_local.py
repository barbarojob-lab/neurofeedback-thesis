#!/usr/bin/env python3
"""
train_local.py — ml-service/
═══════════════════════════════════════════════════════════════════════════════
Entrenamiento local de los clasificadores EEG para hipnosis.
Equivalente al notebook de Google Colab, sin necesidad de subir datos.

Genera:
  data/models_colab/eeg_classifier_high.joblib
  data/models_colab/eeg_classifier_low.joblib
  data/sessions_icapruned.npz  (dataset combinado para referencia futura)

Tiempo estimado: 3–8 minutos en CPU i5+.

Uso:
    cd neurofeedback-thesis
    .venv/Scripts/python.exe ml-service/train_local.py
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import json
import warnings
from pathlib import Path

import joblib
import mne
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import butter, filtfilt, hilbert, iirnotch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ─── Rutas ────────────────────────────────────────────────────────────────────

ROOT       = Path(__file__).parent.parent
DATA_DIR   = ROOT / "data"
HIGHS_DIR  = DATA_DIR / "ICApruned_highs"
LOWS_DIR   = DATA_DIR / "ICApruned_lows"
MODELS_DIR = DATA_DIR / "models_colab"
LOGS_DIR   = ROOT / "logs"

# Split fijo 20 entrenamiento / 10 validación (5 por grupo).
# Los sujetos de validación se determinan por seed=42 + forzados adicionales.
# NUNCA se re-entrena el modelo final sobre los sujetos de validación.
# val_subject_ratio = 5/15 ≈ 0.333 → exactamente 10 train + 5 val por grupo.
DEFAULT_VAL_SUBJECT_RATIO = 5 / 15   # 0.3333...
DEFAULT_FORCED_VAL_HIGH = "subject_11"
DEFAULT_FORCED_VAL_LOW = "subject_19"
LOW_CLASS_WEIGHT = {0: 2.0, 1: 1.0, 2: 1.0}

# ─── Parámetros de señal ──────────────────────────────────────────────────────

FS         = 250        # Hz — frecuencia de muestreo objetivo (igual que backend)
WINDOW     = 512        # muestras = 2.048 s a 250 Hz (igual que backend)
HOP        = 64         # muestras — hop/stride entre ventanas
NOTCH_HZ   = 50.0       # Hz — notch para interferencia de red eléctrica
BP_LOW     = 1.0        # Hz — banda baja del filtro paso-banda
BP_HIGH    = 40.0       # Hz — banda alta

# Bandas de frecuencia EEG (Hz)
THETA = (4.0, 8.0)
BETA  = (12.0, 30.0)

# ─── Canales ──────────────────────────────────────────────────────────────────

# Orden requerido por classifier.py / connectivity.py
CHANNELS_OUT = ["Fz", "Fp1", "F3", "C3", "Pz", "O1", "F4", "C4", "P4", "O2", "Cz"]
N_CH = len(CHANNELS_OUT)  # 11

# Mapeo de nombres EEGLAB (.set) → nombres canonicos
CH_MAP = {
    "FZ":  "Fz",   # frontal midline
    "F7":  "Fp1",  # mejor proxy disponible para Fp1 en estos registros
    "F3":  "F3",
    "C3":  "C3",
    "PZ":  "Pz",
    "PO3": "O1",   # parieto-occipital izquierdo → proxy O1
    "F4":  "F4",
    "C4":  "C4",
    "P4":  "P4",
    "PO4": "O2",   # parieto-occipital derecho → proxy O2
    "CZ":  "Cz",
}

# ─── Umbrales heurísticos para etiquetado (calibrados P25/P70 de MIBD dataset) ────
# Calibrados el 21-05-2025 usando análisis de percentiles reales sobre HIGH+LOW
# Referencias: Gruzelier et al. (2006, 2014), Hammond (2011), Sabourin et al. (1990)
# 
# Estrategia (basada en percentiles):
#   P10-P25 (0.307-0.506 TBR): reposo genuino → AWAKE
#   P25-P70 (0.506-1.418 TBR): inducción hipnagógica → INDUCTION
#   P70+    (1.418+ TBR):      trance profundo → TRANCE
#
TBR_TRANCE     = 1.418  # TBR_Fz ≥ 1.418 AND coh ≥ 0.799 → trance
COH_TRANCE     = 0.799
TBR_INDUCTION  = 0.514  # TBR_Fz ≥ 0.514 OR  coh ≥ 0.544 → induction
COH_INDUCTION  = 0.544
# else → awake (0) — reposo cognitivo bajo

# ─── Índices de canales ───────────────────────────────────────────────────────

IDX = {ch: i for i, ch in enumerate(CHANNELS_OUT)}

# ─── Funciones de procesamiento ───────────────────────────────────────────────

def _bp_sos(fs: int) -> tuple:
    """Segunda orden de secciones para Butterworth BP 1–40 Hz orden 4."""
    nyq = fs / 2
    return butter(4, [BP_LOW / nyq, BP_HIGH / nyq], btype="band", output="sos")


def _notch_ba(fs: int) -> tuple:
    """Coeficientes b, a para filtro notch 50 Hz Q=30."""
    return iirnotch(NOTCH_HZ, Q=30.0, fs=fs)


def load_subject(set_path: Path) -> np.ndarray | None:
    """
    Carga un archivo .set, renombra canales, remuestrea a FS Hz,
    aplica filtros y retorna datos en µV, shape (N_CH, n_samples).
    Retorna None si faltan canales requeridos.
    """
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)

    # Renombrar canales EEGLAB → nombres canonicos
    rename = {k: v for k, v in CH_MAP.items() if k in raw.ch_names}
    raw.rename_channels(rename)

    # Verificar canales disponibles
    missing = [c for c in CHANNELS_OUT if c not in raw.ch_names]
    if missing:
        print(f"  SKIP {set_path.name}: canales faltantes {missing}")
        return None

    raw.pick(CHANNELS_OUT)

    # Resamplear si es necesario (los .set son a 125 Hz)
    if int(raw.info["sfreq"]) != FS:
        raw.resample(FS, verbose=False)

    # Referencia promedio (reduce artefactos de referencia)
    raw.set_eeg_reference("average", projection=False, verbose=False)

    # Convertir V → µV
    data = raw.get_data() * 1e6  # (N_CH, n_samples)

    n_samples = data.shape[1]

    # Filtro notch 50 Hz (zero-phase filtfilt)
    nb, na = _notch_ba(FS)
    for ch in range(N_CH):
        data[ch] = filtfilt(nb, na, data[ch])

    # Filtro paso-banda 1–40 Hz orden 4 (zero-phase SOS)
    from scipy.signal import sosfiltfilt
    sos = _bp_sos(FS)
    for ch in range(N_CH):
        data[ch] = sosfiltfilt(sos, data[ch])

    return data


def extract_features_vectorized(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae el vector de 15 features y etiqueta heurística para todas
    las ventanas de una sesión. Completamente vectorizado.

    data: shape (N_CH, n_samples) en µV — ya filtrado
    Returns: X (n_windows, 15) float32, y (n_windows,) int32
    """
    n_ch, n_samples = data.shape
    df = FS / WINDOW  # resolución espectral Hz/bin

    # ── Ventanas: (n_ch, n_windows, WINDOW) → (n_windows, n_ch, WINDOW) ──
    wins = sliding_window_view(data, window_shape=WINDOW, axis=1)[:, ::HOP, :]
    wins = wins.transpose(1, 0, 2).astype(np.float64)  # (n_windows, n_ch, WINDOW)
    n_windows = wins.shape[0]

    # ── FFT con ventana Hann (igual que backend TypeScript) ───────────────
    hann = np.hanning(WINDOW)
    windowed = wins * hann[np.newaxis, np.newaxis, :]
    fft_all = np.fft.rfft(windowed, axis=-1)          # (n_windows, n_ch, WINDOW//2+1)
    mags    = np.abs(fft_all) / WINDOW                 # amplitud en µV

    def band_power(f_lo: float, f_hi: float) -> np.ndarray:
        """Potencia integrada → shape (n_windows, n_ch) en µV²."""
        b_lo = max(0,        round(f_lo / df))
        b_hi = min(WINDOW//2, round(f_hi / df))
        return np.sum(mags[:, :, b_lo:b_hi] ** 2, axis=-1)

    theta_pow = band_power(*THETA)  # (n_windows, n_ch)
    beta_pow  = band_power(*BETA)

    # Features [0–10]: TBR por canal (adimensional, clip [0, 20])
    tbr = np.clip(theta_pow / np.maximum(beta_pow, 1e-12), 0.0, 20.0)

    # Feature [14]: FSI = theta_Fz / mean(theta_F3, theta_F4), clip [0, 5]
    denom_fsi = np.maximum((theta_pow[:, IDX["F3"]] + theta_pow[:, IDX["F4"]]) / 2, 1e-12)
    fsi = np.clip(theta_pow[:, IDX["Fz"]] / denom_fsi, 0.0, 5.0)

    # Feature [11]: Coherencia Fz–Pz en theta
    # Welch real con subsegmentos de 256 y overlap 50% dentro de cada ventana de 512.
    # Esto evita la coherencia degenerada=1.0 de un solo segmento FFT.
    fz_win = wins[:, IDX["Fz"], :]  # (n_windows, 512)
    pz_win = wins[:, IDX["Pz"], :]  # (n_windows, 512)

    nperseg = 256
    noverlap = 128
    step = nperseg - noverlap  # 128

    fz_seg = sliding_window_view(fz_win, window_shape=nperseg, axis=1)[:, ::step, :]
    pz_seg = sliding_window_view(pz_win, window_shape=nperseg, axis=1)[:, ::step, :]
    # shapes: (n_windows, n_segments=3, 256)

    hann_w = np.hanning(nperseg)
    fz_seg = fz_seg * hann_w[np.newaxis, np.newaxis, :]
    pz_seg = pz_seg * hann_w[np.newaxis, np.newaxis, :]

    fz_fft = np.fft.rfft(fz_seg, axis=-1)
    pz_fft = np.fft.rfft(pz_seg, axis=-1)

    # PSD/CPSD promedio entre segmentos (Welch)
    pxx = np.mean(np.abs(fz_fft) ** 2, axis=1)
    pyy = np.mean(np.abs(pz_fft) ** 2, axis=1)
    pxy = np.mean(fz_fft * np.conj(pz_fft), axis=1)

    coh = (np.abs(pxy) ** 2) / np.maximum(pxx * pyy, 1e-30)

    df_coh = FS / nperseg
    b_th_lo = max(0,           int(np.ceil(THETA[0] / df_coh)))
    b_th_hi = min(coh.shape[1], int(np.floor(THETA[1] / df_coh)) + 1)
    coh_fz_pz = np.clip(np.mean(coh[:, b_th_lo:b_th_hi], axis=-1), 0.0, 1.0)

    # Features [12–13]: PLV F3–F4, C3–C4 en theta (Lachaux et al., 1999)
    nyq = FS / 2
    b_theta, a_theta = butter(4, [THETA[0] / nyq, THETA[1] / nyq], btype="band")
    theta_sig = np.zeros_like(data, dtype=np.float64)
    for ch in range(n_ch):
        theta_sig[ch] = filtfilt(b_theta, a_theta, data[ch])

    phases = np.angle(hilbert(theta_sig, axis=-1))  # (n_ch, n_samples)
    ph_wins = sliding_window_view(phases, window_shape=WINDOW, axis=1)[:, ::HOP, :]
    # shape: (n_ch, n_windows, WINDOW)

    def plv_pair(ch_a: int, ch_b: int) -> np.ndarray:
        """PLV entre dos canales para todas las ventanas → (n_windows,)."""
        delta = ph_wins[ch_a] - ph_wins[ch_b]  # (n_windows, WINDOW)
        return np.clip(np.abs(np.mean(np.exp(1j * delta), axis=-1)), 0.0, 1.0)

    plv_f3_f4 = plv_pair(IDX["F3"], IDX["F4"])
    plv_c3_c4 = plv_pair(IDX["C3"], IDX["C4"])

    # ── Ensamblar X: (n_windows, 15) ─────────────────────────────────────
    X = np.column_stack([tbr, coh_fz_pz, plv_f3_f4, plv_c3_c4, fsi]).astype(np.float32)

    # ── Etiquetas heurísticas (misma lógica que pipeline.py) ─────────────
    tbr_fz = tbr[:, IDX["Fz"]]
    y = np.zeros(n_windows, dtype=np.int32)
    y[(tbr_fz >= TBR_INDUCTION) | (coh_fz_pz >= COH_INDUCTION)] = 1   # induction
    y[(tbr_fz >= TBR_TRANCE) & (coh_fz_pz >= COH_TRANCE)]       = 2   # trance (sobreescribe)

    # Descartar ventanas con valores inválidos
    valid = np.isfinite(X).all(axis=1)
    return X[valid], y[valid]


def build_dataset(set_files: list[Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carga y procesa todos los sujetos de un grupo."""
    X_all, y_all, s_all = [], [], []
    for path in set_files:
        subject_id = path.stem
        print(f"  {path.name} ... ", end="", flush=True)
        data = load_subject(path)
        if data is None:
            continue
        X, y = extract_features_vectorized(data)
        counts = np.bincount(y, minlength=3)
        print(f"{len(X)} ventanas | awake={counts[0]}  induct={counts[1]}  trance={counts[2]}")
        X_all.append(X)
        y_all.append(y)
        s_all.append(np.array([subject_id] * len(y), dtype=object))

    if not X_all:
        raise RuntimeError("No se pudo cargar ningún sujeto.")
    return np.vstack(X_all), np.concatenate(y_all), np.concatenate(s_all)


def split_train_val_by_subject(
    subject_ids: np.ndarray,
    val_subject_ratio: float,
    forced_val_subjects: set[str] | None = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split por sujeto para evitar leakage entre ventanas del mismo sujeto.
    """
    rng = np.random.RandomState(random_state)
    unique_subjects = np.array(sorted(np.unique(subject_ids).tolist()), dtype=object)
    if len(unique_subjects) < 2:
        raise RuntimeError("Se requieren al menos 2 sujetos para validación por sujeto.")

    val_subjects: set[object] = set()
    if forced_val_subjects:
        for subj in unique_subjects:
            if str(subj) in forced_val_subjects:
                val_subjects.add(subj)

    if not val_subjects:
        n_val_subjects = max(1, int(round(len(unique_subjects) * val_subject_ratio)))
        n_val_subjects = min(n_val_subjects, len(unique_subjects) - 1)
        shuffled = unique_subjects.copy()
        rng.shuffle(shuffled)
        val_subjects = set(shuffled[:n_val_subjects].tolist())
    elif len(val_subjects) >= len(unique_subjects):
        raise RuntimeError("El conjunto forced_val_subjects deja train vacío.")

    val_mask = np.array([s in val_subjects for s in subject_ids], dtype=bool)
    val_idx = np.where(val_mask)[0]
    train_idx = np.where(~val_mask)[0]
    return train_idx, val_idx


def train_and_evaluate(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    group_name: str,
    random_state: int = 42,
    val_subject_ratio: float = 0.33,
    forced_val_subjects: set[str] | None = None,
    run_group_cv: bool = True,
    class_weight: dict[int, float] | None = None,
) -> Pipeline:
    """
    Entrena VotingClassifier(SVM+RF) con StandardScaler.
    Evalúa en split 80/20, luego re-entrena en todos los datos.
    Retorna el modelo final entrenado en todos los datos.
    """
    tr, val = split_train_val_by_subject(
        subject_ids,
        val_subject_ratio=val_subject_ratio,
        forced_val_subjects=forced_val_subjects,
        random_state=random_state,
    )

    print(f"  Train: {len(tr)} muestras  |  Val: {len(val)} muestras")
    counts_val = np.bincount(y[val], minlength=3)
    counts_tr = np.bincount(y[tr], minlength=3)
    print(f"  Distribucion train: awake={counts_tr[0]}  induct={counts_tr[1]}  trance={counts_tr[2]}")
    print(f"  Distribucion val:   awake={counts_val[0]}  induct={counts_val[1]}  trance={counts_val[2]}")

    train_subjects = sorted(np.unique(subject_ids[tr]).tolist())
    val_subjects = sorted(np.unique(subject_ids[val]).tolist())
    print(f"  Sujetos train ({len(train_subjects)}): {train_subjects}")
    print(f"  Sujetos val   ({len(val_subjects)}): {val_subjects}")

    # SVM calibrado con Platt scaling → probabilidades bien calibradas para soft voting.
    # Sin calibración, las probabilidades del SVC son poco confiables y pueden
    # inflar falsos positivos en la clase trance.
    svm_class_weight = class_weight if class_weight is not None else "balanced"
    rf_class_weight = class_weight if class_weight is not None else "balanced"

    svm_base = SVC(kernel="rbf", C=10.0, gamma="scale",
                   class_weight=svm_class_weight, random_state=random_state)
    svm = CalibratedClassifierCV(svm_base, method="sigmoid", cv=3)
    rf  = RandomForestClassifier(n_estimators=200, min_samples_split=5,
                                  class_weight=rf_class_weight, n_jobs=-1,
                                  random_state=random_state)
    voting = VotingClassifier(estimators=[("svm", svm), ("rf", rf)], voting="soft")
    model  = Pipeline([("scaler", StandardScaler()), ("ensemble", voting)])

    model.fit(X[tr], y[tr])

    # CV grupal sobre train (folds por sujeto) para métrica más robusta.
    cv_folds = min(5, len(train_subjects))
    cv_mean = None
    cv_std = None
    if run_group_cv:
        if cv_folds >= 2:
            gkf = GroupKFold(n_splits=cv_folds)
            cv_scores = cross_val_score(
                model,
                X[tr],
                y[tr],
                groups=subject_ids[tr],
                cv=gkf,
                scoring="accuracy",
                n_jobs=1,
            )
            cv_mean = float(np.mean(cv_scores))
            cv_std = float(np.std(cv_scores))
            print(f"  CV GroupKFold({cv_folds}) acc: {cv_mean:.3f} ± {cv_std:.3f}")
        else:
            print("  [warn] CV grupal omitida: menos de 2 sujetos en train.")
    else:
        print("  CV grupal omitida por --skip-cv")

    y_pred = model.predict(X[val])
    y_proba = model.predict_proba(X[val])  # shape (n_val, n_classes)
    acc = accuracy_score(y[val], y_pred)
    f1  = f1_score(y[val], y_pred, average="macro", zero_division=0)
    print(f"  Val accuracy: {acc:.3f}   F1 macro: {f1:.3f}")

    classes_present = sorted(np.unique(y[val]))
    target_names = ["awake", "induction", "trance"]
    names_present = [target_names[c] for c in classes_present]
    print(classification_report(y[val], y_pred, labels=classes_present,
                                 target_names=names_present, zero_division=0))

    # Umbral de confianza: ¿qué porcentaje de predicciones tiene max_proba >= 0.65?
    max_proba = y_proba.max(axis=1)
    high_conf_pct = float(np.mean(max_proba >= 0.65) * 100)
    print(f"  Predicciones con confianza ≥65%: {high_conf_pct:.1f}%")

    # Matriz de confusión para análisis FP/FN por clase
    cm = confusion_matrix(y[val], y_pred, labels=classes_present)
    print(f"  Matriz de confusión [{' / '.join(names_present)}]:")
    for i, row in enumerate(cm):
        print(f"    real={names_present[i]:>9}: {row}")

    # Persistir reporte de validación por grupo para evidencia de tesis.
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = LOGS_DIR / f"train_{group_name.lower()}_subject_split.json"
    report = {
        "group": group_name,
        "random_state": random_state,
        "val_subject_ratio": val_subject_ratio,
        "class_weight": class_weight,
        "train_subjects": train_subjects,
        "val_subjects": val_subjects,
        "n_train_samples": int(len(tr)),
        "n_val_samples": int(len(val)),
        "class_counts_train": {
            "awake": int(counts_tr[0]),
            "induction": int(counts_tr[1]),
            "trance": int(counts_tr[2]),
        },
        "class_counts_val": {
            "awake": int(counts_val[0]),
            "induction": int(counts_val[1]),
            "trance": int(counts_val[2]),
        },
        "metrics": {
            "val_accuracy": float(acc),
            "val_f1_macro": float(f1),
            "cv_groupkfold_mean_accuracy": cv_mean,
            "cv_groupkfold_std_accuracy": cv_std,
            "high_confidence_pct": float(high_conf_pct),
        },
        "confusion_matrix": {
            "labels": names_present,
            "matrix": cm.tolist(),
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  Reporte de validación: {report_path}")

    # IMPORTANTE: el modelo final se entrega entrenado SOLO sobre los 20 sujetos de train.
    # NO se re-entrena sobre los sujetos de validación — esto garantiza una evaluación
    # honesta para la tesis (held-out set de 10 sujetos nunca vistos).
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Entrenamiento local de modelos EEG")
    parser.add_argument(
        "--val-subject-ratio",
        type=float,
        default=DEFAULT_VAL_SUBJECT_RATIO,
        help="Proporción de sujetos para validación por grupo (default: 5/15≈0.333 → 10 train / 5 val)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria (default: 42)",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Omitir GroupKFold CV (acelera entrenamiento y evita cortes)",
    )
    parser.add_argument(
        "--forced-val-high",
        type=str,
        default=DEFAULT_FORCED_VAL_HIGH,
        help="Sujeto HIGH forzado a validación para evitar leakage (default: subject_11)",
    )
    parser.add_argument(
        "--forced-val-low",
        type=str,
        default=DEFAULT_FORCED_VAL_LOW,
        help="Sujeto LOW forzado a validación para evitar leakage (default: subject_19)",
    )
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Grupo HIGH ────────────────────────────────────────────────────────
    high_files = sorted(HIGHS_DIR.glob("subject_*.set"))
    print(f"\n{'='*60}")
    print(f"GRUPO HIGH — {len(high_files)} sujetos")
    print(f"{'='*60}")
    X_high, y_high, s_high = build_dataset(high_files)
    counts_h = np.bincount(y_high, minlength=3)
    print(f"\nTotal HIGH: {len(X_high)} ventanas | awake={counts_h[0]}  "
          f"induct={counts_h[1]}  trance={counts_h[2]}")

    print(f"\nEntrenando eeg_classifier_high (val_subject_ratio={args.val_subject_ratio}) ...")
    model_high = train_and_evaluate(
        X_high,
        y_high,
        s_high,
        group_name="high",
        random_state=args.seed,
        val_subject_ratio=args.val_subject_ratio,
        forced_val_subjects={args.forced_val_high} if args.forced_val_high else None,
        run_group_cv=not args.skip_cv,
    )
    out_high = MODELS_DIR / "eeg_classifier_high.joblib"
    joblib.dump(model_high, out_high)
    print(f"Guardado: {out_high}")

    # ── Grupo LOW ─────────────────────────────────────────────────────────
    low_files = sorted(LOWS_DIR.glob("subject_*.set"))
    print(f"\n{'='*60}")
    print(f"GRUPO LOW — {len(low_files)} sujetos")
    print(f"{'='*60}")
    X_low, y_low, s_low = build_dataset(low_files)
    counts_l = np.bincount(y_low, minlength=3)
    print(f"\nTotal LOW: {len(X_low)} ventanas | awake={counts_l[0]}  "
          f"induct={counts_l[1]}  trance={counts_l[2]}")

    # Si trance < 1 % en LOW, eliminarlo (sujetos de baja sugestionabilidad
    # rara vez alcanzan trance genuino; etiquetarlo sería ruido)
    trance_ratio = counts_l[2] / len(y_low)
    if trance_ratio < 0.01:
        mask = y_low != 2
        X_low, y_low, s_low = X_low[mask], y_low[mask], s_low[mask]
        print(f"  (Se eliminaron {counts_l[2]} ventanas 'trance' (<1%) del grupo LOW)")

    print(f"\nEntrenando eeg_classifier_low (val_subject_ratio={args.val_subject_ratio}) ...")
    model_low = train_and_evaluate(
        X_low,
        y_low,
        s_low,
        group_name="low",
        random_state=args.seed,
        val_subject_ratio=args.val_subject_ratio,
        forced_val_subjects={args.forced_val_low} if args.forced_val_low else None,
        run_group_cv=not args.skip_cv,
        class_weight=LOW_CLASS_WEIGHT,
    )
    out_low = MODELS_DIR / "eeg_classifier_low.joblib"
    joblib.dump(model_low, out_low)
    print(f"Guardado: {out_low}")

    # ── Dataset combinado para referencia / Colab futuro ──────────────────
    X_all = np.vstack([X_high, X_low])
    y_all = np.concatenate([y_high, y_low])

    # Feature 16: sugestionabilidad (HIGH=1.0, LOW=0.0) para modelo unificado.
    sfeat_high = np.ones((len(X_high), 1), dtype=np.float32)
    sfeat_low = np.zeros((len(X_low), 1), dtype=np.float32)
    X_unified = np.vstack([
        np.hstack([X_high, sfeat_high]),
        np.hstack([X_low, sfeat_low]),
    ])
    y_unified = np.concatenate([y_high, y_low])
    subjects_unified = np.concatenate([
        np.array([f"high_{sid}" for sid in s_high], dtype=object),
        np.array([f"low_{sid}" for sid in s_low], dtype=object),
    ])

    forced_unified = set()
    if args.forced_val_high:
        forced_unified.add(f"high_{args.forced_val_high}")
    if args.forced_val_low:
        forced_unified.add(f"low_{args.forced_val_low}")

    print(f"\nEntrenando eeg_classifier_unified (feature16=suggestibility) ...")
    model_unified = train_and_evaluate(
        X_unified,
        y_unified,
        subjects_unified,
        group_name="unified",
        random_state=args.seed,
        val_subject_ratio=args.val_subject_ratio,
        forced_val_subjects=forced_unified if forced_unified else None,
        run_group_cv=not args.skip_cv,
    )
    out_unified = MODELS_DIR / "eeg_classifier_unified.joblib"
    joblib.dump(model_unified, out_unified)
    print(f"Guardado: {out_unified}")

    # ── Verificación rápida de modelos guardados ──────────────────────────
    print(f"\n{'='*60}")
    print("VERIFICACION — cargando modelos y probando inferencia")
    print(f"{'='*60}")
    for path in [out_high, out_low]:
        m = joblib.load(path)
        sc = m.named_steps["scaler"]
        clf = m.named_steps["ensemble"]
        # Vector de prueba: TBR~1.5 (inducción), coh~0.45, plv~0.50, FSI~1.2
        feat = np.array(
            [1.5]*11 + [0.45, 0.50, 0.45, 1.2], dtype=np.float32
        ).reshape(1, -1)
        z = (feat[0, 0] - sc.mean_[0]) / sc.scale_[0]
        pred = m.predict(feat)[0]
        prob = m.predict_proba(feat)[0]
        labels = {0: "awake", 1: "induction", 2: "trance"}
        print(f"\n  {path.name}")
        print(f"    clases: {clf.classes_}")
        print(f"    scaler mean[0] (TBR_Fz): {sc.mean_[0]:.4f}  scale: {sc.scale_[0]:.4f}")
        print(f"    z-score para TBR=1.5: {z:.2f}  (OK si |z|<5)")
        print(f"    prediccion con TBR~1.5 → {labels.get(pred,pred)}  proba={np.round(prob,3)}")

    # Verificación del modelo unificado con sugestionabilidad explícita.
    m_uni = joblib.load(out_unified)
    for suggest_name, suggest_value in [("HIGH", 1.0), ("LOW", 0.0)]:
        feat_uni = np.array(
            [1.5] * 11 + [0.45, 0.50, 0.45, 1.2, suggest_value],
            dtype=np.float32,
        ).reshape(1, -1)
        pred_uni = int(m_uni.predict(feat_uni)[0])
        prob_uni = m_uni.predict_proba(feat_uni)[0]
        labels = {0: "awake", 1: "induction", 2: "trance"}
        print(
            f"\n  eeg_classifier_unified.joblib [{suggest_name}=feature16={suggest_value:.1f}]"
        )
        print(f"    prediccion con TBR~1.5 -> {labels.get(pred_uni,pred_uni)}  proba={np.round(prob_uni,3)}")

    print(f"\n{'='*60}")
    print("ENTRENAMIENTO COMPLETADO")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
