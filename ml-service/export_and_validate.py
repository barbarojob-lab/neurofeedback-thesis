"""
export_and_validate.py  —  ml-service/

Dos tareas en un solo script:

  1. Exportar BrainVision (.vhdr) → EDF estándar por sujeto
     → data/sub-001.edf  y  data/sub-002.edf

  2. Validación cruzada paciente-a-paciente (Leave-One-Subject-Out)
     → Entrena con sub-001, prueba en sub-002
     → Entrena con sub-002, prueba en sub-001
     → Reporta accuracy, F1, matriz de confusión por fold

Uso:
  python ml-service/export_and_validate.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

sys.path.insert(0, os.path.dirname(__file__))
from prepare_dataset import process_subject
from src.classifier import CHANNELS_ORDERED

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def _make_model() -> Pipeline:
    svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
              class_weight="balanced", random_state=42)
    rf  = RandomForestClassifier(n_estimators=100, min_samples_split=5,
                                 class_weight="balanced", random_state=42,
                                 n_jobs=-1)
    ensemble = VotingClassifier(estimators=[("svm", svm), ("rf", rf)],
                                voting="soft", weights=[1, 1])
    return Pipeline([("scaler", StandardScaler()), ("ensemble", ensemble)])

# ─── Rutas ────────────────────────────────────────────────────────────────────
ROOT  = os.path.dirname(os.path.dirname(__file__))
DATA  = os.path.join(ROOT, "data")
YOGA  = os.path.join(DATA, "yoga_eeg")

SUBJECTS = {
    "sub-001": {
        "vhdr": os.path.join(YOGA, "sub-001", "ses-01", "eeg",
                             "sub-001_ses-01_task-default_run-01_eeg.vhdr"),
        "tsv":  os.path.join(YOGA, "sub-001", "ses-01", "eeg",
                             "sub-001_ses-01_task-default_run-01_events.tsv"),
        "edf":  os.path.join(DATA, "sub-001.edf"),
    },
    "sub-002": {
        "vhdr": os.path.join(YOGA, "sub-002", "ses-01", "eeg",
                             "sub-002_ses-02_task-default_run-01_eeg.vhdr"),
        "tsv":  os.path.join(YOGA, "sub-002", "ses-01", "eeg",
                             "sub-002_ses-02_task-default_run-01_events.tsv"),
        "edf":  os.path.join(DATA, "sub-002.edf"),
    },
}

CLASS_NAMES = {0: "awake", 1: "induction", 2: "trance"}


# ─── 1. Exportar a EDF ────────────────────────────────────────────────────────
def export_to_edf(subject_id: str, info: dict) -> None:
    import mne
    edf_path = info["edf"]

    if os.path.exists(edf_path):
        print(f"  [skip] {edf_path} ya existe")
        return

    print(f"  Cargando {subject_id} ...")
    raw = mne.io.read_raw_brainvision(info["vhdr"], preload=True, verbose=False)
    raw.pick_channels(CHANNELS_ORDERED)
    raw.resample(250, verbose=False)
    raw.notch_filter(50.0, verbose=False)
    raw.filter(1.0, 40.0, verbose=False)

    print(f"  Exportando → {edf_path}")
    raw.export(edf_path, fmt="edf", verbose=False, overwrite=True)
    print(f"  OK — {raw.n_times} muestras, {raw.info['nchan']} canales, 250 Hz")


# ─── 2. Extraer features por sujeto ──────────────────────────────────────────
def extract(subject_id: str, info: dict):
    print(f"\n  Extrayendo features: {subject_id} ...")
    X, y = process_subject(info["vhdr"], info["tsv"])
    print(f"    {len(y)} ventanas → {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y


# ─── 3. Validación cruzada LOSO ──────────────────────────────────────────────
def run_loso(features: dict) -> None:
    subs = list(features.keys())
    fold_results = []

    for test_sub in subs:
        train_sub = [s for s in subs if s != test_sub][0]

        X_train, y_train = features[train_sub]
        X_test,  y_test  = features[test_sub]

        print(f"\n  ── Fold: train={train_sub}  test={test_sub} ──")
        print(f"    Train: {len(y_train)} ventanas")
        print(f"    Test : {len(y_test)} ventanas")

        model = _make_model()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="macro", zero_division=0)

        print(f"\n    Accuracy: {acc:.3f}")
        print(f"    F1 macro: {f1:.3f}")
        print()
        print(classification_report(
            y_test, y_pred,
            target_names=["awake", "induction", "trance"],
            zero_division=0,
        ))

        cm = confusion_matrix(y_test, y_pred)
        print("    Matriz de confusión (filas=real, columnas=predicho):")
        header = f"{'':15s}" + "".join(f"{n:12s}" for n in ["awake", "induction", "trance"])
        print("   ", header)
        for i, row in enumerate(cm):
            row_str = f"{CLASS_NAMES[i]:15s}" + "".join(f"{v:12d}" for v in row)
            print("   ", row_str)

        fold_results.append({"train": train_sub, "test": test_sub, "acc": acc, "f1": f1})

    print("\n" + "=" * 60)
    print("  RESUMEN LOSO")
    print("=" * 60)
    mean_acc = np.mean([r["acc"] for r in fold_results])
    mean_f1  = np.mean([r["f1"]  for r in fold_results])
    for r in fold_results:
        print(f"  {r['train']} → {r['test']}:  acc={r['acc']:.3f}  f1={r['f1']:.3f}")
    print(f"\n  Media:  acc={mean_acc:.3f}  f1={mean_f1:.3f}")
    print()
    print("  Interpretación:")
    if mean_acc >= 0.70:
        print("  ✅ Buena generalización entre sujetos (>70%)")
    elif mean_acc >= 0.55:
        print("  ⚠️  Generalización moderada (55–70%) — esperable con 2 sujetos")
    else:
        print("  ❌ Baja generalización — revisar calidad de datos o features")
    print()
    print("  Nota científica:")
    print("  La validación LOSO con 2 sujetos mide generalización real.")
    print("  El modelo no vio datos del sujeto de prueba durante entrenamiento.")
    print("=" * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  PASO 1 — EXPORTAR BrainVision → EDF")
    print("=" * 60)
    for sid, info in SUBJECTS.items():
        export_to_edf(sid, info)

    print("\n" + "=" * 60)
    print("  PASO 2 — EXTRACCIÓN DE FEATURES")
    print("=" * 60)
    features = {}
    for sid, info in SUBJECTS.items():
        features[sid] = extract(sid, info)

    print("\n" + "=" * 60)
    print("  PASO 3 — VALIDACIÓN CRUZADA (LOSO)")
    print("=" * 60)
    run_loso(features)

    print("\nEDFs guardados en:")
    for sid, info in SUBJECTS.items():
        if os.path.exists(info["edf"]):
            size_mb = os.path.getsize(info["edf"]) / 1e6
            print(f"  {info['edf']}  ({size_mb:.1f} MB)")
    print("\nCarga estos EDFs en la UI: campo DATASET → CARGAR DATASET → INICIAR SESIÓN")


if __name__ == "__main__":
    main()
