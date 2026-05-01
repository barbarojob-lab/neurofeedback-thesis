"""
train_classifier.py  —  ml-service/

Script CLI standalone para entrenar y evaluar el clasificador EEG.

Uso:
  # Entrenamiento con datos SINTÉTICOS (demo/validación de arquitectura):
  python train_classifier.py

  # Con más muestras y semilla personalizada:
  python train_classifier.py --n-samples 3000 --seed 123

  # Entrenamiento con datos REALES etiquetados:
  python train_classifier.py --real data/sessions.npz

    El archivo .npz debe contener:
      X : array float32 shape (n_samples, 15) — features
      y : array int32   shape (n_samples,)    — etiquetas 0/1/2

  # Ver help:
  python train_classifier.py --help

Pasos para preparar datos reales desde las sesiones de tu plataforma:
  1. Activar logging en server.ts: guardar feature_vector de cada epoch
     con la etiqueta de clase asignada por el terapeuta (0/1/2).
  2. Exportar a CSV → convertir a NumPy y guardar como .npz:
       import numpy as np
       np.savez("data/sessions.npz", X=X_array, y=y_array)
  3. Ejecutar: python train_classifier.py --real data/sessions.npz
"""

import argparse
import os
import sys
from typing import Dict, Optional

import numpy as np

# Añadir el directorio raíz del módulo al path para imports relativos
sys.path.insert(0, os.path.dirname(__file__))

from src.classifier import (
    CLASS_LABELS,
    MODEL_PATH,
    MODEL_PATH_HIGH,
    MODEL_PATH_LOW,
    N_FEATURES,
    _feature_names,
    generate_synthetic_training_data,
    train_classifier,
)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Entrena el clasificador de estados EEG "
            "(awake / induction / trance)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--real",
        type=str,
        default=None,
        metavar="PATH",
        help="Ruta a .npz con arrays X (features) e y (etiquetas) reales.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1500,
        metavar="N",
        help="Número de muestras sintéticas totales (default: 1500, ~500/clase).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Semilla aleatoria para reproducibilidad (default: 42).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suprimir reporte detallado (solo imprimir métricas finales).",
    )
    parser.add_argument(
        "--ignore-subject-split",
        action="store_true",
        help=(
            "Ignora arrays split/subject_id del .npz y usa split estratificado "
            "clásico por muestras."
        ),
    )
    parser.add_argument(
        "--suggestibility-model",
        choices=["both", "high", "low"],
        default="both",
        help=(
            "Perfil de modelo a entrenar con datos reales. 'high' y 'low' "
            "guardan modelos separados; 'both' entrena el modelo general."
        ),
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Carga / generación de datos
# ─────────────────────────────────────────────────────────────────────────────

def load_real_data(path: str) -> tuple:
    """
    Carga datos reales desde un archivo .npz.

    Valida que X tenga la forma correcta y que y solo contenga
    etiquetas 0, 1 o 2.
    """
    if not os.path.exists(path):
        print(f"[error] Archivo no encontrado: {path}")
        sys.exit(1)

    data = np.load(path)

    if "X" not in data or "y" not in data:
        print(f"[error] El archivo .npz debe contener arrays 'X' e 'y'.")
        sys.exit(1)

    X, y = data["X"], data["y"]
    meta: Dict[str, Optional[np.ndarray]] = {
        "subject_id": data["subject_id"] if "subject_id" in data else None,
        "split": data["split"] if "split" in data else None,
        "suggestibility": data["suggestibility"] if "suggestibility" in data else None,
    }

    # Validar shape de X
    if X.ndim != 2 or X.shape[1] != N_FEATURES:
        print(
            f"[error] X debe tener shape (n_samples, {N_FEATURES}), "
            f"recibido: {X.shape}"
        )
        print(f"        Features esperados: {_feature_names()}")
        sys.exit(1)

    # Validar etiquetas
    invalid = set(np.unique(y)) - {0, 1, 2}
    if invalid:
        print(f"[error] y contiene etiquetas inválidas: {invalid}. Solo 0, 1, 2.")
        sys.exit(1)

    return X.astype(np.float32), y.astype(np.int32), meta


def _print_subject_split_summary(subject_ids: np.ndarray, split: np.ndarray) -> None:
    train_subjects = sorted(np.unique(subject_ids[split == "train"]).tolist())
    test_subjects = sorted(np.unique(subject_ids[split == "test"]).tolist())
    print("\n  Split por sujeto detectado en dataset:")
    print(f"    Sujetos train: {len(train_subjects)}")
    print(f"    Sujetos test : {len(test_subjects)}")


def print_data_summary(X: np.ndarray, y: np.ndarray, source: str) -> None:
    """Imprime resumen estadístico de los datos antes del entrenamiento."""
    print(f"\n  Fuente de datos : {source}")
    print(f"  Muestras totales: {len(X)}")
    print(f"  Features        : {X.shape[1]}")
    print(f"\n  Distribución de clases:")
    for cls in sorted(int(v) for v in np.unique(y)):
        n = int((y == cls).sum())
        pct = 100 * n / len(y)
        print(f"    {cls} ({CLASS_LABELS[cls]:10s}): {n:5d} muestras ({pct:.1f}%)")

    print(f"\n  Estadísticas de features (primeras 5 + conectividad):")
    names = _feature_names()
    idxs  = [0, 2, 4, 6, 10, 11, 12, 13, 14]  # Fz, F3, C3, C4, Cz + conn
    for i in idxs:
        col = X[:, i]
        print(
            f"    {names[i]:25s}: "
            f"mean={col.mean():.3f}  std={col.std():.3f}  "
            f"min={col.min():.3f}  max={col.max():.3f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    sep = "=" * 65
    print(f"\n{sep}")
    print("  ENTRENAMIENTO — Clasificador EEG (awake / induction / trance)")
    print(sep)

    # ── Carga o generación de datos ───────────────────────────────────────
    if args.real:
        print(f"\n  Modo: datos reales")
        print(f"  Cargando: {args.real}")
        X, y, meta = load_real_data(args.real)
    else:
        print(f"\n  Modo: datos sintéticos (seed={args.seed}, n={args.n_samples})")
        print(
            "  ⚠️  Para uso clínico real, entrenar con datos EEG etiquetados.\n"
            "     Los datos sintéticos solo validan la arquitectura del pipeline."
        )
        X, y = generate_synthetic_training_data(
            n_samples=args.n_samples,
            random_state=args.seed,
        )
        meta = {"subject_id": None, "split": None}

    if args.real and args.suggestibility_model in {"high", "low"}:
        sug = meta.get("suggestibility")
        if sug is None:
            print("\n  [error] El dataset no contiene array 'suggestibility' para filtrar high/low.")
            sys.exit(1)
        mask = (sug.astype(str) == args.suggestibility_model)
        if not np.any(mask):
            print(f"\n  [error] No hay muestras para suggestibility={args.suggestibility_model}.")
            sys.exit(1)
        X = X[mask]
        y = y[mask]
        for key in ["subject_id", "split", "suggestibility"]:
            if meta.get(key) is not None:
                meta[key] = meta[key][mask]
        print(f"\n  Filtrado por suggestibility_model={args.suggestibility_model}")

    print_data_summary(X, y, source=args.real or "sintético")

    # ── Entrenamiento ─────────────────────────────────────────────────────
    print(f"\n  Entrenando SVM (RBF) + Random Forest (100 árboles)...")
    print(f"  Ensemble: soft voting | Split: segun dataset/configuracion | CV: 5-fold")
    print()

    split_indices = None
    groups = None
    use_group_cv = False

    if args.real and not args.ignore_subject_split:
        subject_id = meta.get("subject_id")
        split = meta.get("split")
        if subject_id is not None and split is not None:
            split = split.astype(str)
            train_idx = np.where(split == "train")[0]
            test_idx = np.where(split == "test")[0]
            if len(train_idx) > 0 and len(test_idx) > 0:
                split_indices = (train_idx, test_idx)
                groups = subject_id.astype(str)
                use_group_cv = True
                _print_subject_split_summary(groups, split)
            else:
                print("\n  [warn] split detectado pero inválido; usando split estratificado por muestras.")

    target_model_path = MODEL_PATH
    if args.suggestibility_model == "high":
        target_model_path = MODEL_PATH_HIGH
    elif args.suggestibility_model == "low":
        target_model_path = MODEL_PATH_LOW

    model, metrics = train_classifier(
        X,
        y,
        save_model=True,
        verbose=not args.quiet,
        split_indices=split_indices,
        groups=groups,
        use_group_cv=use_group_cv,
        random_state=args.seed,
        model_path=target_model_path,
    )

    # ── Resumen final ─────────────────────────────────────────────────────
    if args.quiet:
        print(f"\n  Split strategy:       {metrics['split_strategy']}")
        print(f"\n  Accuracy test:        {metrics['accuracy']:.4f}")
        print(f"  F1 macro:             {metrics['f1_macro']:.4f}")
        print(f"  CV accuracy (5-fold): {metrics['cv_mean_accuracy']:.4f}"
              f" ± {metrics['cv_std_accuracy']:.4f}")
        if metrics.get("n_train_subjects") is not None:
            print(f"  Train/Test sujetos:   {metrics['n_train_subjects']} / {metrics['n_test_subjects']}")
        print(f"\n  F1 por clase:")
        for cls, f1 in metrics["f1_per_class"].items():
            print(f"    {cls:12s}: {f1:.4f}")

    print(f"\n  Modelo guardado en: {target_model_path}")
    print(f"  Listo para inferencia tiempo real (python server.py)")
    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
