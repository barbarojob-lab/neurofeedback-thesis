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

import numpy as np

# Añadir el directorio raíz del módulo al path para imports relativos
sys.path.insert(0, os.path.dirname(__file__))

from src.classifier import (
    CLASS_LABELS,
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

    return X.astype(np.float32), y.astype(np.int32)


def print_data_summary(X: np.ndarray, y: np.ndarray, source: str) -> None:
    """Imprime resumen estadístico de los datos antes del entrenamiento."""
    print(f"\n  Fuente de datos : {source}")
    print(f"  Muestras totales: {len(X)}")
    print(f"  Features        : {X.shape[1]}")
    print(f"\n  Distribución de clases:")
    for cls in [0, 1, 2]:
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
        X, y = load_real_data(args.real)
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

    print_data_summary(X, y, source=args.real or "sintético")

    # ── Entrenamiento ─────────────────────────────────────────────────────
    print(f"\n  Entrenando SVM (RBF) + Random Forest (100 árboles)...")
    print(f"  Ensemble: soft voting | Split: 80/20 estratificado | CV: 5-fold")
    print()

    model, metrics = train_classifier(
        X, y,
        save_model=True,
        verbose=not args.quiet,
    )

    # ── Resumen final ─────────────────────────────────────────────────────
    if args.quiet:
        print(f"\n  Accuracy test:        {metrics['accuracy']:.4f}")
        print(f"  F1 macro:             {metrics['f1_macro']:.4f}")
        print(f"  CV accuracy (5-fold): {metrics['cv_mean_accuracy']:.4f}"
              f" ± {metrics['cv_std_accuracy']:.4f}")
        print(f"\n  F1 por clase:")
        for cls, f1 in metrics["f1_per_class"].items():
            print(f"    {cls:12s}: {f1:.4f}")

    print(f"\n  Modelo guardado en: ml-service/models/eeg_classifier.joblib")
    print(f"  Listo para inferencia tiempo real (python server.py)")
    print(f"\n{sep}\n")


if __name__ == "__main__":
    main()
