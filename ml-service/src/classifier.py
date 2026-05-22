"""
classifier.py  —  ml-service/src/

Clasificador de estados EEG para neurofeedback de trance hipnótico.

Clases objetivo:
  0 = awake      — Vigilia alerta: beta dominante, TBR < 1
  1 = induction  — Inducción: theta aumentando, TBR 1–2.5
  2 = trance     — Trance profundo: theta dominante, TBR > 2.5, coherencia Fz-Pz alta

Vector de features (15 dimensiones):
  [0:11]  — theta/beta ratio (TBR) por cada uno de los 11 canales
  [11]    — coherencia espectral Fz–Pz en banda theta
  [12]    — PLV interhemisférico F3–F4 en banda theta
  [13]    — PLV interhemisférico C3–C4 en banda theta
  [14]    — índice de especificidad frontal (FSI = theta_Fz / media(theta_F3, theta_F4))

Estrategia de clasificación:
  Ensemble de votación suave (soft voting) entre:
    1. SVM con kernel RBF  — captura fronteras no lineales en espacio de features
    2. Random Forest (100 árboles) — robusto a no-estacionariedad del EEG

  Soft voting: promedia las probabilidades de ambos modelos antes de decidir.
  Más calibrado que hard voting, permite reportar confianza informativa.

Evaluación:
  - Split estratificado 80/20 (mantiene proporción de clases en test)
  - Validación cruzada 5-fold sobre el conjunto de entrenamiento
  - Métricas: accuracy, F1 por clase, F1 macro, matriz de confusión

Referencias:
  - Gruzelier (2014): theta frontal como marcador de profundidad hipnótica
  - Hammond (2011): protocolos neurofeedback y umbrales TBR
  - Scheinost et al. (2014): coherencia theta en estados alterados
"""

import os
import warnings
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

# Etiquetas de clase con su interpretación clínica
CLASS_LABELS: Dict[int, str] = {
    0: "awake",
    1: "induction",
    2: "trance",
}

CLASS_DESCRIPTIONS: Dict[int, str] = {
    0: "Vigilia alerta  — arousal alto, beta dominante sobre theta",
    1: "Inducción       — theta aumentando, TBR 1–2.5, relajación inicial",
    2: "Trance profundo — theta dominante, TBR > 2.5, coherencia Fz-Pz elevada",
}

# Umbral mínimo de confianza para que la predicción se considere válida.
# Por debajo de este umbral, la UI debería mostrar "calibrando..." en lugar
# de la etiqueta de clase.
CONFIDENCE_THRESHOLD = 0.50

# ─── Umbrales heurísticos (fallback cuando no hay modelo) ────────────────────────────
# Calibrados con P20 del dataset MIBD: ~20% awake, ~60% induction, ~20% trance
TBR_TRANCE_HEUR = 2.0
COH_TRANCE_HEUR = 0.70
TBR_INDUCTION_HEUR = 0.45
COH_INDUCTION_HEUR = 0.45

# Directorio y rutas de persistencia del modelo
_THIS_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", "data", "models_colab"))

MODEL_PATH = os.path.join(MODEL_DIR, "eeg_classifier.joblib")
MODEL_PATH_HIGH = os.path.join(MODEL_DIR, "eeg_classifier_high.joblib")
MODEL_PATH_LOW = os.path.join(MODEL_DIR, "eeg_classifier_low.joblib")
MODEL_PATH_UNIFIED = os.path.join(MODEL_DIR, "eeg_classifier_unified.joblib")

# Número de features del vector de entrada
N_FEATURES = 15

# Canales en el orden esperado por build_feature_vector()
CHANNELS_ORDERED = [
    "Fz", "Fp1", "F3", "C3", "Pz", "O1", "F4", "C4", "P4", "O2", "Cz",
]


# ─────────────────────────────────────────────────────────────────────────────
# Construcción del vector de features
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_vector(
    band_powers_per_channel: Dict[str, Dict[str, float]],
    coherence_fz_pz: float,
    plv_f3_f4: float,
    plv_c3_c4: float,
    frontal_specificity: float,
) -> np.ndarray:
    """
    Construye el vector de features normalizado para el clasificador.

    Justificación de cada grupo de features:

      TBR por canal (11 features, índices 0–10):
        El theta/beta ratio es la medida núcleo del neurofeedback de trance.
        Usar el TBR canal a canal permite al clasificador capturar la
        distribución espacial del estado:
          - Fz/Cz: máxima discriminación (theta frontal midline en trance)
          - F3/F4: balance interhemisférico
          - O1/O2: supresión visual occipital (theta bajo en trance visual)
        Distribuciones típicas:
          awake:     TBR ~ 0.3–1.0  (beta > theta: arousal cortical)
          induction: TBR ~ 1.0–2.5  (theta igualando beta)
          trance:    TBR ~ 2.5–6.0  (theta dominante)
        (Gruzelier, 2014; Hammond, 2011)

      Coherencia Fz–Pz (índice 11):
        Marcador de profundidad hipnótica más replicado en EEG.
        Coherencia > 0.5 distingue trance de vigilia con p < 0.01.
        (Sabourin et al., 1990; Gruzelier et al., 2006)

      PLV F3–F4 (índice 12):
        Balance interhemisférico frontal.
        En trance: PLV aumenta por sincronización bilateral prefrontal.
        En asimetría emocional (estrés): PLV decrece.
        (Gruzelier, 2006; Basar et al., 2001)

      PLV C3–C4 (índice 13):
        Sincronización motora central.
        En trance profundo con inhibición motora: PLV tiende a bajar
        respecto a inducción activa, donde la corteza motora aún tiene
        control voluntario activo.

      Frontal Specificity Index — FSI (índice 14):
        FSI = theta(Fz) / media(theta(F3), theta(F4))
        Valida que el theta observado es realmente FRONTAL MIDLINE theta
        y no un artefacto difuso. FSI > 1.5 confirma trance genuino.
        (Criterio topográfico: Terhune et al., 2011)

    Args:
        band_powers_per_channel : {canal: {banda: potencia_µV²}}
        coherence_fz_pz         : scalar coherencia theta Fz–Pz
        plv_f3_f4               : scalar PLV theta F3–F4
        plv_c3_c4               : scalar PLV theta C3–C4
        frontal_specificity     : FSI (theta_Fz / media theta F3,F4)

    Returns:
        feature_vector : shape (15,), dtype float32
    """
    # ── Features 0–10: TBR por canal ──────────────────────────────────────
    tbr_features = []
    for ch in CHANNELS_ORDERED:
        powers = band_powers_per_channel.get(ch, {})
        # Clip en 1e-9 para evitar divisiones por cero.
        # En EEG real, la potencia beta nunca es literalmente 0; puede ser muy
        # baja durante supresión cortical pero siempre > ruido de fondo.
        theta = max(float(powers.get("theta", 1e-9)), 1e-9)
        beta  = max(float(powers.get("beta",  1e-9)), 1e-9)
        tbr   = theta / beta
        # Clamping del TBR: valores extremos (> 20) indican artefacto.
        # En trance profundo el TBR fisiológico rara vez supera 10.
        tbr_features.append(min(tbr, 20.0))

    # ── Features 11–14: conectividad + topografía ─────────────────────────
    connectivity_features = [
        float(np.clip(coherence_fz_pz,    0.0, 1.0)),
        float(np.clip(plv_f3_f4,          0.0, 1.0)),
        float(np.clip(plv_c3_c4,          0.0, 1.0)),
        float(np.clip(frontal_specificity, 0.0, 5.0)),  # FSI clampado a 5
    ]

    feature_vector = np.array(
        tbr_features + connectivity_features, dtype=np.float32
    )
    return feature_vector


# ─────────────────────────────────────────────────────────────────────────────
# Generación de datos de entrenamiento sintéticos
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_training_data(
    n_samples: int = 1200,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera datos de entrenamiento sintéticos con distribuciones
    fisiológicamente realistas para las 3 clases.

    ⚠️  IMPORTANTE: estos datos permiten validar la arquitectura del
    pipeline y obtener un baseline de demostración. Para un clasificador
    de uso clínico real, REEMPLAZAR con datos EEG etiquetados de sesiones
    reales de pacientes/sujetos (idealmente N ≥ 30 sujetos, sesiones
    balanceadas entre clases, con validación inter-sujeto).

    Distribuciones por clase (basadas en literatura):
    ────────────────────────────────────────────────────────────────────
    AWAKE (clase 0):
      TBR Fz:        ~ N(0.65, 0.20), clampado a [0.2, 2.0]
      TBR otros ch: ~ N(0.55, 0.18) — beta ligeramente dominante en todos
      Coh Fz–Pz:    ~ N(0.28, 0.08) — baja sincronización frontoparietal
      PLV F3–F4:    ~ N(0.38, 0.10) — balance interhemisférico moderado
      PLV C3–C4:    ~ N(0.32, 0.09)
      FSI:          ~ N(1.05, 0.15) — theta no especialmente frontal

    INDUCTION (clase 1):
      TBR Fz:        ~ N(1.60, 0.45) — theta igualando beta
      TBR otros ch: ~ N(1.40, 0.40)
      Coh Fz–Pz:    ~ N(0.47, 0.09) — sincronización aumentando
      PLV F3–F4:    ~ N(0.52, 0.10)
      PLV C3–C4:    ~ N(0.44, 0.09)
      FSI:          ~ N(1.32, 0.18)

    TRANCE (clase 2):
      TBR Fz:        ~ N(3.80, 0.80) — theta claramente dominante
      TBR otros ch: ~ N(3.20, 0.70)
      Coh Fz–Pz:    ~ N(0.72, 0.10) — alta coherencia frontoparietal
      PLV F3–F4:    ~ N(0.66, 0.09) — sincronización bilateral frontal
      PLV C3–C4:    ~ N(0.58, 0.09) — inhibición motora bilateral
      FSI:          ~ N(1.80, 0.25) — theta frontalmente específico
    ────────────────────────────────────────────────────────────────────

    Referencias: Gruzelier (2006, 2014), Sabourin (1990), Terhune (2011),
                 Hammond (2011), Jensen & Tesche (2002).

    Args:
        n_samples    : número total de muestras (dividido en 3 clases iguales)
        random_state : semilla para reproducibilidad

    Returns:
        X : shape (n_samples, N_FEATURES), dtype float32
        y : shape (n_samples,), dtype int32 — etiquetas 0/1/2
    """
    rng         = np.random.RandomState(random_state)
    n_per_class = n_samples // 3
    X_list      = []
    y_list      = []

    # ── Clase 0: Awake ─────────────────────────────────────────────────────
    for _ in range(n_per_class):
        # TBR canal Fz (más discriminante)
        tbr_fz    = float(np.clip(rng.normal(0.65, 0.20), 0.15, 2.0))
        # TBR otros 10 canales (ligeramente más bajos que Fz en vigilia)
        tbr_rest  = np.clip(rng.normal(0.55, 0.18, size=10), 0.10, 2.0)
        tbr_all   = np.concatenate([[tbr_fz], tbr_rest])

        coh_fz_pz = float(np.clip(rng.normal(0.28, 0.08), 0.05, 0.65))
        plv_f3_f4 = float(np.clip(rng.normal(0.38, 0.10), 0.10, 0.65))
        plv_c3_c4 = float(np.clip(rng.normal(0.32, 0.09), 0.08, 0.60))
        fsi       = float(np.clip(rng.normal(1.05, 0.15), 0.60, 1.80))

        feat = np.concatenate([tbr_all, [coh_fz_pz, plv_f3_f4, plv_c3_c4, fsi]])
        X_list.append(feat.astype(np.float32))
        y_list.append(0)

    # ── Clase 1: Induction ─────────────────────────────────────────────────
    for _ in range(n_per_class):
        tbr_fz    = float(np.clip(rng.normal(1.60, 0.45), 0.60, 3.5))
        tbr_rest  = np.clip(rng.normal(1.40, 0.40, size=10), 0.50, 3.0)
        tbr_all   = np.concatenate([[tbr_fz], tbr_rest])

        coh_fz_pz = float(np.clip(rng.normal(0.47, 0.09), 0.25, 0.75))
        plv_f3_f4 = float(np.clip(rng.normal(0.52, 0.10), 0.25, 0.78))
        plv_c3_c4 = float(np.clip(rng.normal(0.44, 0.09), 0.20, 0.70))
        fsi       = float(np.clip(rng.normal(1.32, 0.18), 0.85, 2.00))

        feat = np.concatenate([tbr_all, [coh_fz_pz, plv_f3_f4, plv_c3_c4, fsi]])
        X_list.append(feat.astype(np.float32))
        y_list.append(1)

    # ── Clase 2: Trance ────────────────────────────────────────────────────
    for _ in range(n_per_class):
        tbr_fz    = float(np.clip(rng.normal(3.80, 0.80), 2.0, 8.0))
        tbr_rest  = np.clip(rng.normal(3.20, 0.70, size=10), 1.5, 7.5)
        tbr_all   = np.concatenate([[tbr_fz], tbr_rest])

        coh_fz_pz = float(np.clip(rng.normal(0.72, 0.10), 0.45, 0.95))
        plv_f3_f4 = float(np.clip(rng.normal(0.66, 0.09), 0.40, 0.90))
        plv_c3_c4 = float(np.clip(rng.normal(0.58, 0.09), 0.35, 0.85))
        fsi       = float(np.clip(rng.normal(1.80, 0.25), 1.20, 3.00))

        feat = np.concatenate([tbr_all, [coh_fz_pz, plv_f3_f4, plv_c3_c4, fsi]])
        X_list.append(feat.astype(np.float32))
        y_list.append(2)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)

    # Mezclar aleatoriamente para que el orden no biase el entrenamiento
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    save_model: bool = True,
    verbose: bool = True,
    split_indices: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    groups: Optional[np.ndarray] = None,
    use_group_cv: bool = False,
    random_state: int = 42,
    test_size: float = 0.20,
    model_path: str = MODEL_PATH,
) -> Tuple[Pipeline, Dict]:
    """
    Entrena el ensemble clasificador de estados EEG y evalúa su rendimiento.

    Arquitectura del pipeline sklearn:
    ┌─────────────────────────────────────────────────────────────────┐
    │  StandardScaler                                                 │
    │    Normaliza cada feature a µ=0, σ=1.                          │
    │    CRÍTICO para SVM-RBF: la distancia euclidiana en el espacio  │
    │    de features es la base del kernel. Sin escalar, el TBR        │
    │    (rango 0.3–8) dominaría sobre coherencia/PLV (rango 0–1)     │
    │    dando un clasificador sesgado hacia el TBR.                  │
    │    Random Forest también se beneficia: mejora la convergencia   │
    │    de los splits al tener features en escala comparable.        │
    ├─────────────────────────────────────────────────────────────────┤
    │  VotingClassifier (soft voting)                                 │
    │  ├── SVC (kernel='rbf', C=1.0, gamma='scale')                  │
    │  │     Kernel RBF: K(x,y) = exp(−γ||x−y||²)                   │
    │  │     Captura fronteras no lineales entre clases.              │
    │  │     C=1.0: regularización moderada (no sobreajusta a EEG    │
    │  │     no-estacionario). probability=True habilita soft voting. │
    │  │     class_weight='balanced': compensa distribución desigual  │
    │  │     de sesiones clínicas (más awake que trance en datos reales)│
    │  │                                                              │
    │  └── RandomForestClassifier (n_estimators=100)                  │
    │        100 árboles con bootstrap sampling.                      │
    │        Robusto a features irrelevantes (Fz/Fp1 pueden tener     │
    │        mucho ruido en algunos sujetos).                         │
    │        min_samples_split=5: evita hojas con 1–2 muestras.      │
    │        Proporciona importancia de features (feature_importances_)│
    └─────────────────────────────────────────────────────────────────┘

    Evaluación:
      1. CV 5-fold sobre X_train (estimación robusta de generalización):
         StratifiedKFold garantiza proporción de clases en cada fold.
      2. Predict sobre X_test (hold-out definitivo, no visto durante CV).
      3. Reporte completo: classification_report (precision, recall, F1)
         y confusion matrix para identificar pares de clases confundidas.

    Args:
        X          : features, shape (n_samples, N_FEATURES)
        y          : etiquetas, shape (n_samples,), valores 0/1/2
        save_model : si True, serializa modelo en MODEL_PATH con joblib
        verbose    : si True, imprime reporte completo de evaluación

    Returns:
        model   : Pipeline entrenado listo para predict_proba()
        metrics : dict con todas las métricas (JSON-serializable)
    """
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Split train/test ───────────────────────────────────────────────────
    # Si se reciben índices predefinidos (split por sujeto), se respetan.
    # En caso contrario se usa split estratificado clásico por muestra.
    if split_indices is not None:
        train_idx, test_idx = split_indices
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

    # ── Definición de los modelos base ─────────────────────────────────────
    svm = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",       # γ = 1/(n_features · Var(X)) — adaptativo
        probability=True,    # necesario para soft voting y predict_proba()
        random_state=random_state,
        class_weight="balanced",
    )

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,          # árboles completos — la regularización
        min_samples_split=5,     # viene del ensemble, no de la profundidad
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,               # usar todos los cores disponibles
    )

    ensemble = VotingClassifier(
        estimators=[("svm", svm), ("rf", rf)],
        voting="soft",           # promedia probabilidades → más calibrado
        weights=[1, 1],          # mismo peso inicial; ajustar con datos reales
    )

    model = Pipeline([
        ("scaler",   StandardScaler()),
        ("ensemble", ensemble),
    ])

    # ── Validación cruzada 5-fold ANTES del entrenamiento final ───────────
    # Se evalúa sobre X_train para no contaminar X_test.
    # n_jobs=-1: paraleliza los folds en múltiples cores.
    train_groups = None
    if groups is not None:
        train_groups = groups[train_idx] if split_indices is not None else None

    if use_group_cv and train_groups is not None:
        unique_groups = np.unique(train_groups)
        n_splits = min(5, len(unique_groups))
        if n_splits < 2:
            raise ValueError("Se requieren al menos 2 sujetos en train para CV por grupos.")
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            groups=train_groups,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        split_strategy = "subject_holdout + stratified_group_kfold"
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
        )
        split_strategy = "stratified_sample_split + stratified_kfold"

    # ── Entrenamiento final en TODO X_train ───────────────────────────────
    model.fit(X_train, y_train)

    # ── Evaluación en test set (hold-out definitivo) ───────────────────────
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc          = accuracy_score(y_test, y_pred)
    f1_per_class = f1_score(y_test, y_pred, average=None)
    f1_macro     = f1_score(y_test, y_pred, average="macro")
    cm           = confusion_matrix(y_test, y_pred)

    labels_present = sorted(int(v) for v in np.unique(y))

    metrics = {
        "accuracy":          float(acc),
        "f1_macro":          float(f1_macro),
        "f1_per_class":      {
            CLASS_LABELS[label]: float(score)
            for label, score in zip(labels_present, f1_per_class)
        },
        "confusion_matrix":  cm.tolist(),
        "cv_mean_accuracy":  float(cv_scores.mean()),
        "cv_std_accuracy":   float(cv_scores.std()),
        "cv_scores":         cv_scores.tolist(),
        "n_train":           int(len(X_train)),
        "n_test":            int(len(X_test)),
        "n_features":        N_FEATURES,
        "feature_names":     _feature_names(),
        "split_strategy":    split_strategy,
        "labels_present":    labels_present,
    }

    if groups is not None:
        if split_indices is not None:
            metrics["n_train_subjects"] = int(len(np.unique(groups[train_idx])))
            metrics["n_test_subjects"] = int(len(np.unique(groups[test_idx])))
        else:
            metrics["n_train_subjects"] = None
            metrics["n_test_subjects"] = None

    if verbose:
        _print_training_report(metrics, y_test, y_pred)

    if save_model:
        joblib.dump(model, model_path)
        if verbose:
            print(f"[classifier] Modelo guardado -> {model_path}")

    return model, metrics


# ─────────────────────────────────────────────────────────────────────────────
# Carga e inferencia
# ─────────────────────────────────────────────────────────────────────────────

def load_classifier(model_path: str = MODEL_PATH) -> Optional[Pipeline]:
    """
    Carga el clasificador entrenado desde disco (joblib).

    Returns:
        Pipeline entrenado, o None si no existe modelo.
        Si retorna None, el pipeline de inferencia usará un clasificador
        heurístico como fallback (ver pipeline.py).
    """
    if not os.path.exists(model_path):
        return None
    try:
        model: Pipeline = joblib.load(model_path)
        return model
    except Exception as exc:
        print(f"[classifier] Error cargando modelo desde {model_path}: {exc}")
        return None


def predict_state(
    model: Pipeline,
    feature_vector: np.ndarray,
) -> Dict:
    """
    Inferencia en tiempo real: dado un vector de 15 features, devuelve
    la clase predicha con probabilidades y metadata.

    La confianza (probabilidad de la clase predicha) es el indicador
    principal para la UI: si confidence < CONFIDENCE_THRESHOLD, el
    sistema debería indicar "calibrando..." en lugar de mostrar la clase.

    Args:
        model          : Pipeline entrenado (StandardScaler + VotingClassifier)
        feature_vector : shape (15,) — output de build_feature_vector()

    Returns:
        dict JSON-serializable con:
          predicted_class       (int)   : 0=awake, 1=induction, 2=trance
          predicted_label       (str)   : nombre de la clase
          confidence            (float) : probabilidad de la clase predicha [0,1]
          class_probabilities   (dict)  : {label: prob} para las 3 clases
          is_confident          (bool)  : confidence >= CONFIDENCE_THRESHOLD
    """
    X          = feature_vector.reshape(1, -1)
    probs = model.predict_proba(X)[0]
    classes = [int(c) for c in model.classes_]
    pred_idx = int(np.argmax(probs))
    pred_class = classes[pred_idx]
    confidence = float(probs[pred_idx])

    full_probs = {CLASS_LABELS[k]: 0.0 for k in CLASS_LABELS}
    for label, prob in zip(classes, probs):
        full_probs[CLASS_LABELS[label]] = round(float(prob), 4)

    return {
        "predicted_class":     pred_class,
        "predicted_label":     CLASS_LABELS[pred_class],
        "confidence":          round(confidence, 4),
        "class_probabilities": full_probs,
        "is_confident":        confidence >= CONFIDENCE_THRESHOLD,
        "method":              "ml_ensemble",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades internas
# ─────────────────────────────────────────────────────────────────────────────

def _feature_names() -> list:
    """Retorna los nombres descriptivos de los 15 features, para logs y UI."""
    channels = ["Fz", "Fp1", "F3", "C3", "Pz", "O1", "F4", "C4", "P4", "O2", "Cz"]
    names    = [f"TBR_{ch}" for ch in channels]
    names   += ["coh_Fz_Pz", "plv_F3_F4", "plv_C3_C4", "frontal_specificity"]
    return names


def _print_training_report(metrics: Dict, y_test, y_pred) -> None:
    """Imprime el reporte de entrenamiento con formato legible."""
    sep = "=" * 65
    print(f"\n{sep}")
    print("  ENTRENAMIENTO — Clasificador EEG (awake / induction / trance)")
    print(sep)
    print(f"\n  Validación cruzada 5-fold (sobre train set):")
    print(f"    Accuracy: {metrics['cv_mean_accuracy']:.3f} ± {metrics['cv_std_accuracy']:.3f}")
    print(f"    Scores individuales: {[f'{s:.3f}' for s in metrics['cv_scores']]}")
    print(f"\n  Hold-out test set:")
    print(f"    Accuracy: {metrics['accuracy']:.3f}")
    print(f"    F1 macro: {metrics['f1_macro']:.3f}")
    labels_present = metrics.get("labels_present", sorted(int(v) for v in np.unique(y_test)))
    target_names = [CLASS_LABELS[i] for i in labels_present]

    print(f"\n  Reporte por clase:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels_present,
            target_names=target_names,
            digits=3,
        )
    )

    print(f"  Matriz de confusión (filas=real, columnas=predicho):")
    header = "    " + " " * 12 + "  " + "  ".join(f"{name:>10s}" for name in target_names)
    print(header)
    cm = metrics["confusion_matrix"]
    for row_idx, label_name in enumerate(target_names):
        row_vals = "  ".join(f"{int(v):>10d}" for v in cm[row_idx])
        print(f"    {label_name:12s}  {row_vals}")
    print(f"\n  Train: {metrics['n_train']} muestras | Test: {metrics['n_test']} muestras")
    print(sep + "\n")
