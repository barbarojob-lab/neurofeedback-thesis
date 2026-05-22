"""
pipeline.py  —  ml-service/src/

Función unificada process_window() que orquesta los tres componentes
del módulo de análisis de conectividad EEG:
  1. Coherencia espectral (Welch, banda theta 4–8 Hz)
  2. Phase Locking Value (Hilbert, banda theta 4–8 Hz)
  3. Clasificador ML de estado (awake / induction / trance)

Entrada:
    - Ventana EEG de ~2 s (N_CHANNELS × 512 muestras), ya re-referenciada y
        filtrada por el backend Node.js.
    - Potencias de banda por canal calculadas en Node.js y reutilizadas aquí.
  - Frontal specificity index (del server.ts existente).

Salida:
  - Dict JSON-serializable listo para enviar por WebSocket al frontend React.
    Claves: coherence_matrix, plv_matrix, channel_labels,
            connectivity_features, classifier_prediction,
            feature_vector, processing_ms.

Integración con server.ts:
  El backend Node.js se conecta a este servicio por WebSocket en el
  puerto 8001. Cada vez que procesa un hop (64 muestras, ~256 ms), envía
  la ventana completa de 512 muestras + band powers al servicio Python,
  recibe el resultado y lo incluye en el FeedbackPayload al frontend.

  Pseudo-código en server.ts:
    const mlResult = await mlService.processWindow({
      eeg_window:              multiChannelBuffer,   // float[][] [11][512]
      band_powers_per_channel: allChannelBandPowers, // Record<ch, BandPowers>
      frontal_specificity:     frontalSpecificity,   // number
    });
    // mlResult.classifier_prediction.predicted_label → "awake"|"induction"|"trance"
"""

import time
from typing import Any, Dict, Optional

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt
from sklearn.pipeline import Pipeline as SKPipelineType

from .classifier import (
    CLASS_LABELS,
    CONFIDENCE_THRESHOLD,
    MODEL_PATH_HIGH,
    MODEL_PATH_LOW,
    MODEL_PATH_UNIFIED,
    N_FEATURES,
    build_feature_vector,
    load_classifier,
    predict_state,
)
from .connectivity import (
    CHANNEL_INDEX,
    CHANNELS,
    N_CHANNELS,
    WINDOW_SIZE,
    compute_coherence_matrix,
    compute_plv_matrix,
    get_connectivity_features,
)


NOTCH_HZ = 50.0
BP_LOW = 1.0
BP_HIGH = 40.0

# ─── Umbrales heurísticos (fallback cuando no hay modelo ML) ────────────────────────
# Calibrados P25/P70 del dataset MIBD real (21-05-2025)
# Ref: Gruzelier et al., Hammond, Sabourin
TBR_TRANCE_HEUR = 1.418
COH_TRANCE_HEUR = 0.799
TBR_INDUCTION_HEUR = 0.514
COH_INDUCTION_HEUR = 0.544

# ─── Parámetros de fusión dual-model ──────────────────────────────────────────────
FUSION_MIN_CONFIDENCE = 0.65
FUSION_MIN_MARGIN = 0.10


def _bp_sos(fs: int) -> np.ndarray:
    nyq = fs / 2.0
    return butter(4, [BP_LOW / nyq, BP_HIGH / nyq], btype="band", output="sos")


def _notch_ba(fs: int) -> tuple[np.ndarray, np.ndarray]:
    return iirnotch(NOTCH_HZ, Q=30.0, fs=fs)


def _preprocess_window(eeg_window: np.ndarray, fs: int) -> np.ndarray:
    """Replica el preprocesamiento del entrenamiento sobre una ventana fija."""
    filtered = eeg_window.astype(np.float64, copy=True)
    notch_b, notch_a = _notch_ba(fs)
    bp_sos = _bp_sos(fs)

    for ch in range(filtered.shape[0]):
        filtered[ch] = filtfilt(notch_b, notch_a, filtered[ch])
        filtered[ch] = sosfiltfilt(bp_sos, filtered[ch])

    return filtered


def _compute_band_powers_per_channel(
    eeg_window: np.ndarray,
    fs: int,
) -> Dict[str, Dict[str, float]]:
    """Calcula potencias por banda replicando la logica usada en train_local.py."""
    n_samples = eeg_window.shape[1]
    df = fs / n_samples
    hann = np.hanning(n_samples)
    windowed = eeg_window * hann[np.newaxis, :]
    fft_all = np.fft.rfft(windowed, axis=-1)
    mags = np.abs(fft_all) / n_samples

    def band_power(f_lo: float, f_hi: float) -> np.ndarray:
        bin_lo = max(0, round(f_lo / df))
        bin_hi = min(n_samples // 2, round(f_hi / df))
        return np.sum(mags[:, bin_lo:bin_hi] ** 2, axis=-1)

    delta = band_power(1.0, 4.0)
    theta = band_power(4.0, 8.0)
    alpha = band_power(8.0, 12.0)
    beta = band_power(12.0, 30.0)
    gamma = band_power(30.0, 45.0)

    band_powers: Dict[str, Dict[str, float]] = {}
    for idx, channel in enumerate(CHANNELS):
        band_powers[channel] = {
            "delta": float(delta[idx]),
            "theta": float(theta[idx]),
            "alpha": float(alpha[idx]),
            "beta": float(beta[idx]),
            "gamma": float(gamma[idx]),
        }

    return band_powers


# ─────────────────────────────────────────────────────────────────────────────
# Cache del modelo (cargado una sola vez al primer process_window())
# ─────────────────────────────────────────────────────────────────────────────

# El modelo se guarda en memoria de proceso para evitar deserializar joblib
# en cada llamada. A 4 payloads/s, recargar el modelo implicaría ~4 lecturas
# de disco por segundo, que degradan la latencia del pipeline.
_cached_models: Dict[str, Optional[SKPipelineType]] = {
    "high": None,
    "low": None,
}
_models_loaded: Dict[str, bool] = {
    "high": False,
    "low": False,
}
_cached_unified_model: Optional[SKPipelineType] = None
_unified_model_loaded: bool = False


def _normalize_suggestibility(value: str) -> str:
    return "low" if str(value).lower() == "low" else "high"


def _get_model(suggestibility: str = "high") -> Optional[SKPipelineType]:
    """Retorna el modelo cacheado (high/low), intentando cargarlo la primera vez."""
    key = _normalize_suggestibility(suggestibility)
    if not _models_loaded[key]:
        model_path = MODEL_PATH_LOW if key == "low" else MODEL_PATH_HIGH
        _cached_models[key] = load_classifier(model_path=model_path)
        _models_loaded[key] = True
        if _cached_models[key] is None:
            print(
                f"[pipeline] No se encontró modelo entrenado para suggestibility='{key}'. "
                "Usando clasificador heurístico como fallback. "
                "Coloca los modelos en data/models_colab/ o ejecuta: python ml-service/train_local.py"
            )
    return _cached_models[key]


def _get_unified_model() -> Optional[SKPipelineType]:
    global _cached_unified_model, _unified_model_loaded
    if not _unified_model_loaded:
        _cached_unified_model = load_classifier(model_path=MODEL_PATH_UNIFIED)
        _unified_model_loaded = True
        if _cached_unified_model is None:
            print(
                "[pipeline] No se encontró modelo unificado (feature16=suggestibility). "
                "Usando ruta dual high/low como fallback."
            )
    return _cached_unified_model


def reload_model(suggestibility: Optional[str] = None) -> bool:
    """
    Fuerza la recarga del modelo desde disco.
    Llamar después de entrenar/actualizar el modelo.

    Returns:
        True si el modelo se cargó correctamente, False en caso contrario.
    """
    if suggestibility is None:
        global _unified_model_loaded
        _models_loaded["high"] = False
        _models_loaded["low"] = False
        _unified_model_loaded = False
        high_ok = _get_model("high") is not None
        low_ok = _get_model("low") is not None
        unified_ok = _get_unified_model() is not None
        return high_ok or low_ok or unified_ok

    key = _normalize_suggestibility(suggestibility)
    _models_loaded[key] = False
    model = _get_model(key)
    return model is not None


def _build_uncertain_prediction(
    candidates: list[tuple[str, Dict[str, Any]]],
    reason: str,
) -> Dict[str, Any]:
    """Construye una salida estable cuando high/low discrepan sin evidencia clara."""
    probs_acc = {CLASS_LABELS[k]: 0.0 for k in CLASS_LABELS}
    max_conf = 0.0

    for _, pred in candidates:
        probs = pred.get("class_probabilities", {})
        for label in probs_acc:
            probs_acc[label] += float(probs.get(label, 0.0))
        max_conf = max(max_conf, float(pred.get("confidence", 0.0)))

    n = max(1, len(candidates))
    avg_probs = {
        label: round(probs_acc[label] / n, 4)
        for label in probs_acc
    }

    return {
        "predicted_class": -1,
        "predicted_label": "uncertain",
        "confidence": round(max_conf, 4),
        "class_probabilities": avg_probs,
        "is_confident": False,
        "method": "ml_fused",
        "model_profile": "high+low",
        "selection_method": reason,
        "candidate_predictions": [
            {
                "profile": profile,
                "label": pred.get("predicted_label"),
                "confidence": pred.get("confidence"),
            }
            for profile, pred in candidates
        ],
    }


def _fuse_model_predictions(
    candidates: list[tuple[str, Dict[str, Any]]],
) -> tuple[Optional[str], Dict[str, Any]]:
    """Fusiona predicciones high/low con regla de acuerdo + margen de confianza."""
    if not candidates:
        raise ValueError("No hay candidatos para fusionar")

    if len(candidates) == 1:
        profile, pred = candidates[0]
        return profile, {
            **pred,
            "model_profile": profile,
            "selection_method": "single_model_available",
        }

    ordered = sorted(
        candidates,
        key=lambda item: float(item[1].get("confidence", 0.0)),
        reverse=True,
    )
    best_profile, best_pred = ordered[0]
    second_profile, second_pred = ordered[1]

    best_label = str(best_pred.get("predicted_label", ""))
    second_label = str(second_pred.get("predicted_label", ""))
    best_conf = float(best_pred.get("confidence", 0.0))
    second_conf = float(second_pred.get("confidence", 0.0))
    margin = best_conf - second_conf

    if best_label == second_label:
        return best_profile, {
            **best_pred,
            "model_profile": best_profile,
            "selection_method": "agreement_between_profiles",
            "agreement_label": best_label,
            "agreement_confidence_gap": round(margin, 4),
        }

    if best_conf >= FUSION_MIN_CONFIDENCE and margin >= FUSION_MIN_MARGIN:
        return best_profile, {
            **best_pred,
            "model_profile": best_profile,
            "selection_method": "disagreement_clear_margin",
            "runner_up_profile": second_profile,
            "runner_up_label": second_label,
            "confidence_margin": round(margin, 4),
        }

    return None, _build_uncertain_prediction(
        candidates,
        reason="disagreement_low_margin_or_confidence",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def process_window(
    eeg_window: np.ndarray,
    band_powers_per_channel: Dict[str, Dict[str, float]],
    frontal_specificity: float = 1.0,
    model_profile_mode: str = "auto",
    fs: int = 250,
) -> Dict[str, Any]:
    """
    Procesa una ventana EEG de 2 s y retorna métricas de conectividad
    y predicción de estado en un dict JSON-serializable.

    Este es el único punto de entrada que el servidor FastAPI (server.py)
    y el adaptador Node.js deben llamar.

    ── Flujo interno ───────────────────────────────────────────────────────

      eeg_window [N_CH × 512]
          │
          ├──► compute_coherence_matrix()  → coh_matrix [N_CH × N_CH]
          │         (Welch, theta 4–8 Hz)
          │
          ├──► compute_plv_matrix()        → plv_matrix [N_CH × N_CH]
          │         (Hilbert, theta 4–8 Hz)
          │
          ├──► get_connectivity_features() → dict de pares clave escalares
          │
          ├──► build_feature_vector()      → feature_vector [15]
          │         (TBR × 11, coh_Fz_Pz, plv_F3F4, plv_C3C4, FSI)
          │
          └──► predict_state()             → {class, confidence, probs}
                    (SVM+RF ensemble, o heurístico si no hay modelo)

    ── Rendimiento esperado ────────────────────────────────────────────────
      Coherencia (Welch, 11 canales × 55 pares) : ~10–20 ms en CPU
      PLV (Hilbert, 55 pares)                   :  ~5–10 ms en CPU
      Clasificador (predict_proba)              :  < 1 ms
      Total estimado                            : ~15–35 ms en CPU moderna

      Con el hop de 256 ms del backend, hay margen suficiente para completar
      el procesamiento antes del siguiente hop.

    Args:
        eeg_window:
            shape (N_CHANNELS, 512) — ventana de ~2.048 s a 250 Hz.
            Debe llegar re-referenciada, filtrada y en µV desde Node.js.
            Orden de canales: ["Fz", "Fp1", "F3", "C3", "Pz", "O1",
                               "F4", "C4", "P4", "O2", "Cz"]

        band_powers_per_channel:
            {canal: {banda: potencia_µV²}}
            Ejemplo: {"Fz": {"theta": 45.2, "beta": 12.1, "alpha": 8.3, ...}}
            Proveniente del BandPowerExtractor del backend Node.js.

        frontal_specificity:
            FSI = theta_Fz / media(theta_F3, theta_F4).
            Proveniente de server.ts (ya calculado en el pipeline principal).
            Default 1.0 (neutral) si no está disponible.

        fs:
            Frecuencia de muestreo en Hz. Default 250 Hz (OpenBCI).

    Returns:
        dict con:
          coherence_matrix      (list[list[float]]) : matriz NxN coherencia
          plv_matrix            (list[list[float]]) : matriz NxN PLV
          channel_labels        (list[str])         : nombres de canales en orden
          connectivity_features (dict[str, float])  : features escalares clave
          classifier_prediction (dict)              : clase + confianza + probs
          feature_vector        (list[float])       : vector de 15 features
            stage_timings_ms      (dict[str, float])  : desglose de tiempos internos
          processing_ms         (float)             : tiempo de cómputo [ms]
    """
    t0 = time.perf_counter()

    # ── Validación de entrada ──────────────────────────────────────────────
    if eeg_window.shape[0] != N_CHANNELS:
        raise ValueError(
            f"eeg_window debe tener {N_CHANNELS} canales en el eje 0, "
            f"recibido shape={eeg_window.shape}"
        )
    if eeg_window.shape[1] != WINDOW_SIZE:
        raise ValueError(
            f"eeg_window debe tener {WINDOW_SIZE} muestras en el eje 1, "
            f"recibido shape={eeg_window.shape}"
        )

    eeg_f64 = eeg_window.astype(np.float64, copy=False)

    # ── COMPONENTE 1: Coherencia espectral ────────────────────────────────
    coherence_t0 = time.perf_counter()
    coh_matrix = compute_coherence_matrix(eeg_f64, fs=fs)
    coherence_ms = (time.perf_counter() - coherence_t0) * 1000.0

    # ── COMPONENTE 2: PLV ─────────────────────────────────────────────────
    plv_t0 = time.perf_counter()
    plv_matrix = compute_plv_matrix(eeg_f64, fs=fs)
    plv_ms = (time.perf_counter() - plv_t0) * 1000.0

    # ── Features de conectividad (para WebSocket y clasificador) ──────────
    conn_features = get_connectivity_features(coh_matrix, plv_matrix)

    # ── COMPONENTE 3: Clasificador ML ─────────────────────────────────────
    fz_pz_coh = float(coh_matrix[CHANNEL_INDEX["Fz"], CHANNEL_INDEX["Pz"]])
    f3_f4_plv = float(plv_matrix[CHANNEL_INDEX["F3"], CHANNEL_INDEX["F4"]])
    c3_c4_plv = float(plv_matrix[CHANNEL_INDEX["C3"], CHANNEL_INDEX["C4"]])

    features_t0 = time.perf_counter()
    feature_vector = build_feature_vector(
        band_powers_per_channel=band_powers_per_channel,
        coherence_fz_pz=fz_pz_coh,
        plv_f3_f4=f3_f4_plv,
        plv_c3_c4=c3_c4_plv,
        frontal_specificity=frontal_specificity,
    )
    features_ms = (time.perf_counter() - features_t0) * 1000.0

    model_t0 = time.perf_counter()
    mode = str(model_profile_mode).lower()
    auto_mode = mode not in {"high", "low"}
    profiles = ("high", "low") if auto_mode else (mode,)

    model_candidates: list[tuple[str, Dict[str, Any]]] = []

    # Preferir modelo unificado con feature16=suggestibility cuando exista.
    unified_model = _get_unified_model()
    if unified_model is not None:
        for profile in profiles:
            suggestibility_feature = 0.0 if profile == "low" else 1.0
            fv_unified = np.concatenate(
                [feature_vector, np.array([suggestibility_feature], dtype=np.float32)]
            ).astype(np.float32)
            pred = predict_state(unified_model, fv_unified)
            pred = {
                **pred,
                "method": "ml_ensemble",
                "model_profile": profile,
                "model_family": "unified_with_suggestibility_feature",
            }
            model_candidates.append((profile, pred))
    else:
        for profile in profiles:
            model = _get_model(profile)
            if model is not None:
                pred = predict_state(model, feature_vector)
                model_candidates.append((profile, pred))

    selected_model_profile: Optional[str] = None
    if model_candidates:
        if auto_mode:
            selected_model_profile, prediction = _fuse_model_predictions(model_candidates)
        else:
            selected_model_profile, prediction = _fuse_model_predictions(model_candidates)
            prediction = {
                **prediction,
                "selection_method": f"forced_profile_{selected_model_profile}",
            }
    else:
        # Fallback heurístico: usa TBR Fz + coherencia Fz-Pz con umbrales
        # fijos de la literatura (Gruzelier, 2014) hasta que se entrene el modelo.
        prediction = _heuristic_fallback(
            band_powers_per_channel=band_powers_per_channel,
            coh_fz_pz=fz_pz_coh,
        )
    model_ms = (time.perf_counter() - model_t0) * 1000.0

    processing_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "coherence_matrix":      coh_matrix.tolist(),
        "plv_matrix":            plv_matrix.tolist(),
        "channel_labels":        CHANNELS,
        "connectivity_features": conn_features,
        "classifier_prediction": prediction,
        "feature_vector":        feature_vector.tolist(),
        "stage_timings_ms": {
            "preprocess_ms": 0.0,
            "coherence_ms": round(coherence_ms, 2),
            "plv_ms": round(plv_ms, 2),
            "feature_ms": round(features_ms, 2),
            "model_ms": round(model_ms, 2),
        },
        "processing_ms":         round(processing_ms, 2),
        "selected_model_profile": selected_model_profile,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Clasificador heurístico de fallback
# ─────────────────────────────────────────────────────────────────────────────

def _heuristic_fallback(
    band_powers_per_channel: Dict[str, Dict[str, float]],
    coh_fz_pz: float,
) -> Dict[str, Any]:
    """
    Clasificación basada en umbrales fijos cuando el modelo ML no está
    disponible (primera ejecución antes de correr train_classifier.py).

    Umbrales basados en literatura (Gruzelier, 2014; Hammond, 2011):
      TBR Fz:     < 1.0                      → awake
      TBR Fz:     1.0 – 2.5                  → induction
      TBR Fz:     > 2.5 + Coh Fz-Pz > 0.55  → trance

    Este método no aprende del paciente individual y tiene menor precisión
    que el clasificador ML, especialmente en sujetos con TBR basal atípico.

    Returns:
        dict con estructura idéntica a predict_state() + "method" key.
    """
    fz_powers = band_powers_per_channel.get("Fz", {})
    theta_fz  = max(float(fz_powers.get("theta", 1e-9)), 1e-9)
    beta_fz   = max(float(fz_powers.get("beta",  1e-9)), 1e-9)
    tbr_fz    = theta_fz / beta_fz

    if tbr_fz >= 2.5 and coh_fz_pz >= 0.55:
        state, conf = 2, 0.72   # trance: criterio doble TBR + coherencia
    elif tbr_fz >= 1.0 or coh_fz_pz >= 0.40:
        state, conf = 1, 0.60   # induction: TBR o coherencia en rango medio
    else:
        state, conf = 0, 0.68   # awake: ningún criterio de trance cumplido

    probs = {CLASS_LABELS[k]: 0.0 for k in CLASS_LABELS}
    probs[CLASS_LABELS[state]] = conf

    return {
        "predicted_class":     state,
        "predicted_label":     CLASS_LABELS[state],
        "confidence":          conf,
        "class_probabilities": probs,
        "is_confident":        True,
        "method":              "heuristic_fallback",
    }
