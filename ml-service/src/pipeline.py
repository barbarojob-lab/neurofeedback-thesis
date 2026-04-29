"""
pipeline.py  —  ml-service/src/

Función unificada process_window() que orquesta los tres componentes
del módulo de análisis de conectividad EEG:
  1. Coherencia espectral (Welch, banda theta 4–8 Hz)
  2. Phase Locking Value (Hilbert, banda theta 4–8 Hz)
  3. Clasificador ML de estado (awake / induction / trance)

Entrada:
  - Ventana EEG de 2 s (N_CHANNELS × 500 muestras), ya filtrada por la
    cadena notch+bandpass del backend Node.js existente.
  - Potencias de banda por canal (del BandPowerExtractor existente).
  - Frontal specificity index (del server.ts existente).

Salida:
  - Dict JSON-serializable listo para enviar por WebSocket al frontend React.
    Claves: coherence_matrix, plv_matrix, channel_labels,
            connectivity_features, classifier_prediction,
            feature_vector, processing_ms.

Integración con server.ts:
  El backend Node.js se conecta a este servicio por WebSocket en el
  puerto 8001. Cada vez que procesa un hop (64 muestras, ~256 ms), envía
  la ventana completa de 500 muestras + band powers al servicio Python,
  recibe el resultado y lo incluye en el FeedbackPayload al frontend.

  Pseudo-código en server.ts:
    const mlResult = await mlService.processWindow({
      eeg_window:              multiChannelBuffer,   // float[][] [11][500]
      band_powers_per_channel: allChannelBandPowers, // Record<ch, BandPowers>
      frontal_specificity:     frontalSpecificity,   // number
    });
    // mlResult.classifier_prediction.predicted_label → "awake"|"induction"|"trance"
"""

import time
from typing import Any, Dict, Optional

import numpy as np
from sklearn.pipeline import Pipeline as SKPipelineType

from .classifier import (
    CLASS_LABELS,
    CONFIDENCE_THRESHOLD,
    N_FEATURES,
    build_feature_vector,
    load_classifier,
    predict_state,
)
from .connectivity import (
    CHANNEL_INDEX,
    CHANNELS,
    N_CHANNELS,
    compute_coherence_matrix,
    compute_plv_matrix,
    get_connectivity_features,
)


# ─────────────────────────────────────────────────────────────────────────────
# Cache del modelo (cargado una sola vez al primer process_window())
# ─────────────────────────────────────────────────────────────────────────────

# El modelo se guarda en memoria de proceso para evitar deserializar joblib
# en cada llamada. A 4 payloads/s, recargar el modelo implicaría ~4 lecturas
# de disco por segundo, que degradan la latencia del pipeline.
_cached_model: Optional[SKPipelineType] = None
_model_loaded: bool = False  # flag para no reintentar si falló la carga


def _get_model() -> Optional[SKPipelineType]:
    """Retorna el modelo cacheado, intentando cargarlo la primera vez."""
    global _cached_model, _model_loaded
    if not _model_loaded:
        _cached_model = load_classifier()
        _model_loaded = True
        if _cached_model is None:
            print(
                "[pipeline] No se encontró modelo entrenado. "
                "Usando clasificador heurístico como fallback. "
                "Ejecuta: python train_classifier.py"
            )
    return _cached_model


def reload_model() -> bool:
    """
    Fuerza la recarga del modelo desde disco.
    Llamar después de entrenar/actualizar el modelo.

    Returns:
        True si el modelo se cargó correctamente, False en caso contrario.
    """
    global _cached_model, _model_loaded
    _model_loaded = False
    model = _get_model()
    return model is not None


# ─────────────────────────────────────────────────────────────────────────────
# Función principal
# ─────────────────────────────────────────────────────────────────────────────

def process_window(
    eeg_window: np.ndarray,
    band_powers_per_channel: Dict[str, Dict[str, float]],
    frontal_specificity: float = 1.0,
    fs: int = 250,
) -> Dict[str, Any]:
    """
    Procesa una ventana EEG de 2 s y retorna métricas de conectividad
    y predicción de estado en un dict JSON-serializable.

    Este es el único punto de entrada que el servidor FastAPI (server.py)
    y el adaptador Node.js deben llamar.

    ── Flujo interno ───────────────────────────────────────────────────────

      eeg_window [N_CH × 500]
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
            shape (N_CHANNELS, 500) — ventana de 2 s a 250 Hz.
            ⚠️  Ya debe estar filtrada (notch 50 Hz + bandpass 1–40 Hz)
            por el pipeline existente en server.ts. NO re-filtrar aquí.
            Orden de canales: ["Fz", "Fp1", "F3", "C3", "Pz", "O1",
                               "F4", "C4", "P4", "O2", "Cz"]

        band_powers_per_channel:
            {canal: {banda: potencia_µV²}}
            Ejemplo: {"Fz": {"theta": 45.2, "beta": 12.1, "alpha": 8.3, ...}}
            Proveniente del BandPowerExtractor existente del backend Node.js.

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
          processing_ms         (float)             : tiempo de cómputo [ms]
    """
    t0 = time.perf_counter()

    # ── Validación de entrada ──────────────────────────────────────────────
    if eeg_window.shape[0] != N_CHANNELS:
        raise ValueError(
            f"eeg_window debe tener {N_CHANNELS} canales en el eje 0, "
            f"recibido shape={eeg_window.shape}"
        )

    # Asegurar float64 para operaciones de scipy (coherence/hilbert)
    eeg_f64 = eeg_window.astype(np.float64)

    # ── COMPONENTE 1: Coherencia espectral ────────────────────────────────
    coh_matrix = compute_coherence_matrix(eeg_f64, fs=fs)

    # ── COMPONENTE 2: PLV ─────────────────────────────────────────────────
    plv_matrix = compute_plv_matrix(eeg_f64, fs=fs)

    # ── Features de conectividad (para WebSocket y clasificador) ──────────
    conn_features = get_connectivity_features(coh_matrix, plv_matrix)

    # ── COMPONENTE 3: Clasificador ML ─────────────────────────────────────
    fz_pz_coh = float(coh_matrix[CHANNEL_INDEX["Fz"], CHANNEL_INDEX["Pz"]])
    f3_f4_plv = float(plv_matrix[CHANNEL_INDEX["F3"], CHANNEL_INDEX["F4"]])
    c3_c4_plv = float(plv_matrix[CHANNEL_INDEX["C3"], CHANNEL_INDEX["C4"]])

    feature_vector = build_feature_vector(
        band_powers_per_channel=band_powers_per_channel,
        coherence_fz_pz=fz_pz_coh,
        plv_f3_f4=f3_f4_plv,
        plv_c3_c4=c3_c4_plv,
        frontal_specificity=frontal_specificity,
    )

    model = _get_model()
    if model is not None:
        prediction = predict_state(model, feature_vector)
    else:
        # Fallback heurístico: usa TBR Fz + coherencia Fz-Pz con umbrales
        # fijos de la literatura (Gruzelier, 2014) hasta que se entrene el modelo.
        prediction = _heuristic_fallback(
            band_powers_per_channel=band_powers_per_channel,
            coh_fz_pz=fz_pz_coh,
        )

    processing_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "coherence_matrix":      coh_matrix.tolist(),
        "plv_matrix":            plv_matrix.tolist(),
        "channel_labels":        CHANNELS,
        "connectivity_features": conn_features,
        "classifier_prediction": prediction,
        "feature_vector":        feature_vector.tolist(),
        "processing_ms":         round(processing_ms, 2),
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
