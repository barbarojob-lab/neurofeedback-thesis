# ml-service/src/__init__.py
# Expone la API pública del módulo de análisis EEG.

from .connectivity import (
    CHANNELS,
    N_CHANNELS,
    CHANNEL_INDEX,
    compute_coherence_matrix,
    compute_plv_matrix,
    get_connectivity_features,
)

from .classifier import (
    CLASS_LABELS,
    N_FEATURES,
    build_feature_vector,
    generate_synthetic_training_data,
    train_classifier,
    load_classifier,
    predict_state,
)

from .pipeline import process_window, reload_model

__all__ = [
    "CHANNELS", "N_CHANNELS", "CHANNEL_INDEX",
    "compute_coherence_matrix", "compute_plv_matrix", "get_connectivity_features",
    "CLASS_LABELS", "N_FEATURES",
    "build_feature_vector", "generate_synthetic_training_data",
    "train_classifier", "load_classifier", "predict_state",
    "process_window", "reload_model",
]
