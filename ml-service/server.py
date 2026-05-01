"""
server.py  —  ml-service/

Servidor FastAPI que expone el módulo de análisis de conectividad y
clasificación EEG al backend Node.js y al frontend React.

Endpoints:
  GET  /health          — health check + estado del modelo
  POST /process_window  — procesa una ventana EEG (REST)
  POST /train           — entrena el clasificador (REST)
  POST /reload_model    — recarga el modelo desde disco sin reiniciar
  WS   /ws              — WebSocket tiempo real para server.ts

Integración con server.ts:
  El backend Node.js actúa como cliente WebSocket de este servicio.
  Cada hop (~256 ms), envía un mensaje JSON:
    {
      "type": "process_window",
      "data": {
        "eeg_window":              [[ch1_s1, ...], [ch2_s1, ...]],  // 11×500
        "band_powers_per_channel": {"Fz": {"theta": 45.2, ...}, ...},
        "frontal_specificity":     1.8
      }
    }

  Recibe la respuesta:
    {
      "type": "connectivity_result",
      "coherence_matrix":      [...],
      "plv_matrix":            [...],
      "channel_labels":        [...],
      "connectivity_features": {...},
      "classifier_prediction": {"predicted_label": "trance", ...},
      "processing_ms":         22.4
    }

  Y la agrega al FeedbackPayload que ya envía al frontend React.

Iniciar el servicio:
  Development:  python server.py
  Production:   uvicorn server:app --host 0.0.0.0 --port 8001 --workers 1
                (workers=1: el modelo en memoria no es thread-safe si
                 fuera multi-worker sin lock; para alta concurrencia
                 usar múltiples procesos con modelo cargado por proceso)
"""

import json

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from src.classifier import generate_synthetic_training_data, train_classifier
from src.connectivity import CHANNELS, N_CHANNELS
from src.pipeline import process_window, reload_model, _get_model


# ─────────────────────────────────────────────────────────────────────────────
# Aplicación FastAPI
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="EEG Connectivity & Classification Service",
    description=(
        "Análisis de conectividad (coherencia, PLV) y clasificación de "
        "estados EEG (awake/induction/trance) en tiempo real."
    ),
    version="1.0.0",
)

# CORS: permite requests desde el frontend React (dev) y desde el backend Node.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server (React)
        "http://localhost:3000",   # Create React App dev server
        "http://localhost:8080",   # Backend Node.js (para REST)
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Modelos Pydantic (validación de inputs)
# ─────────────────────────────────────────────────────────────────────────────

class ProcessWindowRequest(BaseModel):
    """
    Payload para POST /process_window y WS message type='process_window'.

    eeg_window:
        Lista de N_CHANNELS listas de flotantes.
        Cada sublista contiene n_samples (normalmente 512) muestras en µV.
        Ya debe estar filtrada (notch + bandpass) por el backend Node.js.

    band_powers_per_channel:
        {canal: {banda: potencia_µV²}}
        Generado por BandPowerExtractor del pipeline existente.

    frontal_specificity:
        FSI = theta(Fz) / media(theta(F3), theta(F4)).
        Del server.ts existente. Default 1.0 si no disponible.
    """
    eeg_window:              list[list[float]]
    band_powers_per_channel: dict[str, dict[str, float]]
    frontal_specificity:     float = 1.0
    suggestibility:          str = "high"

    @field_validator("eeg_window")
    @classmethod
    def validate_shape(cls, v: list) -> list:
        if len(v) != N_CHANNELS:
            raise ValueError(
                f"eeg_window debe tener {N_CHANNELS} canales (filas), "
                f"recibido: {len(v)}"
            )
        return v


class TrainRequest(BaseModel):
    """
    Payload para POST /train.

    use_synthetic:
        True = genera datos sintéticos para demo/validación.
        En producción, implementar el endpoint con datos reales etiquetados.

    n_synthetic_samples:
        Número de muestras a generar (divididas en 3 clases).
        Mínimo recomendado: 600 (200/clase). Default: 1500.
    """
    use_synthetic:       bool = True
    n_synthetic_samples: int  = 1500


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints REST
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """
    Health check del servicio.

    Retorna:
      status        : "ok"
      channels      : lista de canales EEG esperados
      model_loaded  : True si hay un modelo entrenado disponible
      n_features    : número de features del clasificador
    """
    model_high = _get_model("high")
    model_low = _get_model("low")
    return {
        "status":       "ok",
        "channels":     CHANNELS,
        "model_loaded": (model_high is not None) or (model_low is not None),
        "models_loaded": {
            "high": model_high is not None,
            "low": model_low is not None,
        },
        "n_features":   15,
        "port":         8001,
    }


@app.post("/process_window")
async def api_process_window(req: ProcessWindowRequest):
    """
    Procesa una ventana EEG de 2 s y retorna coherencia, PLV y clasificación.

    Tiempo de respuesta esperado: ~15–35 ms en CPU moderna (11 canales).
    El cliente debería usar el endpoint WS para tiempo real y este
    endpoint REST solo para debug o integración puntual.
    """
    try:
        eeg_arr = np.array(req.eeg_window, dtype=np.float32)
        result  = process_window(
            eeg_window=eeg_arr,
            band_powers_per_channel=req.band_powers_per_channel,
            frontal_specificity=req.frontal_specificity,
            suggestibility=req.suggestibility,
        )
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/train")
async def api_train(req: TrainRequest):
    """
    Entrena el clasificador EEG y lo persiste en disco.

    Con use_synthetic=True: genera datos con distribuciones fisiológicamente
    realistas para validación de la arquitectura. Para producción, integrar
    un pipeline de recolección de datos EEG etiquetados.

    El modelo entrenado queda disponible inmediatamente para inferencia
    en el mismo proceso (sin necesidad de reiniciar el servidor).
    """
    try:
        if req.use_synthetic:
            X, y = generate_synthetic_training_data(
                n_samples=req.n_synthetic_samples
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Para datos reales, extender este endpoint para recibir "
                    "X (features) e y (etiquetas) como arrays en el body. "
                    "O usar el script train_classifier.py con --real data.npz"
                ),
            )

        model, metrics = train_classifier(X, y, save_model=True, verbose=True)

        # Invalidar caché para que la próxima inferencia use el nuevo modelo
        reload_model()

        return {
            "status":  "trained",
            "metrics": metrics,
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/reload_model")
async def api_reload_model():
    """
    Recarga el modelo entrenado desde disco sin reiniciar el servidor.
    Útil cuando se re-entrena el modelo externamente con datos nuevos.
    """
    success = reload_model()
    return {
        "status":       "reloaded" if success else "not_found",
        "model_loaded": success,
    }


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket — integración tiempo real con server.ts
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket para integración en tiempo real con el backend Node.js.

    Protocolo de mensajes:

    Cliente → Servidor:
      { "type": "process_window",
        "data": { "eeg_window": [...], "band_powers_per_channel": {...},
                  "frontal_specificity": 1.8 } }
      { "type": "ping" }

    Servidor → Cliente:
      { "type": "connectivity_result",
        "coherence_matrix": [...], "plv_matrix": [...],
        "classifier_prediction": {...}, ... }
      { "type": "pong" }
      { "type": "error", "message": "..." }

    Gestión de errores:
      Los errores de procesamiento se reportan como mensajes de tipo "error"
      sin cerrar la conexión, para que el servidor Node.js pueda seguir
      enviando ventanas en el siguiente hop.
    """
    await websocket.accept()
    print(f"[ws] Cliente conectado: {websocket.client}")

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type":    "error",
                    "message": "JSON inválido en el mensaje recibido",
                })
                continue

            msg_type = msg.get("type", "")

            # ── process_window: path caliente (~4 veces/segundo) ──────────
            if msg_type == "process_window":
                data = msg.get("data", {})
                try:
                    eeg_arr = np.array(data["eeg_window"], dtype=np.float32)
                    result  = process_window(
                        eeg_window=eeg_arr,
                        band_powers_per_channel=data.get("band_powers_per_channel", {}),
                        frontal_specificity=float(data.get("frontal_specificity", 1.0)),
                        suggestibility=str(data.get("suggestibility", "high")),
                    )
                    await websocket.send_json({
                        "type": "connectivity_result",
                        **result,
                    })
                except Exception as exc:
                    await websocket.send_json({
                        "type":    "error",
                        "message": f"process_window falló: {exc}",
                    })

            # ── ping / pong ────────────────────────────────────────────────
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})

            # ── reload_model sin HTTP ──────────────────────────────────────
            elif msg_type == "reload_model":
                success = reload_model()
                await websocket.send_json({
                    "type":         "model_reloaded",
                    "model_loaded": success,
                })

            else:
                await websocket.send_json({
                    "type":    "error",
                    "message": f"Tipo de mensaje no reconocido: '{msg_type}'",
                })

    except WebSocketDisconnect:
        print(f"[ws] Cliente desconectado: {websocket.client}")


# ─────────────────────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  EEG Connectivity & Classification Service")
    print("  Puerto: 8001  |  WebSocket: ws://localhost:8001/ws")
    print("  Docs:   http://localhost:8001/docs")
    print("=" * 60)
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=True)
