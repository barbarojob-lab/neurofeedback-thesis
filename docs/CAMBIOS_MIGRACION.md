# Registro de Cambios: Entrenamiento Local → Google Colab

**Fecha:** 2026-05-01  
**Descripción:** Se eliminó todo el pipeline de entrenamiento local para migrar a Google Colab.

---

## Archivos ELIMINADOs

✂️ **Entrenamiento:**
- `ml-service/train_classifier.py` - Script CLI para entrenar modelos
- `ml-service/prepare_dataset.py` - Preparación de datos para entrenamiento
- `ml-service/prepare_dataset_eeglab.py` - Conversión desde EEGLAB formato
- `ml-service/export_and_validate.py` - Validación cruzada y exportación

**Razón:** Estos scripts ejecutaban entrenamientos locales. Ahora todo se hace en Colab.

---

## Archivos MODIFICADOS

📝 **ml-service/server.py**
- ✂️ Eliminado import: `generate_synthetic_training_data, train_classifier`
- ✂️ Eliminada clase Pydantic: `TrainRequest`
- ✂️ Eliminado endpoint: `POST /train`
- ✏️ Actualizado docstring: nota que entrenamiento es en Colab
- ✅ Mantiene: endpoints `/health`, `/process_window`, `/reload_model`, `/ws`

**Razón:** El servidor ahora **solo carga modelos para inferencia**, no entrena.

---

## Archivos CREADOS

✨ **Documentación:**
- `docs/COLAB_TRAINING_GUIDE.md` - Guía paso a paso para entrenar en Colab
- `docs/CAMBIOS_MIGRACION.md` - Este archivo

---

## Archivos SIN CAMBIOS (todavía útiles)

🟢 **ml-service/convert_to_edf.py**
- Convierte archivos EEGLAB a EDF para testing
- Sigue siendo válido para cargar datasets de prueba

🟢 **backend/sim-test.js** / **backend/load-dataset.js**
- Test del simulador y carga de datasets
- No están afectados

🟢 **data/models_colab/** 
- Carpeta donde colocar modelos descargados de Colab
- `eeg_classifier_high.joblib`, `eeg_classifier_low.joblib`

---

## Flujo de trabajo ANTES vs DESPUÉS

### ❌ ANTES (Entrenamiento Local)
```
PC local
  ├─ python train_classifier.py
  ├─ python prepare_dataset.py
  ├─ python export_and_validate.py
  └─ POST /train (llamada HTTP al servidor)
       ↓
  Entrena localmente (lento sin GPU)
  ↓
  Modelos en data/models_colab/
```

### ✅ DESPUÉS (Entrenamiento en Colab)
```
Google Colab (GPU gratuita)
  ├─ Celda 1: Montar Drive
  ├─ Celda 2: Instalar dependencias
  ├─ ...
  ├─ Celda 9: Guardar a Drive
  └─ Descargar .joblib a PC local
       ↓
  Coloca en data/models_colab/
  ↓
  python ml-service/server.py (solo carga, no entrena)
  ↓
  Backend Node.js → llamadas WebSocket para INFERENCIA
  ↓
  Frontend recibe predicciones en tiempo real
```

---

## Verificación: Confirmar que funciona

1. **Descargar modelo** desde Colab y colocar en `data/models_colab/`

2. **Verificar que server.py los carga:**
   ```bash
   curl http://localhost:8001/health
   ```
   Debe devolver:
   ```json
   {
     "status": "ok",
     "models_loaded": {
       "high": true,
       "low": true
     }
   }
   ```

3. **Lanzar backend Node.js + frontend React** como de costumbre
   - No hay cambios en backend/ts ni frontend/

---

## Ventajas de esta migración

| Ventaja | Razón |
|---------|-------|
| **GPU gratis** | Colab incluye Tesla T4 / P100 |
| **Más rápido** | Entrenamientos 10-50x más rápidos |
| **Reproducible** | Mismo ambiente cada vez |
| **Colaborativo** | Fácil compartir notebook con asesor |
| **Entorno limpio** | PC local solo para desarrollo/demostración |
| **Escalable** | Puedes entrenar con más datos sin problemas |

---

## Próximos pasos

1. ✅ Lee `docs/COLAB_TRAINING_GUIDE.md`
2. ✅ Sube datos a Google Drive
3. ✅ Copia celdas en Colab
4. ✅ Ejecuta entrenamiento en Colab
5. ✅ Descarga modelo
6. ✅ Coloca en `data/models_colab/`
7. ✅ Prueba servidor local

---

## Contacto / Dudas

Si hay problemas:
- Revisa el archivo `COLAB_TRAINING_GUIDE.md` → sección "Troubleshooting"
- Verifica que el modelo esté en la carpeta correcta
- Ejecuta `GET /health` para confirmar carga
