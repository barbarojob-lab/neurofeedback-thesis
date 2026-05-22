# Justificacion de Entrenamiento y Validacion

Fecha: 2026-05-01
Proyecto: neurofeedback-thesis

## 1. Objetivo metodologico

El objetivo del entrenamiento es ajustar los parametros del clasificador para aprender patrones EEG asociados a estados hipnoticos. El objetivo de la validacion es estimar que tan bien generaliza el modelo en sujetos no vistos durante el entrenamiento.

Por esta razon, entrenar y validar en los mismos datos no es valido cientificamente: inflaria el rendimiento por sobreajuste.

## 2. Como se hizo la separacion (train/test)

Se utilizo una separacion por sujeto completo (subject-level split) en el dataset `data/sessions_icapruned.npz`.

- Train: sujetos etiquetados como `split=train`
- Test: sujetos etiquetados como `split=test`

Este esquema evita fuga de informacion entre entrenamiento y validacion.

## 3. Por que se entrena y se valida

- Entrenar: para que el modelo aprenda patrones discriminativos de las clases.
- Validar: para medir rendimiento real fuera de la muestra de entrenamiento.
- Sin validacion, no hay evidencia de generalizacion y el resultado no es defendible para tesis.

## 4. Ejecuciones registradas (logs)

Se generaron logs reproducibles en:

- `logs/train_high.log`
- `logs/train_low.log`

Estos archivos registran comandos, distribucion de clases, CV, test hold-out, metricas y matriz de confusion.

## 5. Resultado resumido de validacion

### Modelo High (suggestibility_model=high)

- Muestras: 4196
- Sujetos: train=8, test=7
- CV accuracy: 0.529 +- 0.105
- Test accuracy: 0.432
- F1 macro: 0.263

### Modelo Low (suggestibility_model=low)

- Muestras: 3004
- Sujetos: train=8, test=7
- CV accuracy: 0.553 +- 0.076
- Test accuracy: 0.579
- F1 macro: 0.507

## 6. Evidencia de artefactos entrenados

Modelos generados/actualizados:

- `data/models_colab/eeg_classifier_high.joblib`
- `data/models_colab/eeg_classifier_low.joblib`

## 7. Conclusion

La validacion con mitad para entrenamiento y mitad para prueba por sujeto ya esta implementada y ejecutada. Los logs sirven como evidencia tecnica y metodologica para reproducibilidad y defensa academica.

## 8. Trazabilidad del remapeo de canales para EDF de prueba

Para la prueba en tiempo real con backend (modo hardware con dataset), el EDF debe exponer 13 canales:

- Fz, Fp1, Fp2, F3, F4, C3, Cz, C4, P3, Pz, P4, O1, O2

Los archivos ICApruned de origen usan nomenclatura alternativa (F7/F8, OZ, PO3, PO4). Para compatibilidad operacional se aplico remapeo de etiquetas en exportacion EDF sin sintetizar muestras.

Equivalencias usadas:

- F7 -> Fp1
- F8 -> Fp2
- PO3 -> O1
- PO4 -> O2
- FZ -> Fz
- PZ -> Pz
- CZ -> Cz

Importante:

- No se uso zero-padding ni interpolacion para crear canales faltantes.
- La señal exportada proviene de canales medidos en el registro original.
- Si un canal requerido no existe en el archivo fuente, la exportacion falla para evitar introducir sesgo analitico.