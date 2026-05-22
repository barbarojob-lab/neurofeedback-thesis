# Guía: Entrenamiento en Google Colab

**Fecha:** 2026-05-01  
**Estado del proyecto:** Entrenamiento migrado a Google Colab (entrenamientos locales eliminados)

---

## ¿Por qué Google Colab?

1. **GPU gratuita**: Entrenamientos más rápidos sin invertir en hardware.
2. **Entorno replicable**: Mismas librerías, mismas versiones.
3. **Integración con Drive**: Guardar modelos en la nube automáticamente.
4. **Aislado**: No afecta la máquina local ni el servidor en ejecución.
5. **Compartible**: Fácil compartir con asesor/equipo.

---

## Paso 1: Preparación en Drive

1. **Sube tus datos a Google Drive:**
   - Crear carpeta: `/Mi unidad/neurofeedback-thesis/`
   - Subir: `data/ICApruned_highs/`, `data/ICApruned_lows/`, `data/sessions_icapruned.npz`

2. **Opcional:** Usa `data/models_colab/` para guardar modelos entrenados.

---

## Paso 2: Crear Notebook en Colab

1. Ir a https://colab.research.google.com
2. Crear notebook nuevo
3. Renombrarlo: `EEG_Classifier_Training`

---

## Paso 3: Celdas de Colab (copia-pega)

### Celda 1: Montar Drive
```python
from google.colab import drive
drive.mount('/content/drive')
print("✅ Drive montado en /content/drive")
```

### Celda 2: Instalar dependencias
```python
!pip install -q numpy scikit-learn joblib mne scipy matplotlib scipy seaborn
print("✅ Dependencias instaladas")
```

### Celda 3: Descargar código del proyecto
```python
import os
import sys
from pathlib import Path

# Rutas
DRIVE_ROOT = Path('/content/drive/MyDrive/neurofeedback-thesis')
COLAB_ROOT = Path('/content/neurofeedback-thesis')

# Clonar o descargar el repo if needed (o sincronizar carpeta ml-service)
# Para este ejemplo, asume que subiste ml-service/src/ a Drive

# Crear ruta local en Colab
os.makedirs(COLAB_ROOT / 'ml-service' / 'src', exist_ok=True)
os.makedirs(COLAB_ROOT / 'data', exist_ok=True)

# Copiar archivos desde Drive (ajusta según tu estructura)
!cp -r /content/drive/MyDrive/neurofeedback-thesis/ml-service/src/* {COLAB_ROOT}/ml-service/src/ 2>/dev/null || true
!cp -r /content/drive/MyDrive/neurofeedback-thesis/data/* {COLAB_ROOT}/data/ 2>/dev/null || true

sys.path.insert(0, str(COLAB_ROOT / 'ml-service'))

print(f"✅ Estructura preparada en {COLAB_ROOT}")
```

### Celda 4: Importar módulos
```python
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

print("✅ Imports completados")
```

### Celda 5: Cargar datos

```python
# Ajusta según tu estructura de datos
DATA_PATH = Path('/content/drive/MyDrive/neurofeedback-thesis/data')

# Si tienes archivo .npz directamente
X_file = DATA_PATH / 'sessions_icapruned.npz'

if X_file.exists():
    data = np.load(X_file)
    X = data['X'].astype(np.float32)  # features
    y = data['y'].astype(np.int32)    # etiquetas
    print(f"✅ Datos cargados: X shape={X.shape}, y shape={y.shape}")
    print(f"   Clases: {np.unique(y)} (conteos: {np.bincount(y)})")
else:
    print(f"⚠️  Archivo no encontrado en {X_file}")
    print(f"   Archivos disponibles en {DATA_PATH}:")
    for f in DATA_PATH.glob('*'):
        print(f"     - {f.name}")
```

### Celda 6: Definir modelo y entrenar

```python
# Hiperparámetros recomendados
N_EPOCHS = 50  # Prueba primero con 10, sube poco a poco
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_STATE = 42

# Crear modelo ensemble
svm = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
          class_weight="balanced", random_state=RANDOM_STATE)
rf = RandomForestClassifier(n_estimators=100, min_samples_split=5,
                            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
voting = VotingClassifier(estimators=[('svm', svm), ('rf', rf)], voting='soft')

# Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', voting),
])

# Train/valid split
n_train = int(len(X) * (1 - VALIDATION_SPLIT))
idx = np.random.permutation(len(X))
train_idx = idx[:n_train]
val_idx = idx[n_train:]

X_train, X_val = X[train_idx], X[val_idx]
y_train, y_val = y[train_idx], y[val_idx]

print(f"📊 Train: {X_train.shape[0]} samples")
print(f"📊 Valid: {X_val.shape[0]} samples")
print(f"   Train distribution: {np.bincount(y_train)}")
print(f"   Valid distribution: {np.bincount(y_val)}")

# Entrenar
print("\n🚀 Iniciando entrenamiento...")
pipeline.fit(X_train, y_train)
print("✅ Entrenamiento completado")

# Evaluación
train_acc = pipeline.score(X_train, y_train)
val_acc = pipeline.score(X_val, y_val)
y_pred_val = pipeline.predict(X_val)
val_f1 = f1_score(y_val, y_pred_val, average='macro')

print(f"\n📈 Resultados:")
print(f"   Train accuracy: {train_acc:.3f}")
print(f"   Valid accuracy: {val_acc:.3f}")
print(f"   Valid F1 (macro): {val_f1:.3f}")
```

### Celda 7: Validación cruzada (opcional pero recomendado)

```python
# Cross-validation para ver estabilidad del modelo
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='accuracy')

print(f"\n🎯 Cross-Validation (5 folds):")
print(f"   Scores por fold: {cv_scores}")
print(f"   Media: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

### Celda 8: Matriz de confusión

```python
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

y_pred = pipeline.predict(X_val)
cm = confusion_matrix(y_val, y_pred)
labels = ['awake', 'induction', 'trance']

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
ax.set_title('Matriz de Confusión (Validation Set)')
ax.set_ylabel('Real')
ax.set_xlabel('Predicho')
plt.tight_layout()
plt.show()

print(f"\n📋 Clasificación detallada:")
print(classification_report(y_val, y_pred, target_names=labels))
```

### Celda 9: Guardar modelo en Drive

```python
# Define nombres según suggestibility
SUGGESTIBILITY = "high"  # o "low"
MODEL_NAME = f"eeg_classifier_{SUGGESTIBILITY}.joblib"
SAVE_PATH = Path('/content/drive/MyDrive/neurofeedback-thesis/models') / MODEL_NAME

SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# Guardar
joblib.dump(pipeline, SAVE_PATH)
print(f"✅ Modelo guardado en: {SAVE_PATH}")

# Verificar
loaded = joblib.load(SAVE_PATH)
test_acc = loaded.score(X_val, y_val)
print(f"✅ Verificación: accuracy en modelo cargado = {test_acc:.3f}")
```

### Celda 10 (Opcional): Visualizar curvas de entrenamiento

```python
# Si usas Keras/TensorFlow, puedes agregar histórico
# Para sklearn, registra manualmente scores en cada época

# Ejemplo simple:
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

history = {'train': [], 'valid': []}

for epoch in range(1, 6):  # mini-demostración
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    clf = VotingClassifier(estimators=[
        ('svm', SVC(kernel="rbf", C=10.0, probability=True, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42))
    ], voting='soft')
    
    clf.fit(X_train_scaled, y_train)
    history['train'].append(clf.score(X_train_scaled, y_train))
    history['valid'].append(clf.score(X_val_scaled, y_val))
    print(f"Epoch {epoch}: train={history['train'][-1]:.3f}, valid={history['valid'][-1]:.3f}")

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(history['train'], label='Train', marker='o')
plt.plot(history['valid'], label='Validation', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Curva de Entrenamiento')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Paso 4: Descargar modelo a máquina local

1. En Colab, después de entrenar, el modelo está en Drive.
2. Descárgalo desde Drive a tu máquina.
3. Colócalo en: `data/models_colab/eeg_classifier_high.joblib` (o `_low.joblib`)

---

## Paso 5: Verificar que el servidor local lo carga

```bash
cd ml-service
python server.py
```

Espera a ver:
```
GET  http://localhost:8001/health
```

Debería mostrar:
```json
{
  "status": "ok",
  "model_loaded": true,
  "models_loaded": {
    "high": true,
    "low": true
  }
}
```

---

## Notas importantes

| Aspecto | Recomendación |
|--------|---------------|
| **Primeras épocas a probar** | 10-20 (para ver estabilidad) |
| **Incrementar si mejora** | Sí, de 10 en 10 hasta 100 si es necesario |
| **Si train loss baja pero val loss sube** | Overfitting → reducir epochs o usar regularización |
| **Batch size** | 32 (defecto), o 16/64 si quieres probar |
| **Learning rate** | SVM/RF lo manejan internamente, no es tuneable fácil |
| **Reproducibilidad** | Usar `random_state=42` en todas partes |

---

## Troubleshooting

**Problema:** "ModuleNotFoundError: No module named 'src.classifier'"  
**Solución:** Verifica que copiaste los archivos de `ml-service/src/` a Colab correctamente.

**Problema:** Accuracy muy baja (< 30%)  
**Solución:** Verifica que las etiquetas `y` sean correctas (0, 1, 2) y no tengan valores perdidos.

**Problema:** Colab se desconecta durante entrenamiento  
**Solución:** Usa `!tensorboard` o guarda checkpoints frecuentemente a Drive.

---

## Flujo completo resumido

```
1. Prepara datos en Drive
2. Abre Colab y copia celdas
3. Ejecuta from Celda 1 to 9
4. Descarga modelo desde Drive
5. Coloca en data/models_colab/
6. Inicia server.py local
7. Verifica con GET /health
8. ¡Listo! Usar app frontend con modelo de Colab
```

---

## Recursos

- [Google Colab Docs](https://colab.research.google.com)
- [scikit-learn Pipeline Docs](https://scikit-learn.org/stable/modules/compose.html#pipeline)
- [joblib para serializar modelos](https://joblib.readthedocs.io/)
