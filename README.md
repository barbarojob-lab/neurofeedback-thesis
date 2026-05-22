# tesis

Sistema de neurofeedback basado en EEG para clasificar estados `awake`, `induction` y `trance`.

Este repositorio incluye:
- entrenamiento de modelos ML (Python)
- servicio backend (Node/TypeScript)
- frontend de visualizacion (Vite + TS)
- generacion automatica de resultados para el capitulo de tesis

## 1. Estructura del proyecto

- `ml-service/`: entrenamiento y servicio ML
- `backend/`: servidor de integracion y procesamiento en tiempo real
- `frontend/`: interfaz web
- `data/`: datos y modelos (datasets grandes ignorados en Git)
- `logs/`: reportes de validacion por split
- `results_thesis/`: artefactos para el capitulo de resultados
- `scripts/`: utilidades (ej. generador de resultados de tesis)
- `docs/`: guias y notas tecnicas

## 2. Requisitos

- Python 3.11+ (recomendado usar `.venv`)
- Node.js 18+
- npm 9+

## 3. Configuracion rapida

### 3.1 Python (entrenamiento)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ml-service\requirements.txt
```

### 3.2 Backend

```powershell
cd backend
npm install
```

### 3.3 Frontend

```powershell
cd frontend
npm install
```

## 4. Entrenamiento de modelos

Desde la raiz del proyecto:

```powershell
.\.venv\Scripts\python.exe ml-service\train_local.py
```

Salida esperada:
- modelos en `data/models_colab/`
- reportes en `logs/train_high_subject_split.json`, `logs/train_low_subject_split.json`, `logs/train_unified_subject_split.json`

## 5. Generar resultados para tesis

```powershell
.\.venv\Scripts\python.exe scripts\generate_thesis_results.py
```

Se generan:
- `results_thesis/summary_metrics.csv`
- `results_thesis/class_metrics.csv`
- `results_thesis/thesis_table.tex`
- `results_thesis/interpretation_notes.txt`
- `results_thesis/manifest.json`

## 6. Ejecutar aplicacion (backend + frontend)

### 6.1 Backend

```powershell
cd backend
npm run dev
```

### 6.2 Frontend

```powershell
cd frontend
npm run dev
```

Luego abre la URL que muestra Vite (por defecto `http://localhost:5173`).

## 7. Flujo recomendado para nuevos colaboradores

1. Clonar repositorio.
2. Configurar entorno Python e instalar dependencias.
3. Instalar dependencias de `backend/` y `frontend/`.
4. Ejecutar entrenamiento local o usar modelos existentes.
5. Generar `results_thesis/` para reportes.

## 8. Publicar en GitHub (repositorio `tesis`)

```powershell
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<TU_USUARIO>/tesis.git
git push -u origin main
```

## 9. Notas importantes

- Los datasets EEG pesados y modelos entrenados estan ignorados por `.gitignore`.
- Los reportes clave de validacion por split en `logs/` si pueden versionarse.
- `results_thesis/` esta pensado para copiar/pegar resultados directamente en la tesis.
