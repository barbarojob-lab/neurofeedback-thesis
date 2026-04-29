"""
=============================================================
DESCARGADOR — Dataset EEGMeditation (Hugging Face)
=============================================================
Descarga los 4 archivos necesarios por sujeto:
  - .eeg     (señal EEG)
  - .vhdr    (cabecera)
  - .vmrk    (marcadores)
  - events.tsv (etiquetas de tareas)

Uso:
    pip install huggingface_hub
    python descargar_dataset.py

Cambia SUJETOS_A_DESCARGAR para elegir cuántos bajar.
=============================================================
"""

from huggingface_hub import hf_hub_download
import os

# ─── CONFIGURACIÓN — cambia esto según necesites ─────────
SUJETOS_A_DESCARGAR = 3       # cambia a 10 o 20 si quieres más
CARPETA_DESTINO     = "./data/yoga_eeg"  # donde se guardan
REPO_ID             = "alexeykashevnik/EEGMeditation"
# ─────────────────────────────────────────────────────────

OK   = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
INFO = "\033[94mℹ\033[0m"

os.makedirs(CARPETA_DESTINO, exist_ok=True)

print("\n" + "="*55)
print("  DESCARGADOR — EEGMeditation Dataset")
print("="*55)
print(f"\n{INFO} Sujetos a descargar : sub-001 → sub-{SUJETOS_A_DESCARGAR:03d}")
print(f"{INFO} Peso estimado       : ~{SUJETOS_A_DESCARGAR * 1.34:.1f} GB")
print(f"{INFO} Destino             : {CARPETA_DESTINO}")
print(f"\n{'─'*55}\n")

descargados = 0
errores     = 0

for i in range(1, SUJETOS_A_DESCARGAR + 1):
    sujeto  = f"sub-{i:03d}"
    prefijo = f"{sujeto}/ses-01/eeg/{sujeto}_ses-01_task-default_run-01"

    archivos = [
        f"{prefijo}_eeg.eeg",      # señal EEG — el más pesado
        f"{prefijo}_eeg.vhdr",     # cabecera — MNE lo necesita
        f"{prefijo}_eeg.vmrk",     # marcadores internos
        f"{prefijo}_events.tsv",   # etiquetas AWAKE/TRANCE
    ]

    print(f"📂 {sujeto} ({i}/{SUJETOS_A_DESCARGAR})")

    for archivo in archivos:
        nombre = archivo.split("/")[-1]
        try:
            hf_hub_download(
                repo_id   = REPO_ID,
                repo_type = "dataset",
                filename  = archivo,
                local_dir = CARPETA_DESTINO,
            )
            print(f"   {OK} {nombre}")
            descargados += 1

        except Exception as e:
            print(f"   {FAIL} {nombre} — Error: {e}")
            errores += 1

    print()

# ─── RESUMEN ─────────────────────────────────────────────
print("="*55)
print("  RESUMEN FINAL")
print("="*55)
print(f"\n  {OK} Archivos descargados : {descargados}")
if errores:
    print(f"  {FAIL} Errores              : {errores}")
else:
    print(f"  {OK} Sin errores")

print(f"\n  Archivos guardados en:")
print(f"  {CARPETA_DESTINO}/")
print(f"  ├── sub-001/ses-01/eeg/")
print(f"  │   ├── ...eeg.eeg")
print(f"  │   ├── ...eeg.vhdr")
print(f"  │   ├── ...eeg.vmrk")
print(f"  │   └── ...events.tsv")
print(f"  ├── sub-002/ses-01/eeg/")
print(f"  └── ...")

if errores == 0:
    print(f"\n  ✅ Listo — ya puedes correr el pipeline EEG\n")
else:
    print(f"\n  ⚠️  Algunos archivos fallaron — vuelve a ejecutar el script\n")