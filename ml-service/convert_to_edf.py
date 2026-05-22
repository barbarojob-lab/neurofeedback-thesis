"""
convert_to_edf.py  —  ml-service/

Convierte archivos EEGLAB .set/.fdt del grupo de VALIDACION (split=test)
al formato EDF estándar que lee el DatasetReplayer del backend Node.js.

Sujetos seleccionados:
  HIGH sugestionabilidad: subject_11  (awake/induction/trance completo)
  LOW  sugestionabilidad: subject_19  (awake/induction)

Salida:
  data/subject_11_high_test.edf
  data/subject_19_low_test.edf

Uso:
  python ml-service/convert_to_edf.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml-service"))

import mne

CHANNELS_ORDERED = ["Fz", "Fp1", "Fp2", "F3", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4", "O1", "O2"]

CHANNEL_ALIASES = {
    "Fz":  ["Fz", "FZ"],
    "Fp1": ["Fp1", "FP1", "Fpz", "FPZ", "F7"],
    "Fp2": ["Fp2", "FP2", "F8"],
    "F3":  ["F3"],
    "F4":  ["F4"],
    "C3":  ["C3"],
    "Cz":  ["Cz", "CZ"],
    "C4":  ["C4"],
    "P3":  ["P3"],
    "Pz":  ["Pz", "PZ"],
    "P4":  ["P4"],
    "O1":  ["O1", "PO3", "Oz", "OZ"],
    "O2":  ["O2", "PO4", "Oz", "OZ"],
}

SUBJECTS = [
    {
        "set_path": os.path.join(ROOT, "data", "ICApruned_highs", "subject_11.set"),
        "edf_path": os.path.join(ROOT, "data", "subject_11_high_test.edf"),
        "group":    "HIGH",
        "label":    "subject_11 — alta sugestionabilidad (validación)",
    },
    {
        "set_path": os.path.join(ROOT, "data", "ICApruned_lows", "subject_19.set"),
        "edf_path": os.path.join(ROOT, "data", "subject_19_low_test.edf"),
        "group":    "LOW",
        "label":    "subject_19 — baja sugestionabilidad (validación)",
    },
]


def resolve_channel_mapping(ch_names):
    """Mapear CHANNELS_ORDERED a canales disponibles sin duplicados."""
    available = {ch.lower(): ch for ch in ch_names}
    used = set()
    mapping = {}

    for target in CHANNELS_ORDERED:
        found = None
        # Intentar alias en orden
        for cand in CHANNEL_ALIASES.get(target, [target]):
            src = available.get(cand.lower())
            if src is None or src in used:
                continue
            found = src
            break

        if found is None:
            # print(f"  [WARN] Canal '{target}' no encontrado o ya usado — se omite")
            pass
        else:
            mapping[target] = found
            used.add(found)

    return mapping


def convert(info: dict) -> None:
    set_path = info["set_path"]
    edf_path = info["edf_path"]
    group    = info["group"]
    label    = info["label"]

    print(f"\n[{group}] {label}")
    print(f"  Fuente : {set_path}")

    if not os.path.isfile(set_path):
        print(f"  [ERROR] Archivo no encontrado: {set_path}")
        return

    if os.path.isfile(edf_path):
        print(f"  [info]  Sobrescribiendo EDF existente: {edf_path}")

    print("  Cargando .set/.fdt ...")
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose=False)
    print(f"  Canales originales : {raw.ch_names}")
    fs_orig = int(raw.info["sfreq"])
    dur_orig = raw.times[-1]
    print(f"  Fs={fs_orig} Hz  Duración={dur_orig:.1f} s  ({int(dur_orig*fs_orig)} samples)")

    mapping = resolve_channel_mapping(raw.ch_names)
    if not mapping:
        print("  [ERROR] Ningún canal válido encontrado — abortando")
        return

    missing = [ch for ch in CHANNELS_ORDERED if ch not in mapping]
    if missing:
        print(f"  [ERROR] Faltan canales requeridos para backend: {missing}")
        return

    # Seleccionar canales en el orden de CHANNELS_ORDERED (que tiene duplicados resueltos)
    picks_by_name = [mapping[t] for t in CHANNELS_ORDERED if t in mapping]
    picks_idx = [raw.ch_names.index(ch) for ch in picks_by_name]
    
    print(f"  Seleccionando canales: {picks_by_name}")
    raw.pick(picks_idx)
    
    # Renombrar a nombres estándar
    rename_dict = {old: new for old, new in zip(picks_by_name, [t for t in CHANNELS_ORDERED if t in mapping])}
    raw.rename_channels(rename_dict)

    # Preprocesado idéntico al de entrenamiento
    if fs_orig != 250:
        print(f"  Resampleando {fs_orig} → 250 Hz ...")
        raw.resample(250, verbose=False)

    print("  Filtro notch 50 Hz + bandpass 1–40 Hz ...")
    raw.notch_filter(50.0, verbose=False)
    raw.filter(1.0, 40.0, verbose=False)

    os.makedirs(os.path.dirname(edf_path), exist_ok=True)
    print(f"  Exportando → {edf_path}")
    raw.export(edf_path, fmt="edf", overwrite=True, verbose=False)

    size_mb = os.path.getsize(edf_path) / (1024 * 1024)
    print(f"  OK — {raw.info['nchan']} canales, {raw.times[-1]:.1f} s, 250 Hz  [{size_mb:.1f} MB]")
    print(f"  Canales en EDF: {raw.ch_names}")


def main():
    print("=" * 60)
    print("  CONVERSIÓN .SET/.FDT → EDF (sujetos de validación)")
    print("=" * 60)

    for info in SUBJECTS:
        convert(info)

    print("\n" + "=" * 60)
    print("  LISTO — EDFs disponibles en data/")
    print("=" * 60)
    print()
    print("  Para cargar en la interfaz, envía via WebSocket:")
    for info in SUBJECTS:
        edf = info["edf_path"].replace("\\", "/")
        print(f'    {{ "type": "load_dataset", "payload": {{ "path": "{edf}" }} }}')


if __name__ == "__main__":
    main()
