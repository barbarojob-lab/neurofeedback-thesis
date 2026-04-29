"""
connectivity.py  —  ml-service/src/

Módulo de análisis de conectividad EEG en tiempo real.

Implementa dos medidas complementarias de sincronización cortical:
  1. Coherencia espectral (método Welch) en banda theta  (4–8 Hz)
  2. Phase Locking Value (PLV) en banda theta           (4–8 Hz)

Rationale neurocientífico:
  Durante estados de trance/hipnosis se observa un incremento robusto de
  coherencia theta de larga distancia, en particular en el eje Fz–Pz.
  Este patrón se interpreta como sincronización funcional entre la corteza
  prefrontal (control ejecutivo) y la corteza parietal (integración
  sensorial), reflejando el aumento de sugestibilidad y la atención
  sostenida interna propias del trance (Gruzelier et al., 2006;
  Sabourin et al., 1990).

  El PLV es más robusto que la coherencia ante diferencias de amplitud
  entre electrodos: mide acoplamiento de fase pura, por lo que no es
  susceptible a variaciones de impedancia o distancia al dipolo cortical
  (Lachaux et al., 1999; Mormann et al., 2000).

Referencias:
  - Gruzelier, J. H. (2006). Theta EEG changes following hypnotic induction.
    International Journal of Clinical and Experimental Hypnosis.
  - Lachaux, J. P., et al. (1999). Measuring phase synchrony in brain signals.
    Human Brain Mapping.
  - Sabourin, M. E., et al. (1990). EEG correlates of hypnotic susceptibility.
"""

import numpy as np
from scipy.signal import coherence, butter, filtfilt, hilbert
from typing import Dict, List, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Configuración del sistema
# ─────────────────────────────────────────────────────────────────────────────

# Orden de canales que DEBE coincidir con el backend TypeScript.
# Este orden es el que se usa para indexar las matrices NxN.
CHANNELS: List[str] = [
    "Fz", "Fp1", "F3", "C3", "Pz", "O1", "F4", "C4", "P4", "O2", "Cz",
]
N_CHANNELS = len(CHANNELS)
CHANNEL_INDEX: Dict[str, int] = {ch: i for i, ch in enumerate(CHANNELS)}

# Parámetros del pipeline de señal (deben coincidir con server.ts)
FS          = 250    # Hz — OpenBCI Cyton default
WINDOW_SIZE = 500    # samples — 2 s a 250 Hz
HOP_SIZE    = 64     # samples — 4 payloads/s, overlap 87.5 %

# Banda theta: 4–8 Hz
# Rationale: el ritmo theta es la banda más estudiada en hipnosis y
# meditación. El rango 4–8 Hz cubre el theta frontal (4–6 Hz, asociado
# a inducción) y el theta frontoparietal (6–8 Hz, asociado a trance
# profundo y memoria episódica — Jensen & Tesche, 2002).
THETA_LOW  = 4.0   # Hz
THETA_HIGH = 8.0   # Hz


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades internas
# ─────────────────────────────────────────────────────────────────────────────

def _theta_bandpass(signal: np.ndarray, fs: int = FS) -> np.ndarray:
    """
    Filtra una señal 1D en banda theta (4–8 Hz) con Butterworth de orden 4.

    Elección de orden 4:
      - Orden 2: pendiente de −40 dB/dec → deja pasar demasiado beta
      - Orden 6+: mayor ringing y posible inestabilidad numérica con filtfilt
      - Orden 4: −80 dB/dec, estable, buena selectividad para 4–8 Hz

    filtfilt (fase cero):
      Aplica el filtro en dirección forward y backward, cancelando el
      retardo de grupo. CRÍTICO para el PLV: cualquier retardo de fase
      residual sesgará la estimación de la diferencia de fase instantánea.

    Args:
        signal : array 1D de muestras EEG [µV], shape (n_samples,)
        fs     : frecuencia de muestreo [Hz]

    Returns:
        Señal filtrada en theta, misma shape que la entrada.
    """
    nyq  = fs / 2.0
    low  = THETA_LOW  / nyq
    high = THETA_HIGH / nyq
    b, a = butter(4, [low, high], btype="bandpass")
    return filtfilt(b, a, signal)


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTE 1 — Coherencia espectral (Welch)
# ─────────────────────────────────────────────────────────────────────────────

def compute_coherence_matrix(
    eeg_window: np.ndarray,
    fs: int = FS,
    nperseg: int = WINDOW_SIZE,
) -> np.ndarray:
    """
    Calcula la matriz de coherencia espectral en banda theta para todos
    los pares de canales usando el método de Welch.

    Fórmula de coherencia:
        C_xy(f) = |P_xy(f)|² / (P_xx(f) · P_yy(f))

      Donde:
        P_xy = cross power spectral density (Welch)
        P_xx, P_yy = auto power spectral density de cada canal

      C_xy ∈ [0, 1]: 0 = independencia estadística, 1 = coherencia perfecta.

    Parámetros Welch:
        nperseg = 500 (2 s a 250 Hz):
          Resolución espectral Δf = fs/nperseg = 250/500 = 0.5 Hz.
          Necesaria para resolver la banda 4–8 Hz (que tiene solo 4 Hz de
          ancho) con al menos 8 bins de frecuencia. Menor nperseg → Δf
          mayor y pérdida de resolución dentro de la banda theta.

        noverlap = nperseg//2 (50 % overlap):
          Balance entre estabilidad estadística y resolución temporal.
          Con window=2 s y overlap=50 %, cada segmento aporta información
          semi-independiente al promedio Welch.

    Resultado: se promedia la coherencia sobre todos los bins de frecuencia
    dentro de [4, 8] Hz para obtener un escalar por par de canales.

    Rationale clínico:
        Coherencia Fz–Pz > 0.5 diferencia trance de vigilia con p<0.01 en
        múltiples estudios (Sabourin et al., 1990; Gruzelier, 2006).
        Coherencia F3–F4 refleja balance interhemisférico frontal.
        Coherencia C3–C4 mide sincronización de la corteza motora
        (inhibición motora en hipnosis: Barker & Burgess, 2013).

    Args:
        eeg_window : shape (N_CHANNELS, n_samples) — ventana preprocesada
        fs         : frecuencia de muestreo [Hz]
        nperseg    : longitud del segmento Welch en muestras

    Returns:
        coh_matrix : shape (N_CHANNELS, N_CHANNELS), simétrica, valores en [0,1]
                     diagonal = 1.0 (auto-coherencia perfecta por definición)
    """
    n_ch, n_samp = eeg_window.shape
    coh_matrix   = np.zeros((n_ch, n_ch), dtype=np.float32)

    # nperseg efectivo: no puede superar el tamaño real de la ventana
    effective_nperseg = min(nperseg, n_samp)

    for i in range(n_ch):
        coh_matrix[i, i] = 1.0  # auto-coherencia = 1 por definición matemática
        for j in range(i + 1, n_ch):
            freqs, cxy = coherence(
                eeg_window[i],
                eeg_window[j],
                fs=fs,
                nperseg=effective_nperseg,
                noverlap=effective_nperseg // 2,
            )

            # Máscara de frecuencias en la banda theta
            theta_mask = (freqs >= THETA_LOW) & (freqs <= THETA_HIGH)

            if theta_mask.sum() == 0:
                # Caso degenereado: resolución espectral tan baja que no hay
                # ningún bin en 4–8 Hz → reportar 0 (sin información)
                coh_val = 0.0
            else:
                # Promedio de coherencia en la banda theta.
                # Se usa media aritmética en lugar del pico para mayor
                # robustez: el pico es sensible al ruido en un único bin.
                coh_val = float(np.mean(cxy[theta_mask]))

            # La coherencia es simétrica: C_xy = C_yx
            coh_matrix[i, j] = coh_val
            coh_matrix[j, i] = coh_val

    return coh_matrix


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENTE 2 — Phase Locking Value (PLV)
# ─────────────────────────────────────────────────────────────────────────────

def compute_plv_matrix(
    eeg_window: np.ndarray,
    fs: int = FS,
) -> np.ndarray:
    """
    Calcula la matriz de Phase Locking Value (PLV) en banda theta para
    todos los pares de canales.

    Algoritmo (Lachaux et al., 1999):
      1. Filtrar en theta (4–8 Hz) con Butterworth orden 4, fase cero.
      2. Transformada de Hilbert → señal analítica z(t) = x(t) + j·H{x(t)}
         Donde H{·} es la transformada de Hilbert (convierte cosenos en
         senos, rotando la fase +90°).
      3. Fase instantánea: φ(t) = arg(z(t)) ∈ [−π, π]
      4. Para cada par (i, j):
            Δφ(t) = φᵢ(t) − φⱼ(t)
            PLV   = |E[exp(j·Δφ(t))]| = |media temporal del vector unitario|

    Interpretación geométrica:
      exp(j·Δφ(t)) es un vector unitario en el plano complejo. Si la
      diferencia de fase es CONSTANTE en el tiempo, todos los vectores
      apuntan en la misma dirección → su suma normalizada tiene módulo 1.
      Si Δφ es aleatoria, los vectores se cancelan → módulo ≈ 0.

    Ventaja sobre coherencia:
      PLV solo mide consistencia de fase; no penaliza ni premia las
      diferencias de amplitud. Un electrodo con impedancia alta puede
      tener baja amplitud pero PLV alto si está sincronizado en fase.
      Esto lo hace más apropiado para comparar canales frontales (Fz,
      con amplitud típica 20–40 µV) con canales occipitales (O1/O2,
      5–15 µV) sin sesgo de amplitud.

    Valores de referencia clínico:
      PLV < 0.2  : no sincronizados (fases aleatorias)
      PLV 0.2–0.4: acoplamiento débil (vigilia alerta)
      PLV 0.4–0.6: acoplamiento moderado (relajación, inducción)
      PLV > 0.6  : sincronización fuerte (trance profundo, meditación)
      (Terhune et al., 2011; Jensen & Tesche, 2002)

    Args:
        eeg_window : shape (N_CHANNELS, n_samples) — ventana preprocesada
        fs         : frecuencia de muestreo [Hz]

    Returns:
        plv_matrix : shape (N_CHANNELS, N_CHANNELS), simétrica, valores en [0,1]
                     diagonal = 1.0 (auto-PLV = sincronización perfecta)
    """
    n_ch = eeg_window.shape[0]

    # ── Paso 1: Filtrar cada canal en banda theta ──────────────────────────
    # Se filtra antes de la Hilbert para evitar que la fase instantánea
    # esté contaminada por componentes de otras bandas (alpha, beta).
    # Sin este paso, la "fase instantánea" de la señal broadband EEG
    # no tiene interpretación física coherente (Pikovsky et al., 2001).
    theta_signals = np.zeros_like(eeg_window, dtype=np.float64)
    for ch in range(n_ch):
        theta_signals[ch] = _theta_bandpass(eeg_window[ch].astype(np.float64), fs)

    # ── Paso 2: Fase instantánea vía Transformada de Hilbert ──────────────
    # scipy.signal.hilbert() retorna la señal analítica compleja.
    # np.angle() extrae la fase en radianes ∈ [−π, π].
    phases = np.zeros_like(theta_signals, dtype=np.float64)
    for ch in range(n_ch):
        analytic_signal = hilbert(theta_signals[ch])
        phases[ch]      = np.angle(analytic_signal)

    # ── Paso 3: PLV para cada par ──────────────────────────────────────────
    plv_matrix = np.zeros((n_ch, n_ch), dtype=np.float32)

    for i in range(n_ch):
        plv_matrix[i, i] = 1.0  # auto-PLV = 1 por definición
        for j in range(i + 1, n_ch):
            # Diferencia de fase instantánea sample a sample
            delta_phase = phases[i] - phases[j]

            # PLV = módulo del vector medio en el plano complejo unitario.
            # La media temporal de exp(j·Δφ) es equivalente al estimador
            # de la magnitud del vector de resultante de Rayleigh, usado en
            # estadística circular para medir concentración de distribuciones
            # angulares (Fisher, 1993).
            plv_val = float(np.abs(np.mean(np.exp(1j * delta_phase))))

            plv_matrix[i, j] = plv_val
            plv_matrix[j, i] = plv_val  # PLV es simétrico

    return plv_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Extracción de features de conectividad clínicamente relevantes
# ─────────────────────────────────────────────────────────────────────────────

def get_connectivity_features(
    coh_matrix: np.ndarray,
    plv_matrix: np.ndarray,
) -> Dict[str, float]:
    """
    Extrae un diccionario de features escalares de las matrices NxN para
    envío por WebSocket y uso como input del clasificador.

    Pares seleccionados con justificación neurocientífica:
      Fz–Pz : Eje frontoparietal sagital medio.
               Marcador más replicado de profundidad hipnótica.
               Refleja sincronización entre DLPFC y corteza parietal posterior
               durante foco atencional interno (Gruzelier, 2006).

      F3–F4 : Balance interhemisférico frontal (izquierdo vs derecho).
               En relajación y trance: asimetría alfay theta decrece →
               PLV F3–F4 AUMENTA reflejando coherencia bilateral.
               En hiperactivación: PLV bajo (procesamiento asimétrico).

      C3–C4 : Sincronización de corteza motora bilateral.
               En trance profundo: inhibición motora → PLV tiende a bajar
               (las cortezas motoras se desacoplan del control voluntario).
               Útil como feature diferencial trance-inducción.

      Fz–Cz : Eje frontocentral.
               Frontal midline theta (Fm-theta): aparece en tareas de memoria
               de trabajo y en meditación focal. Distingue inducción activa
               (alta Fm-theta) de trance pasivo (Jensen & Tesche, 2002).

      O1–O2 : Coherencia occipital interhemisférica.
               En trance visual o hipnosis con imaginería: coherencia occipital
               aumenta con la viveza de las imágenes mentales. En trance sin
               imaginería: puede decrecer por supresión del procesamiento visual.

    Returns:
        dict con claves coh_{par} y plv_{par} para cada par definido.
        Todos los valores son float en [0, 1].
    """
    PAIRS = {
        "Fz_Pz": ("Fz",  "Pz"),
        "F3_F4": ("F3",  "F4"),
        "C3_C4": ("C3",  "C4"),
        "Fz_Cz": ("Fz",  "Cz"),
        "O1_O2": ("O1",  "O2"),
    }

    features: Dict[str, float] = {}
    for name, (ch_a, ch_b) in PAIRS.items():
        i = CHANNEL_INDEX[ch_a]
        j = CHANNEL_INDEX[ch_b]
        features[f"coh_{name}"] = float(coh_matrix[i, j])
        features[f"plv_{name}"] = float(plv_matrix[i, j])

    return features
