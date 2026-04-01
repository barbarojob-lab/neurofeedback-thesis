# Plan Reparación Dashboard

## Objetivo
Dashboard visible SIEMPRE (con o sin backend). Crash React solucionado.

## Cambios
1. App.tsx: ErrorBoundary + lazy WS
2. eegStore.ts: throttle 30Hz
3. useEEGSocket.ts: debounce 500ms

## Verificación
```
http://localhost:5173 → layout completo
WS rojo OK, controles disabled
Backend up → WS verde + EEG live
```

**Estado:** Listo para edits
