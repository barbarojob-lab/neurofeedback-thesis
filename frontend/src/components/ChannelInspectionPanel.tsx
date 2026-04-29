import { memo, useMemo } from "react";
import {
  useEEGStore,
  selectInspectionFilteredSamples,
  selectInspectionFftMagnitudes,
  selectInspectionChannel,
} from "../store/eegStore";

function buildPolyline(samples: number[], width: number, height: number): string {
  if (samples.length === 0) return "";
  let peak = 1;
  for (let i = 0; i < samples.length; i++) peak = Math.max(peak, Math.abs(samples[i] ?? 0));

  return samples
    .map((v, i) => {
      const x = (i / Math.max(1, samples.length - 1)) * width;
      const y = height / 2 - (v / peak) * (height * 0.42);
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    })
    .join(" ");
}

const ChannelInspectionPanel = memo(function ChannelInspectionPanel() {
  const channel = useEEGStore(selectInspectionChannel);
  const samples = useEEGStore(selectInspectionFilteredSamples);
  const fft = useEEGStore(selectInspectionFftMagnitudes);

  const polyline = useMemo(() => buildPolyline(samples, 320, 90), [samples]);
  const fftBars = useMemo(() => {
    if (fft.length === 0) return [] as number[];
    const bins = fft.slice(0, 48);
    const max = Math.max(...bins, 1e-6);
    return bins.map((v) => Math.max(0, Math.min(1, v / max)));
  }, [fft]);

  return (
    <div style={{
      backgroundColor: "#0d0d1a",
      borderRadius: 8,
      border: "1px solid rgba(255,255,255,0.07)",
      padding: "12px 14px",
      display: "grid",
      gridTemplateColumns: "1fr 1fr",
      gap: 12,
    }}>
      <div>
        <div style={{
          fontSize: 9,
          color: "rgba(255,255,255,0.35)",
          letterSpacing: "0.12em",
          marginBottom: 8,
        }}>
          INSPECCION DE CANAL · {channel}
        </div>
        <svg viewBox="0 0 320 90" width="100%" height="110" role="img" aria-label={`Senal de ${channel}`}>
          <rect x="0" y="0" width="320" height="90" fill="rgba(255,255,255,0.02)" stroke="rgba(255,255,255,0.08)" />
          <line x1="0" y1="45" x2="320" y2="45" stroke="rgba(255,255,255,0.08)" strokeDasharray="4 4" />
          <polyline fill="none" stroke="#00e5ff" strokeWidth="1.6" points={polyline} />
        </svg>
      </div>

      <div>
        <div style={{
          fontSize: 9,
          color: "rgba(255,255,255,0.35)",
          letterSpacing: "0.12em",
          marginBottom: 8,
        }}>
          FFT ({channel})
        </div>
        <div style={{
          height: 110,
          display: "flex",
          alignItems: "flex-end",
          gap: 2,
          padding: "6px 8px",
          background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.08)",
        }}>
          {fftBars.map((v, idx) => (
            <div
              key={idx}
              style={{
                width: 4,
                height: `${Math.max(4, v * 96)}px`,
                background: "linear-gradient(180deg, #ffd600 0%, #00e5ff 100%)",
                opacity: 0.9,
              }}
            />
          ))}
        </div>
      </div>
    </div>
  );
});

export default ChannelInspectionPanel;
