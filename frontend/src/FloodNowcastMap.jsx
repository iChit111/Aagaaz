import { useState, useMemo, useEffect } from 'react';
import Map, { Layer, Source } from 'react-map-gl/mapbox';
import 'mapbox-gl/dist/mapbox-gl.css';
import puneRoadsData from './pune_roads.json';

const API_BASE_URL = 'http://127.0.0.1:8000';
const MAP_STYLE = 'mapbox://styles/mapbox/dark-v11';

// IMD real-time rainfall intensity brackets (mm/hr)
const RAINFALL_CATEGORIES = [
  { max: 2.5, label: 'No / Light Drizzle', color: '#38bdf8' },
  { max: 7.5, label: 'Light Rain', color: '#22c55e' },
  { max: 35.5, label: 'Moderate Rain', color: '#eab308' },
  { max: 64.5, label: 'Heavy Rain', color: '#f97316' },
  { max: 124.5, label: 'Very Heavy Rain', color: '#ef4444' },
  { max: Infinity, label: 'Extreme Rain', color: '#b91c1c' },
];

function getRainfallCategory(mmPerHr) {
  return RAINFALL_CATEGORIES.find((bucket) => mmPerHr <= bucket.max);
}

function getFloodStatus(depthCm) {
  if (depthCm >= 30) return 'impassable';
  if (depthCm >= 10) return 'pooling';
  return 'dry';
}

// The Data-Driven Paint Rules for Roads
const roadLayer = {
  id: 'road-floods',
  type: 'line',
  paint: {
    'line-width': 4,
    'line-color': [
      'case',
      // Condition 1: Depth >= 30cm -> RED (Impassable / Surcharging)
      ['>=', ['get', 'flood_depth'], 30], '#ef4444',
      // Condition 2: Depth >= 10cm -> YELLOW (Warning / Pooling)
      ['>=', ['get', 'flood_depth'], 10], '#eab308',
      // Default: Depth < 10cm -> GREEN (Safe / Dry)
      '#22c55e'
    ],
    'line-opacity': 0.9,
  },
};

export default function FloodNowcastMap() {
  // Nowcast time series: [{ elapsed_min, depths: { roadId: cm } }, ...]
  const [frames, setFrames] = useState([]);
  const [frameIndex, setFrameIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [rainfallIntensity, setRainfallIntensity] = useState(50);
  const [error, setError] = useState('');
  const [isSimulating, setIsSimulating] = useState(false);
  const [lastRun, setLastRun] = useState(null);

  const rainfallCategory = getRainfallCategory(rainfallIntensity);
  const currentFrame = frames[frameIndex];
  const currentDepths = currentFrame?.depths ?? {};

  const runSummary = useMemo(() => {
    const depths = Object.values(currentDepths);
    return {
      impassable: depths.filter((d) => getFloodStatus(d) === 'impassable').length,
      pooling: depths.filter((d) => getFloodStatus(d) === 'pooling').length,
      dry: depths.filter((d) => getFloodStatus(d) === 'dry').length,
    };
  }, [currentDepths]);

  // The Injection: Merge static roads with the currently scrubbed frame's depths
  const dynamicMapData = useMemo(() => {
    const updatedFeatures = puneRoadsData.features.map(feature => {
      // Overpass Turbo usually assigns OSM IDs as strings like "way/12345"
      const roadId = feature.id ?? feature.properties?.id;
      const currentDepth = roadId == null ? 0 : currentDepths[String(roadId)] ?? 0;

      return {
        ...feature,
        properties: {
          ...feature.properties,
          flood_depth: currentDepth
        }
      };
    });

    return { ...puneRoadsData, features: updatedFeatures };
  }, [currentDepths]);

  // Auto-advance the scrubber through the nowcast window while playing
  useEffect(() => {
    if (!isPlaying || frames.length === 0) return undefined;
    const timer = setInterval(() => {
      setFrameIndex((index) => {
        if (index >= frames.length - 1) {
          setIsPlaying(false);
          return index;
        }
        return index + 1;
      });
    }, 600);
    return () => clearInterval(timer);
  }, [isPlaying, frames.length]);

  async function simulateRainfall() {
    setIsSimulating(true);
    setIsPlaying(false);
    setError('');

    try {
      const response = await fetch(`${API_BASE_URL}/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rainfall_mm_per_hr: rainfallIntensity }),
      });
      if (!response.ok) {
        throw new Error(`Simulation request failed (${response.status})`);
      }
      const { frames: newFrames } = await response.json();
      setFrames(newFrames);
      setFrameIndex(0);
      setIsPlaying(true);
      setLastRun(new Date());
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsSimulating(false);
    }
  }

  return (
    <main style={styles.mapShell}>
      <Map
        mapboxAccessToken={import.meta.env.VITE_MAPBOX_TOKEN}
        initialViewState={{
          longitude: 73.84, // Deccan Gymkhana
          latitude: 18.51,
          zoom: 14
        }}
        mapStyle={MAP_STYLE}
      >
        {/* Render the dynamically colored roads */}
        <Source id="pune-roads" type="geojson" data={dynamicMapData}>
          <Layer {...roadLayer} />
        </Source>
      </Map>

      <section style={styles.controlPanel} aria-label="Flood simulation controls">
        <div style={styles.panelHeader}>
          <div>
            <p style={styles.eyebrow}>SIH 26085 &middot; Deccan Gymkhana, Pune</p>
            <h1 style={styles.title}>Urban Flood Nowcast</h1>
            <p style={styles.subtitle}>MoES &middot; NCMRWF &middot; Disaster Management</p>
          </div>
          <span style={styles.statusDot} aria-label="API connected" />
        </div>

        <label htmlFor="rainfall-intensity" style={styles.label}>
          <span>Rainfall intensity</span>
          <strong>{rainfallIntensity} mm/hr</strong>
        </label>
        <input
          id="rainfall-intensity"
          type="range"
          min="0"
          max="150"
          step="1"
          value={rainfallIntensity}
          onChange={(event) => setRainfallIntensity(Number(event.target.value))}
          style={styles.slider}
        />
        <div style={styles.rangeLabels} aria-hidden="true">
          <span>0</span>
          <span>150</span>
        </div>
        <p style={{ ...styles.categoryTag, color: rainfallCategory.color }}>
          {rainfallCategory.label}
        </p>

        <button
          type="button"
          onClick={simulateRainfall}
          disabled={isSimulating}
          style={styles.button}
        >
          {isSimulating ? 'Running...' : 'Run Simulation'}
        </button>
        {error && <p role="alert" style={styles.error}>{error}</p>}

        {frames.length > 0 && !error && (
          <div style={styles.summary}>
            <p style={styles.summaryTimestamp}>
              Nowcast run: {lastRun?.toLocaleTimeString()}
            </p>

            <div style={styles.timelineHeader}>
              <button
                type="button"
                onClick={() => setIsPlaying((playing) => !playing)}
                style={styles.playButton}
                aria-label={isPlaying ? 'Pause nowcast playback' : 'Play nowcast playback'}
              >
                {isPlaying ? '⏸' : '▶'}
              </button>
              <label htmlFor="frame-scrubber" style={styles.timelineLabel}>
                <span>0-3hr forecast window</span>
                <strong>T+{currentFrame?.elapsed_min ?? 0} min</strong>
              </label>
            </div>
            <input
              id="frame-scrubber"
              type="range"
              min="0"
              max={Math.max(frames.length - 1, 0)}
              step="1"
              value={frameIndex}
              onChange={(event) => {
                setIsPlaying(false);
                setFrameIndex(Number(event.target.value));
              }}
              style={styles.slider}
            />

            <div style={styles.summaryRow}>
              <span style={{ color: '#ef4444' }}>{runSummary.impassable} impassable</span>
              <span style={{ color: '#eab308' }}>{runSummary.pooling} pooling</span>
              <span style={{ color: '#9ca3af' }}>{runSummary.dry} clear</span>
            </div>
          </div>
        )}
      </section>

      <aside style={styles.legend} aria-label="Flood depth legend">
        <p style={styles.legendTitle}>Road status</p>
        {LEGEND_ITEMS.map((item) => (
          <div key={item.label} style={styles.legendRow}>
            <span style={{ ...styles.legendSwatch, background: item.color }} />
            <span>{item.label}</span>
          </div>
        ))}
      </aside>
    </main>
  );
}

const LEGEND_ITEMS = [
  { label: 'Clear (< 10 cm)', color: '#22c55e' },
  { label: 'Pooling (10-30 cm)', color: '#eab308' },
  { label: 'Impassable (>= 30 cm)', color: '#ef4444' },
];

// Keeping your exact UI styles
const styles = {
  mapShell: { position: 'relative', width: '100vw', height: '100vh', overflow: 'hidden', background: '#111827' },
  controlPanel: { position: 'absolute', left: 24, bottom: 24, width: 'min(340px, calc(100vw - 48px))', padding: 20, color: '#f9fafb', background: 'rgba(17, 24, 39, 0.94)', border: '1px solid rgba(156, 163, 175, 0.3)', borderRadius: 8, boxShadow: '0 12px 30px rgba(0, 0, 0, 0.35)', fontFamily: 'system-ui, sans-serif' },
  panelHeader: { display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 24 },
  eyebrow: { margin: '0 0 4px', color: '#9ca3af', fontSize: 11, fontWeight: 700, letterSpacing: '0.12em', textTransform: 'uppercase' },
  title: { margin: 0, fontSize: 24, lineHeight: 1.1 },
  subtitle: { margin: '4px 0 0', color: '#6b7280', fontSize: 11, letterSpacing: '0.04em' },
  statusDot: { width: 10, height: 10, marginTop: 5, borderRadius: '50%', background: '#22c55e', boxShadow: '0 0 0 4px rgba(34, 197, 94, 0.15)' },
  label: { display: 'flex', justifyContent: 'space-between', gap: 12, color: '#d1d5db', fontSize: 14 },
  slider: { width: '100%', margin: '16px 0 4px', accentColor: '#60a5fa' },
  rangeLabels: { display: 'flex', justifyContent: 'space-between', color: '#6b7280', fontSize: 12 },
  categoryTag: { margin: '10px 0 0', fontSize: 13, fontWeight: 600 },
  button: { width: '100%', marginTop: 20, padding: '11px 14px', color: '#111827', background: '#60a5fa', border: 0, borderRadius: 6, cursor: 'pointer', fontSize: 14, fontWeight: 700 },
  error: { margin: '12px 0 0', color: '#fca5a5', fontSize: 13 },
  summary: { marginTop: 16, paddingTop: 12, borderTop: '1px solid rgba(156, 163, 175, 0.2)' },
  summaryTimestamp: { margin: '0 0 8px', color: '#6b7280', fontSize: 11 },
  summaryRow: { display: 'flex', justifyContent: 'space-between', fontSize: 12, fontWeight: 600, marginTop: 8 },
  timelineHeader: { display: 'flex', alignItems: 'center', gap: 10 },
  playButton: { width: 28, height: 28, flexShrink: 0, borderRadius: '50%', border: '1px solid rgba(156, 163, 175, 0.4)', background: 'transparent', color: '#f9fafb', cursor: 'pointer', fontSize: 12, display: 'flex', alignItems: 'center', justifyContent: 'center' },
  timelineLabel: { flex: 1, display: 'flex', justifyContent: 'space-between', color: '#d1d5db', fontSize: 12 },
  legend: { position: 'absolute', right: 24, top: 24, padding: '14px 16px', color: '#f9fafb', background: 'rgba(17, 24, 39, 0.94)', border: '1px solid rgba(156, 163, 175, 0.3)', borderRadius: 8, boxShadow: '0 12px 30px rgba(0, 0, 0, 0.35)', fontFamily: 'system-ui, sans-serif' },
  legendTitle: { margin: '0 0 10px', color: '#9ca3af', fontSize: 11, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase' },
  legendRow: { display: 'flex', alignItems: 'center', gap: 8, fontSize: 12, marginBottom: 6 },
  legendSwatch: { width: 12, height: 12, borderRadius: 3, flexShrink: 0 },
};