import { useEffect, useState } from 'react';
import Map, { Layer, Source } from 'react-map-gl/mapbox';
import 'mapbox-gl/dist/mapbox-gl.css';
import puneRoadsData from './pune_roads.json';

const API_BASE_URL = 'http://127.0.0.1:8000';
const MAP_STYLE = 'mapbox://styles/mapbox/dark-v11';

const pipeLayer = {
  id: 'flood-pipes',
  type: 'line',
  filter: ['==', ['geometry-type'], 'LineString'],
  layout: {
    'line-cap': 'round',
    'line-join': 'round',
  },
  paint: {
    'line-color': [
      'case',
      ['==', ['get', 'status'], 'SURCHARGING'],
      '#ef4444',
      '#22c55e',
    ],
    'line-width': 5,
    'line-opacity': 0.9,
  },
};

const nodeLayer = {
  id: 'flood-nodes',
  type: 'circle',
  filter: ['==', ['geometry-type'], 'Point'],
  paint: {
    'circle-color': [
      'case',
      ['==', ['get', 'status'], 'FLOODED'],
      '#3b82f6',
      '#9ca3af',
    ],
    'circle-radius': 8,
    'circle-stroke-color': '#ffffff',
    'circle-stroke-width': 1.5,
  },
};

export default function FloodNowcastMap() {
  const [networkData, setNetworkData] = useState(null);
  const [rainfallIntensity, setRainfallIntensity] = useState(50);
  const [error, setError] = useState('');
  const [isSimulating, setIsSimulating] = useState(false);

  useEffect(() => {
    const controller = new AbortController();

    async function loadNetwork() {
      try {
        const response = await fetch(`${API_BASE_URL}/network`, {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`Network request failed (${response.status})`);
        }
        setNetworkData(await response.json());
      } catch (requestError) {
        if (requestError.name !== 'AbortError') {
          setError(requestError.message);
        }
      }
    }

    loadNetwork();
    return () => controller.abort();
  }, []);

  async function simulateRainfall() {
    setIsSimulating(true);
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
      setNetworkData(await response.json());
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setIsSimulating(false);
    }
  }

  return (
    <main style={styles.mapShell}>
      <Map
        initialViewState={{
          longitude: 73.84, // Deccan Gymkhana approx longitude
          latitude: 18.51,  // Deccan Gymkhana approx latitude
          zoom: 14
        }}
        mapStyle="mapbox://styles/mapbox/dark-v11"
      >
        {/* The Static Road Geometry */}
        <Source id="pune-roads" type="geojson" data={puneRoadsData}>
          {/* The Data-Driven Styling Layer */}
          <Layer 
            id="road-floods" 
            type="line" 
            paint={{
              'line-width': 3,
              // We will add the color expression here later based on your API response
              'line-color': '#00FF00' 
            }} 
          />
        </Source>
      </Map>

      <section style={styles.controlPanel} aria-label="Flood simulation controls">
        <div style={styles.panelHeader}>
          <div>
            <p style={styles.eyebrow}>Pune network</p>
            <h1 style={styles.title}>Flood nowcast</h1>
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

        <button
          type="button"
          onClick={simulateRainfall}
          disabled={isSimulating}
          style={styles.button}
        >
          {isSimulating ? 'Running...' : 'Run Simulation'}
        </button>
        {error && <p role="alert" style={styles.error}>{error}</p>}
      </section>
    </main>
  );
}

const styles = {
  mapShell: {
    position: 'relative',
    width: '100vw',
    height: '100vh',
    overflow: 'hidden',
    background: '#111827',
  },
  controlPanel: {
    position: 'absolute',
    left: 24,
    bottom: 24,
    width: 'min(340px, calc(100vw - 48px))',
    padding: 20,
    color: '#f9fafb',
    background: 'rgba(17, 24, 39, 0.94)',
    border: '1px solid rgba(156, 163, 175, 0.3)',
    borderRadius: 8,
    boxShadow: '0 12px 30px rgba(0, 0, 0, 0.35)',
    fontFamily: 'system-ui, sans-serif',
  },
  panelHeader: {
    display: 'flex',
    alignItems: 'flex-start',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  eyebrow: {
    margin: '0 0 4px',
    color: '#9ca3af',
    fontSize: 11,
    fontWeight: 700,
    letterSpacing: '0.12em',
    textTransform: 'uppercase',
  },
  title: {
    margin: 0,
    fontSize: 24,
    lineHeight: 1.1,
  },
  statusDot: {
    width: 10,
    height: 10,
    marginTop: 5,
    borderRadius: '50%',
    background: '#22c55e',
    boxShadow: '0 0 0 4px rgba(34, 197, 94, 0.15)',
  },
  label: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: 12,
    color: '#d1d5db',
    fontSize: 14,
  },
  slider: {
    width: '100%',
    margin: '16px 0 4px',
    accentColor: '#60a5fa',
  },
  rangeLabels: {
    display: 'flex',
    justifyContent: 'space-between',
    color: '#6b7280',
    fontSize: 12,
  },
  button: {
    width: '100%',
    marginTop: 20,
    padding: '11px 14px',
    color: '#111827',
    background: '#60a5fa',
    border: 0,
    borderRadius: 6,
    cursor: 'pointer',
    fontSize: 14,
    fontWeight: 700,
  },
  error: {
    margin: '12px 0 0',
    color: '#fca5a5',
    fontSize: 13,
  },
};

