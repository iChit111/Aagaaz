import { useState, useMemo, useEffect, useRef } from 'react';
import Map, { Layer, Source, Marker, Popup } from 'react-map-gl/mapbox';
import 'mapbox-gl/dist/mapbox-gl.css';
import './FloodNowcastMap.css';
import puneRoadsData from './pune_roads.json';
import RouteStopsInput from './RouteStopsInput';
import { describePoint } from './places';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
// Layer 1 (PySWMM physics) is the authoritative output shown to users; the
// GNN surrogate at /simulate_gnn is not displayed.
const SIMULATE_ENDPOINT = '/simulate';
const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;
const MAP_STYLE = 'mapbox://styles/mapbox/dark-v11';
const HEALTH_CHECK_INTERVAL_MS = 30000;
const HEALTH_CHECK_TIMEOUT_MS = 5000;
// Wait for the scrubber to settle before re-planning the route
const ROUTE_DEBOUNCE_MS = 250;
const EMPTY_STOPS = { origin: null, destination: null };
const STOP_COLORS = { origin: '#22c55e', destination: '#ef4444' };
// Zoom used when jumping to a single searched place
const SEARCH_RESULT_ZOOM = 15.5;

const TOTAL_ROADS = puneRoadsData.features.length;
// Stable fallback so memos/effects keyed on depths don't re-run every render
const EMPTY_DEPTHS = {};

// IMD real-time rainfall intensity brackets (mm/hr)
const RAINFALL_CATEGORIES = [
  { max: 2.5, label: 'No / Light Drizzle', color: '#38bdf8' },
  { max: 7.5, label: 'Light Rain', color: '#22c55e' },
  { max: 35.5, label: 'Moderate Rain', color: '#eab308' },
  { max: 64.5, label: 'Heavy Rain', color: '#f97316' },
  { max: 124.5, label: 'Very Heavy Rain', color: '#ef4444' },
  { max: Infinity, label: 'Extreme Rain', color: '#b91c1c' },
];

const RAINFALL_MAX = 150;

// One-tap starting points, one per IMD category from moderate upwards
const RAINFALL_PRESETS = [
  { label: 'Moderate', value: 25 },
  { label: 'Heavy', value: 50 },
  { label: 'Very heavy', value: 100 },
  { label: 'Extreme', value: 150 },
];

// Hard-edged colour bands so the slider track shows where each category begins
const RAINFALL_TRACK_GRADIENT = (() => {
  let start = 0;
  const stops = RAINFALL_CATEGORIES.map((category) => {
    const end = (Math.min(category.max, RAINFALL_MAX) / RAINFALL_MAX) * 100;
    const stop = `${category.color} ${start}% ${end}%`;
    start = end;
    return stop;
  });
  return `linear-gradient(to right, ${stops.join(', ')})`;
})();

const API_STATUS_LABELS = {
  checking: 'Checking server…',
  online: 'Server online',
  offline: 'Server offline',
};

const FLOOD_STATUS_LABELS = {
  dry: 'Clear',
  pooling: 'Pooling',
  impassable: 'Impassable',
};

function getRainfallCategory(mmPerHr) {
  return RAINFALL_CATEGORIES.find((bucket) => mmPerHr <= bucket.max);
}

function getFloodStatus(depthCm) {
  if (depthCm >= 30) return 'impassable';
  if (depthCm >= 10) return 'pooling';
  return 'dry';
}

function getRoadId(feature) {
  // Overpass Turbo usually assigns OSM IDs as strings like "way/12345"
  const roadId = feature.id ?? feature.properties?.id ?? feature.properties?.['@id'];
  return roadId == null ? null : String(roadId);
}

// fetch() rejects with a TypeError when the server can't be reached at all,
// which the browser reports as an unhelpful "Failed to fetch".
function describeRequestError(error) {
  if (error instanceof TypeError) {
    return `Can't reach the simulation server at ${API_BASE_URL}. Check that the backend is running.`;
  }
  return error.message;
}

async function readErrorDetail(response, fallback) {
  try {
    const body = await response.json();
    if (typeof body.detail === 'string') return body.detail;
  } catch {
    // Non-JSON error body; fall through to the generic message
  }
  return `${fallback} (${response.status})`;
}

function formatDepth(depthCm) {
  return `${Math.round(depthCm)} cm`;
}

// A dark halo under a bright line reads clearly regardless of the road
// color underneath (green/yellow/red), unlike a single mid-tone line.
const routeHaloLayer = {
  id: 'flood-safe-route-halo',
  type: 'line',
  paint: {
    'line-width': 9,
    'line-color': '#0f172a',
    'line-opacity': 0.85,
  },
};

const routeLayer = {
  id: 'flood-safe-route',
  type: 'line',
  paint: {
    'line-width': 4,
    'line-color': '#ffffff',
    'line-dasharray': [0.2, 1.5],
  },
};

// Flooded roads the route still has to use, drawn under the dashed route line
const routeFloodedLayer = {
  id: 'flood-safe-route-flooded',
  type: 'line',
  paint: {
    'line-width': 9,
    'line-color': ['case', ['>=', ['get', 'flood_depth'], 30], '#ef4444', '#f97316'],
  },
};

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

// Invisible, wider copy of the roads so the 4px lines are easy to hover/tap
const roadHitLayer = {
  id: 'road-hit-area',
  type: 'line',
  paint: {
    'line-width': 16,
    'line-opacity': 0,
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
  const [apiStatus, setApiStatus] = useState('checking');

  const [routeMode, setRouteMode] = useState(false);
  // { origin, destination }, each { coordinates: [lon, lat], label } or null
  const [routeStops, setRouteStops] = useState(EMPTY_STOPS);
  // The stop the next map click fills: the field last focused, else the first empty one
  const [activeStop, setActiveStop] = useState(null);
  const mapRef = useRef(null);
  const [route, setRoute] = useState(null);
  const [routeError, setRouteError] = useState('');
  const [isRouting, setIsRouting] = useState(false);

  // { roadId, name, nameMr, longitude, latitude } for the road under the pointer
  const [hoveredRoad, setHoveredRoad] = useState(null);

  const rainfallCategory = getRainfallCategory(rainfallIntensity);
  const currentFrame = frames[frameIndex];
  const currentDepths = currentFrame?.depths ?? EMPTY_DEPTHS;
  const currentElapsedMin = currentFrame?.elapsed_min ?? 0;

  const runSummary = useMemo(() => {
    const depths = Object.values(currentDepths);
    const impassable = depths.filter((d) => getFloodStatus(d) === 'impassable').length;
    const pooling = depths.filter((d) => getFloodStatus(d) === 'pooling').length;
    // The API only reports depths for roads inside a flood extent, so every
    // other road in the study area is clear.
    return { impassable, pooling, dry: TOTAL_ROADS - impassable - pooling };
  }, [currentDepths]);

  // Per-road peak depth and when it first floods, across the whole 0-3hr window
  const roadTimeline = useMemo(() => {
    const byRoad = {};
    for (const frame of frames) {
      for (const [roadId, depth] of Object.entries(frame.depths)) {
        const entry = (byRoad[roadId] ??= { peakDepth: 0, peakMin: null, firstFloodMin: null });
        if (depth > entry.peakDepth) {
          entry.peakDepth = depth;
          entry.peakMin = frame.elapsed_min;
        }
        if (entry.firstFloodMin === null && depth >= 10) {
          entry.firstFloodMin = frame.elapsed_min;
        }
      }
    }
    return byRoad;
  }, [frames]);

  // The Injection: Merge static roads with the currently scrubbed frame's depths
  const dynamicMapData = useMemo(() => {
    const updatedFeatures = puneRoadsData.features.map(feature => {
      const roadId = getRoadId(feature);
      const currentDepth = roadId == null ? 0 : currentDepths[roadId] ?? 0;

      return {
        ...feature,
        properties: {
          ...feature.properties,
          road_id: roadId,
          flood_depth: currentDepth
        }
      };
    });

    return { ...puneRoadsData, features: updatedFeatures };
  }, [currentDepths]);

  const routeGeoJson = useMemo(() => {
    if (!route) return null;
    return {
      type: 'Feature',
      geometry: { type: 'LineString', coordinates: route.path },
      properties: {},
    };
  }, [route]);

  const routeFloodedGeoJson = useMemo(() => {
    if (!route || route.flooded_road_ids.length === 0) return null;
    const floodedIds = new Set(route.flooded_road_ids);
    return {
      type: 'FeatureCollection',
      features: dynamicMapData.features.filter((feature) => floodedIds.has(feature.properties.road_id)),
    };
  }, [route, dynamicMapData]);

  // Poll the backend so the header shows whether simulations can actually run
  useEffect(() => {
    let cancelled = false;
    async function checkApi() {
      try {
        const response = await fetch(`${API_BASE_URL}/network`, {
          signal: AbortSignal.timeout(HEALTH_CHECK_TIMEOUT_MS),
        });
        if (!cancelled) setApiStatus(response.ok ? 'online' : 'offline');
      } catch {
        if (!cancelled) setApiStatus('offline');
      }
    }
    checkApi();
    const timer = setInterval(checkApi, HEALTH_CHECK_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

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

  // Re-plan whenever the endpoints move or the flood picture changes (new run,
  // scrubbing, playback), so the route always matches the frame on screen.
  useEffect(() => {
    if (!routeStops.origin || !routeStops.destination) return undefined;
    const origin = routeStops.origin.coordinates;
    const destination = routeStops.destination.coordinates;
    let cancelled = false;

    const timer = setTimeout(async () => {
      setIsRouting(true);
      setRouteError('');
      try {
        const response = await fetch(`${API_BASE_URL}/route`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ origin, destination, depths: currentDepths }),
        });
        if (!response.ok) {
          throw new Error(await readErrorDetail(response, 'Routing failed'));
        }
        const body = await response.json();
        if (!cancelled) setRoute({ ...body, elapsed_min: currentElapsedMin });
      } catch (requestError) {
        if (!cancelled) {
          setRoute(null);
          setRouteError(describeRequestError(requestError));
        }
      } finally {
        if (!cancelled) setIsRouting(false);
      }
    }, ROUTE_DEBOUNCE_MS);

    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [routeStops, currentDepths, currentElapsedMin]);

  async function simulateRainfall() {
    setIsSimulating(true);
    setIsPlaying(false);
    setError('');

    try {
      const response = await fetch(`${API_BASE_URL}${SIMULATE_ENDPOINT}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rainfall_mm_per_hr: rainfallIntensity }),
      });
      if (!response.ok) {
        throw new Error(await readErrorDetail(response, 'Simulation request failed'));
      }
      const { frames: newFrames } = await response.json();
      setApiStatus('online');
      setFrames(newFrames);
      setFrameIndex(0);
      setIsPlaying(true);
      setLastRun(new Date());
    } catch (requestError) {
      if (requestError instanceof TypeError) setApiStatus('offline');
      // Drop the previous run so its depths aren't mistaken for this one's
      setFrames([]);
      setFrameIndex(0);
      setLastRun(null);
      setError(describeRequestError(requestError));
    } finally {
      setIsSimulating(false);
    }
  }

  function clearRoute() {
    setRouteStops(EMPTY_STOPS);
    setActiveStop(null);
    setRoute(null);
    setRouteError('');
    setIsRouting(false);
  }

  function toggleRouteMode() {
    setRouteMode((mode) => !mode);
    clearRoute();
  }

  // Bring a searched place into view; with both stops set, frame the whole trip
  function showStops(stops) {
    const map = mapRef.current;
    if (!map) return;
    // Keep the stops clear of the control panel (left on desktop, bottom on phones)
    const padding = window.innerWidth > 600
      ? { top: 80, bottom: 80, left: 400, right: 80 }
      : { top: 60, bottom: Math.round(window.innerHeight * 0.55), left: 40, right: 40 };
    const points = [stops.origin, stops.destination].filter(Boolean).map((stop) => stop.coordinates);
    if (points.length === 2) {
      const lngs = points.map(([lng]) => lng);
      const lats = points.map(([, lat]) => lat);
      map.fitBounds(
        [[Math.min(...lngs), Math.min(...lats)], [Math.max(...lngs), Math.max(...lats)]],
        { padding, maxZoom: 16, duration: 800 }
      );
    } else if (points.length === 1) {
      map.flyTo({ center: points[0], zoom: Math.max(map.getZoom(), SEARCH_RESULT_ZOOM), padding, duration: 800 });
    }
  }

  function setStop(field, stop, { fromSearch = false } = {}) {
    const nextStops = { ...routeStops, [field]: stop };
    setRouteStops(nextStops);
    const otherField = field === 'origin' ? 'destination' : 'origin';
    setActiveStop(nextStops[otherField] ? null : otherField);
    if (fromSearch) showStops(nextStops);
  }

  function clearStop(field) {
    setRouteStops((stops) => ({ ...stops, [field]: null }));
    setActiveStop(field);
    setRoute(null);
    setRouteError('');
  }

  function swapRouteStops() {
    setRouteStops(({ origin, destination }) => ({ origin: destination, destination: origin }));
  }

  function stopAt(lngLat) {
    const coordinates = [lngLat.lng, lngLat.lat];
    return { coordinates, label: describePoint(coordinates) };
  }

  function moveRouteStop(field, lngLat) {
    setRouteStops((stops) => ({ ...stops, [field]: stopAt(lngLat) }));
  }

  function roadFromEvent(event) {
    const feature = event.features?.[0];
    if (!feature) return null;
    return {
      roadId: feature.properties.road_id,
      name: feature.properties.name,
      nameMr: feature.properties['name:mr'],
      longitude: event.lngLat.lng,
      latitude: event.lngLat.lat,
    };
  }

  function handleMapClick(event) {
    if (!routeMode) {
      // Touch screens have no hover, so a tap shows the road's details instead
      setHoveredRoad(roadFromEvent(event));
      return;
    }
    const field = activeStop ?? (!routeStops.origin ? 'origin' : !routeStops.destination ? 'destination' : null);
    // Once both stops are set they're adjusted by dragging the pins, not by clicking
    if (!field) return;
    setStop(field, stopAt(event.lngLat));
  }

  if (!MAPBOX_TOKEN) {
    return (
      <main className="flood-map flood-map--notice">
        <div className="panel notice-card" role="alert">
          <h1 className="panel__title">Map can't load</h1>
          <p>
            <code>VITE_MAPBOX_TOKEN</code> isn't set. Add your Mapbox public token to{' '}
            <code>frontend/.env.local</code> and restart the dev server.
          </p>
        </div>
      </main>
    );
  }

  const hoveredStats = hoveredRoad ? roadTimeline[hoveredRoad.roadId] : null;
  const hoveredDepth = hoveredRoad ? currentDepths[hoveredRoad.roadId] ?? 0 : 0;
  const hoveredStatus = getFloodStatus(hoveredDepth);

  let mapCursor = 'grab';
  if (routeMode && (activeStop || !routeStops.origin || !routeStops.destination)) mapCursor = 'crosshair';
  else if (hoveredRoad) mapCursor = 'pointer';

  return (
    <main className="flood-map" aria-busy={isSimulating}>
      <Map
        ref={mapRef}
        mapboxAccessToken={MAPBOX_TOKEN}
        initialViewState={{
          longitude: 73.84, // Deccan Gymkhana
          latitude: 18.51,
          zoom: 14
        }}
        mapStyle={MAP_STYLE}
        interactiveLayerIds={[roadHitLayer.id]}
        onClick={handleMapClick}
        onMouseMove={(event) => setHoveredRoad(roadFromEvent(event))}
        onMouseLeave={() => setHoveredRoad(null)}
        cursor={mapCursor}
      >
        {/* Render the dynamically colored roads */}
        <Source id="pune-roads" type="geojson" data={dynamicMapData}>
          <Layer {...roadLayer} />
          <Layer {...roadHitLayer} />
        </Source>

        {routeGeoJson && (
          <Source id="flood-safe-route" type="geojson" data={routeGeoJson}>
            <Layer {...routeHaloLayer} />
            <Layer {...routeLayer} />
          </Source>
        )}

        {routeFloodedGeoJson && (
          <Source id="flood-safe-route-flooded" type="geojson" data={routeFloodedGeoJson}>
            <Layer {...routeFloodedLayer} beforeId={routeLayer.id} />
          </Source>
        )}

        {['origin', 'destination'].map((field) => routeStops[field] && (
          <Marker
            key={field}
            longitude={routeStops[field].coordinates[0]}
            latitude={routeStops[field].coordinates[1]}
            color={STOP_COLORS[field]}
            draggable
            onDragEnd={(event) => moveRouteStop(field, event.lngLat)}
          />
        ))}

        {hoveredRoad && (
          <Popup
            longitude={hoveredRoad.longitude}
            latitude={hoveredRoad.latitude}
            closeButton={false}
            closeOnClick={false}
            offset={14}
            maxWidth="260px"
            className="road-popup"
            onClose={() => setHoveredRoad(null)}
          >
            <p className="road-popup__name">{hoveredRoad.name || 'Unnamed road'}</p>
            {hoveredRoad.nameMr && (
              <p className="road-popup__name-mr" lang="mr">{hoveredRoad.nameMr}</p>
            )}
            {frames.length === 0 ? (
              <p className="road-popup__hint">Run a simulation to see flood depth here.</p>
            ) : (
              <>
                <p className="road-popup__now">
                  <span className={`status-badge status-badge--${hoveredStatus}`}>
                    {FLOOD_STATUS_LABELS[hoveredStatus]}
                  </span>
                  <strong>{formatDepth(hoveredDepth)}</strong>
                  <span className="road-popup__time">at T+{currentElapsedMin} min</span>
                </p>
                <dl className="road-popup__stats">
                  <dt>Peak</dt>
                  <dd>
                    {hoveredStats?.peakDepth
                      ? `${formatDepth(hoveredStats.peakDepth)} at T+${hoveredStats.peakMin} min`
                      : 'Stays dry'}
                  </dd>
                  <dt>Floods from</dt>
                  <dd>
                    {hoveredStats?.firstFloodMin != null
                      ? `T+${hoveredStats.firstFloodMin} min`
                      : 'Stays below 10 cm'}
                  </dd>
                </dl>
              </>
            )}
          </Popup>
        )}
      </Map>

      <div className={`map-dim${isSimulating ? ' map-dim--active' : ''}`} aria-hidden="true" />

      <section className="panel control-panel" aria-label="Flood simulation controls">
        <div className="panel__header">
          <p className="eyebrow">SIH 26085 &middot; Deccan Gymkhana, Pune</p>
          <h1 className="panel__title">Urban Flood Nowcast</h1>
          <p className="panel__subtitle">MoES &middot; NCMRWF &middot; Disaster Management</p>
          <p className={`api-status api-status--${apiStatus}`} role="status">
            <span className="api-status__dot" aria-hidden="true" />
            {API_STATUS_LABELS[apiStatus]}
          </p>
        </div>

        {apiStatus === 'offline' && (
          <p className="notice notice--warning">
            Can't reach the simulation server at <code>{API_BASE_URL}</code>. Start the backend,
            then run a simulation.
          </p>
        )}

        <label htmlFor="rainfall-intensity" className="field-label">
          <span>Rainfall intensity</span>
          <strong>{rainfallIntensity} mm/hr</strong>
        </label>
        <input
          id="rainfall-intensity"
          type="range"
          min="0"
          max={RAINFALL_MAX}
          step="1"
          value={rainfallIntensity}
          onChange={(event) => setRainfallIntensity(Number(event.target.value))}
          disabled={isSimulating}
          aria-valuetext={`${rainfallIntensity} mm/hr, ${rainfallCategory.label}`}
          className="slider slider--rainfall"
          style={{
            '--track-gradient': RAINFALL_TRACK_GRADIENT,
            '--thumb-color': rainfallCategory.color,
          }}
        />
        <div className="range-labels" aria-hidden="true">
          <span>0</span>
          <span>{RAINFALL_MAX}</span>
        </div>
        <p className="category-tag" style={{ color: rainfallCategory.color }}>
          {rainfallCategory.label}
        </p>

        <div className="rain-presets" role="group" aria-label="Rainfall presets">
          {RAINFALL_PRESETS.map((preset) => (
            <button
              key={preset.value}
              type="button"
              onClick={() => setRainfallIntensity(preset.value)}
              disabled={isSimulating}
              aria-pressed={rainfallIntensity === preset.value}
              className="rain-preset"
              style={{ '--preset-color': getRainfallCategory(preset.value).color }}
            >
              <span className="rain-preset__label">{preset.label}</span>
              <span className="rain-preset__value">{preset.value} mm/hr</span>
            </button>
          ))}
        </div>

        <button
          type="button"
          onClick={simulateRainfall}
          disabled={isSimulating}
          className="button button--primary"
        >
          {isSimulating ? (
            <>
              <span className="spinner" aria-hidden="true" />
              Running simulation…
            </>
          ) : (
            'Run Simulation'
          )}
        </button>
        {error && <p role="alert" className="notice notice--error">{error}</p>}

        {frames.length > 0 && !error && (
          <div className="panel__section">
            <p className="summary-timestamp">
              Physics engine (PySWMM) &middot; run at {lastRun?.toLocaleTimeString()}
            </p>

            <div className="timeline-header">
              <button
                type="button"
                onClick={() => setIsPlaying((playing) => !playing)}
                className="play-button"
                aria-label={isPlaying ? 'Pause nowcast playback' : 'Play nowcast playback'}
              >
                {isPlaying ? '⏸' : '▶'}
              </button>
              <label htmlFor="frame-scrubber" className="timeline-label">
                <span>0-3hr forecast window</span>
                <strong>T+{currentElapsedMin} min</strong>
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
              className="slider"
            />

            <div className="summary-row">
              <span className="summary-row__impassable">{runSummary.impassable} impassable</span>
              <span className="summary-row__pooling">{runSummary.pooling} pooling</span>
              <span className="summary-row__dry">{runSummary.dry} clear</span>
            </div>
            <p className="summary-hint">Hover or tap a road for its depth and flood timing.</p>
          </div>
        )}

        <div className="panel__section">
          <button
            type="button"
            onClick={toggleRouteMode}
            className={`button ${routeMode ? 'button--danger' : 'button--route'}`}
          >
            {routeMode ? 'Exit route planning' : 'Plan flood-safe route'}
          </button>

          {routeMode && (
            <>
              <RouteStopsInput
                stops={routeStops}
                onFocusStop={setActiveStop}
                onSelectStop={setStop}
                onClearStop={clearStop}
                onSwap={swapRouteStops}
              />
              {routeStops.origin && routeStops.destination && (
                <div className="route-actions">
                  <p className="route-hint">Drag the pins to fine-tune the route.</p>
                  <button type="button" onClick={clearRoute} className="button button--secondary button--inline">
                    Clear
                  </button>
                </div>
              )}
            </>
          )}

          {isRouting && <p className="route-hint" role="status">Finding a flood-safe path…</p>}
          {routeError && <p role="alert" className="notice notice--error">{routeError}</p>}

          {route && !routeError && (
            <div className="route-summary">
              <p className="route-summary__distance">
                {(route.distance_m / 1000).toFixed(2)} km
                {frames.length > 0 && (
                  <span className="route-summary__time">conditions at T+{route.elapsed_min} min</span>
                )}
              </p>
              {route.degraded ? (
                <p className="route-hint route-hint--danger">
                  No fully dry route exists. This path still crosses {route.flooded_road_ids.length}{' '}
                  flooded segment(s), highlighted on the map.
                </p>
              ) : route.flooded_road_ids.length > 0 ? (
                <p className="route-hint route-hint--warning">
                  Avoids impassable roads, but passes {route.flooded_road_ids.length} pooling
                  segment(s), highlighted on the map.
                </p>
              ) : (
                <p className="route-hint route-hint--safe">Fully dry route.</p>
              )}
            </div>
          )}
        </div>
      </section>

      <aside className="panel legend" aria-label="Flood depth legend">
        <p className="eyebrow">Road status</p>
        {LEGEND_ITEMS.map((item) => (
          <div key={item.label} className="legend__row">
            <span className="legend__swatch" style={{ background: item.color }} />
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
