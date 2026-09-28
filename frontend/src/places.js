import puneRoadsData from './pune_roads.json';

const MAPBOX_TOKEN = import.meta.env.VITE_MAPBOX_TOKEN;
const SEARCH_URL = 'https://api.mapbox.com/search/searchbox/v1/forward';
const MAX_ROAD_MATCHES = 3;
const MAX_PLACE_MATCHES = 5;
// Small margin so places on the edge of the road network still show up
const BBOX_PADDING_DEG = 0.002;

// Routing only works on the study-area road network, so search is limited to it
export const ROADS_BBOX = (() => {
  let [minLng, minLat, maxLng, maxLat] = [Infinity, Infinity, -Infinity, -Infinity];
  for (const feature of puneRoadsData.features) {
    for (const [lng, lat] of feature.geometry.coordinates) {
      minLng = Math.min(minLng, lng);
      minLat = Math.min(minLat, lat);
      maxLng = Math.max(maxLng, lng);
      maxLat = Math.max(maxLat, lat);
    }
  }
  return [
    minLng - BBOX_PADDING_DEG,
    minLat - BBOX_PADDING_DEG,
    maxLng + BBOX_PADDING_DEG,
    maxLat + BBOX_PADDING_DEG,
  ];
})();

const BBOX_CENTER = [(ROADS_BBOX[0] + ROADS_BBOX[2]) / 2, (ROADS_BBOX[1] + ROADS_BBOX[3]) / 2];

// OSM splits one street into many ways; list each named street once, placed
// at the middle of its longest piece.
const NAMED_ROADS = (() => {
  const byName = new Map();
  for (const feature of puneRoadsData.features) {
    const { name } = feature.properties;
    if (!name) continue;
    const coordinates = feature.geometry.coordinates;
    const existing = byName.get(name);
    if (!existing || coordinates.length > existing.vertexCount) {
      byName.set(name, {
        name,
        nameMr: feature.properties['name:mr'] ?? existing?.nameMr,
        coordinates: coordinates[Math.floor(coordinates.length / 2)],
        vertexCount: coordinates.length,
      });
    }
  }
  return [...byName.values()];
})();

/** Roads in the study area whose English or Marathi name contains the query. */
export function searchRoads(query) {
  const needle = query.trim().toLowerCase();
  if (needle.length < 2) return [];
  return NAMED_ROADS
    .filter((road) => road.name.toLowerCase().includes(needle) || road.nameMr?.includes(query.trim()))
    // Prefix matches first: "Kar" should rank Karve Road above Tilak Road (Karve)
    .sort((a, b) => Number(!a.name.toLowerCase().startsWith(needle)) - Number(!b.name.toLowerCase().startsWith(needle)))
    .slice(0, MAX_ROAD_MATCHES)
    .map((road) => ({
      id: `road:${road.name}`,
      label: road.name,
      detail: road.nameMr ? `Road in study area · ${road.nameMr}` : 'Road in study area',
      coordinates: road.coordinates,
    }));
}

/** Places (hospitals, colleges, addresses…) from Mapbox, limited to the study area. */
export async function searchPlaces(query, signal) {
  const params = new URLSearchParams({
    q: query.trim(),
    bbox: ROADS_BBOX.join(','),
    proximity: BBOX_CENTER.join(','),
    limit: String(MAX_PLACE_MATCHES),
    language: 'en',
    access_token: MAPBOX_TOKEN,
  });
  const response = await fetch(`${SEARCH_URL}?${params}`, { signal });
  if (!response.ok) throw new Error(`Place search failed (${response.status})`);
  const body = await response.json();
  return body.features.map((feature) => ({
    id: `place:${feature.properties.mapbox_id ?? feature.geometry.coordinates.join(',')}`,
    label: feature.properties.name,
    detail: feature.properties.place_formatted ?? '',
    coordinates: feature.geometry.coordinates,
  }));
}

/** A readable label for a point picked on the map: the nearest named road. */
export function describePoint([lng, lat]) {
  // Equirectangular distance is plenty at street scale
  const lngScale = Math.cos((lat * Math.PI) / 180);
  let nearest = null;
  let nearestDistance = Infinity;
  for (const feature of puneRoadsData.features) {
    if (!feature.properties.name) continue;
    for (const [roadLng, roadLat] of feature.geometry.coordinates) {
      const distance = ((roadLng - lng) * lngScale) ** 2 + (roadLat - lat) ** 2;
      if (distance < nearestDistance) {
        nearestDistance = distance;
        nearest = feature.properties.name;
      }
    }
  }
  return nearest ? `Near ${nearest}` : 'Pinned location';
}
