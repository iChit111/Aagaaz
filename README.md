# Urban Flood Nowcast

Urban flood simulation and flood-safe routing for Deccan Gymkhana, Pune. The project combines a FastAPI backend, a PySWMM drainage simulation, DEM-derived catchment runoff and flood extents, and a React/Mapbox frontend.

## What it does

- Simulates a 0-3 hour storm for a configurable rainfall intensity.
- Converts drainage-network surcharge into estimated street-level flood depths.
- Displays roads as dry, pooling, or impassable on an interactive map.
- Finds a route between two map points while avoiding flooded roads where a detour exists.
- Serves a synthetic spatial rainfall nowcast through the API. The integration point for a live radar feed is in `nowcast.py`.

## How it works

Data prep (run ahead of time, produces the files the API loads at startup):

- **`dem.py`** — Downloads the SRTM 30m elevation tile for the AOI from OpenTopography and caches it locally as a GeoTIFF.
- **`generate_network.py`** — Picks 25 real street intersections from the road graph (`routing.py`), looks up their DEM elevation, and connects each to its nearest downhill neighbor to build a plausible drainage pipe network. Writes `pune_base.inp` (SWMM input) and `network_topology.py` (junction coordinates).
- **`terrain.py`** — Uses **pysheds** to condition the DEM (fills pits/depressions, resolves flats), compute D8 flow direction and flow accumulation, then delineates each junction's contributing catchment (the patch of land that drains into it). Catchments are capped to a 200m radius around each junction so they stay street-scale rather than growing into a regional watershed. Saves the results to `data/catchments.npz`.

Runtime (what happens on each `/simulate` call):

1. **`nowcast.py`** — No live radar feed is available, so this generates a synthetic rainstorm: a drifting, bell-shaped intensity blob queryable at any point and time over the 0-3hr window.
2. **`runoff.py`** — Averages that synthetic rainfall over each junction's catchment (from `terrain.py`) and converts it into a volumetric inflow rate (m³/s) using a runoff coefficient (0.9, concrete-heavy urban assumption).
3. **PySWMM** — `main.py`'s `/simulate` endpoint feeds each junction's inflow into PySWMM every timestep. PySWMM (a real EPA SWMM hydraulics engine) routes the water through the pipe network and determines when a junction's capacity is exceeded, exposing the overflow as `node.statistics["flooding_volume"]`. This overflow math itself is entirely inside PySWMM — this project only supplies inflow and reads the result back out. If PySWMM throws an exception, a fallback hardcodes surcharge at every junction above a rainfall threshold so the demo still shows something (flagged via the `X-Engine-Status: fallback` header).
4. **`flood_fill.py`** — Turns each junction's overflow volume into a street-level puddle: starting from the junction's location on the DEM, it fills in the lowest neighboring terrain cells first (a "bathtub fill" / priority-flood), stopping once the pooled volume matches the overflow. It then checks which road vertices (from `frontend/src/pune_roads.json`) fall inside the puddle to report per-road depth in cm.
5. **`routing.py`** — Builds a graph from the same road GeoJSON and finds a route with **Dijkstra's shortest-path algorithm** (via NetworkX), with edge weights adjusted by current flood depth: roads ≥30cm are excluded entirely wherever a dry detour exists, roads ≥10cm are heavily penalized (×6) but passable, and if no dry route exists at all it falls back to allowing flooded roads (penalized ×50) and flags the result `degraded: true`.

In one sentence: fake rain falls on real terrain → hydrology math (pysheds) computes how much water reaches each drain → a real hydraulics engine (PySWMM) simulates the pipes backing up → the overflow is spread across the DEM as a virtual puddle → nearby streets get marked flooded → Dijkstra-based routing avoids those streets.

## Project structure

```text
main.py                 FastAPI application
nowcast.py              Synthetic rainfall nowcast
runoff.py               DEM-derived catchment runoff
flood_fill.py           Flood extent and road-depth estimation
terrain.py              DEM and catchment processing
routing.py              Flood-aware road routing
generate_network.py     Regenerate the SWMM network and topology
pune_base.inp           SWMM input network
data/                   Generated catchment data
frontend/               React + Vite map application
```

## Requirements

- Python 3.11 or newer
- Node.js 18 or newer
- A Mapbox public access token
- A working PySWMM/SWMM runtime supported by `pyswmm`

## Setup

Create and activate a Python virtual environment, then install the backend dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy the environment template if terrain data needs to be downloaded and add an OpenTopography key when required:

```bash
cp .env.example .env
```

Install the frontend dependencies and configure its Mapbox token:

```bash
cd frontend
npm install
printf 'VITE_MAPBOX_TOKEN=your_mapbox_public_token\n' > .env.local
cd ..
```

## Run locally

Start the API from the repository root:

```bash
source .venv/bin/activate
uvicorn main:app --reload
```

In a second terminal, start the map frontend:

```bash
cd frontend
npm run dev
```

Open the URL printed by Vite, usually `http://localhost:5173`. The frontend expects the API at `http://127.0.0.1:8000`.

The API also exposes interactive documentation at `http://127.0.0.1:8000/docs`.

## API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/network` | Return the baseline drainage network as GeoJSON. |
| `POST` | `/simulate` | Run a storm simulation. Body: `{"rainfall_mm_per_hr": 50}`. |
| `GET` | `/nowcast` | Return a synthetic rainfall intensity grid. |
| `POST` | `/route` | Find a route using origin, destination, and current road depths. |

`/simulate` captures frames at 15-minute intervals. If PySWMM fails, the API returns a fallback frame and sets the `X-Engine-Status: fallback` response header.

## Regenerate model data

The checked-in `pune_base.inp`, `network_topology.py`, and catchment data are the runnable model inputs. To regenerate them after changing the road or DEM data:

```bash
python generate_network.py
python terrain.py
```

Review generated outputs before running the API, since regeneration changes the drainage topology and catchment assignments.

## Frontend checks

```bash
cd frontend
npm run lint
npm run build
```
