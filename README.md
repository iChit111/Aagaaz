# Urban Flood Nowcast

Urban flood simulation and flood-safe routing for Deccan Gymkhana, Pune. The project combines a FastAPI backend, a PySWMM drainage simulation, DEM-derived catchment runoff and flood extents, and a React/Mapbox frontend.

## What it does

- Simulates a 0-3 hour storm for a configurable rainfall intensity.
- Converts drainage-network surcharge into estimated street-level flood depths.
- Displays roads as dry, pooling, or impassable on an interactive map.
- Finds a route between two map points while avoiding flooded roads where a detour exists.
- Serves a synthetic spatial rainfall nowcast through the API. The integration point for a live radar feed is in `nowcast.py`.

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
