import os
import math
import csv
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import psycopg2
from psycopg2.extras import RealDictCursor
from sklearn.cluster import KMeans
import osmnx as ox
import numpy as np
import uvicorn

app = FastAPI(title="FAT Auto Placement Service")

# ==========================================
# Configuration
# ==========================================
DB_CONFIG = {
    "host": "10.101.110.235",
    "port": 5432,
    "database": "datamap",
    "user": "haifriday",
    "password": "PWDSuperKuat88!",
    "connect_timeout": 5,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FALLBACK_PATH = os.getenv(
    "PALEMBANG_BUILDINGS_CSV",
    os.path.join(BASE_DIR, "palembang_buildings.csv"),
)
ENABLE_ROAD_SNAP = os.getenv("ENABLE_ROAD_SNAP", "false").lower() in {"1", "true", "yes", "on"}
ROAD_SNAP_TIMEOUT = int(os.getenv("ROAD_SNAP_TIMEOUT", "15"))
_CSV_BUILDINGS_CACHE = None

# ==========================================
# Models
# ==========================================
class Bbox(BaseModel):
    north: float
    south: float
    east: float
    west: float

class GenerateFatRequest(BaseModel):
    bbox: Bbox
    capacity_per_fat: int = Field(default=16, gt=0)
    sample_area: dict[str, Any] | None = None

# ==========================================
# Helper Functions
# ==========================================
def get_db_connection():
    """Connect to PostgreSQL."""
    return psycopg2.connect(**DB_CONFIG)

def normalize_bbox(bbox: Bbox):
    """Return bbox limits as lat_min, lat_max, lon_min, lon_max."""
    return (
        min(bbox.south, bbox.north),
        max(bbox.south, bbox.north),
        min(bbox.west, bbox.east),
        max(bbox.west, bbox.east),
    )

def load_buildings_from_csv():
    """Load local CSV once so requests can still run when PostgreSQL is unavailable."""
    global _CSV_BUILDINGS_CACHE
    if _CSV_BUILDINGS_CACHE is not None:
        return _CSV_BUILDINGS_CACHE

    if not os.path.exists(CSV_FALLBACK_PATH):
        _CSV_BUILDINGS_CACHE = []
        return _CSV_BUILDINGS_CACHE

    buildings = []
    with open(CSV_FALLBACK_PATH, newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        for row_number, row in enumerate(reader, start=1):
            try:
                lat = float(row["latitude"])
                lon = float(row["longitude"])
            except (KeyError, TypeError, ValueError):
                continue

            raw_id = row.get("id") or row.get("osm_id") or row_number
            try:
                building_id = int(float(raw_id))
            except (TypeError, ValueError):
                building_id = row_number

            buildings.append({
                "id": building_id,
                "latitude": lat,
                "longitude": lon,
            })

    _CSV_BUILDINGS_CACHE = buildings
    print(f"Loaded {len(buildings)} buildings from CSV fallback.")
    return _CSV_BUILDINGS_CACHE

def fetch_buildings_from_csv(bbox: Bbox):
    """Fetch building coordinates from local CSV within the given bounding box."""
    lat_min, lat_max, lon_min, lon_max = normalize_bbox(bbox)
    buildings = load_buildings_from_csv()
    return [
        building
        for building in buildings
        if lat_min <= building["latitude"] <= lat_max
        and lon_min <= building["longitude"] <= lon_max
    ]

def fetch_buildings(bbox: Bbox):
    """Fetch building coordinates from PostgreSQL within the given bounding box."""
    query = """
        SELECT id, latitude, longitude
        FROM palembang_buildings
        WHERE latitude BETWEEN %s AND %s
          AND longitude BETWEEN %s AND %s
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                lat_min, lat_max, lon_min, lon_max = normalize_bbox(bbox)
                cur.execute(query, (lat_min, lat_max, lon_min, lon_max))
                rows = cur.fetchall()
                print(f"Fetched {len(rows)} buildings from PostgreSQL.")
                return rows
    except Exception as e:
        print(f"Warning: PostgreSQL unavailable, using CSV fallback: {e}")
        rows = fetch_buildings_from_csv(bbox)
        if rows:
            print(f"Fetched {len(rows)} buildings from CSV fallback.")
            return rows
        raise HTTPException(
            status_code=500,
            detail="Database connection error and CSV fallback returned no buildings.",
        )

def snap_centroids_to_roads(centroids, bbox: Bbox):
    """Snap all centroids using one OSMnx graph request; fallback to centroids if disabled/fails."""
    if not ENABLE_ROAD_SNAP:
        return [(float(lat), float(lon), False) for lat, lon in centroids]

    try:
        ox.settings.use_cache = True
        ox.settings.requests_timeout = ROAD_SNAP_TIMEOUT

        lat_min, lat_max, lon_min, lon_max = normalize_bbox(bbox)
        pad = 0.002
        graph_bbox = (lat_max + pad, lat_min - pad, lon_max + pad, lon_min - pad)
        graph = ox.graph_from_bbox(graph_bbox, network_type="drive")
        node_ids = ox.distance.nearest_nodes(
            graph,
            X=[float(lon) for _, lon in centroids],
            Y=[float(lat) for lat, _ in centroids],
        )
        return [
            (float(graph.nodes[node_id]["y"]), float(graph.nodes[node_id]["x"]), True)
            for node_id in node_ids
        ]
    except Exception as e:
        print(f"Warning: road snapping failed, using centroids: {e}")
        return [(float(lat), float(lon), False) for lat, lon in centroids]

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate Haversine distance in meters."""
    R = 6371000  # Earth radius in meters
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2.0) ** 2 + \
        math.cos(phi_1) * math.cos(phi_2) * \
        math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

# ==========================================
# Main Endpoint
# ==========================================
@app.post("/generate-fat")
def generate_fat(request: GenerateFatRequest):
    """
    Generates optimal FAT placements for the buildings in the bounding box.
    """
    print(f"Generating FAT placements for bbox: {request.bbox}")
    
    # 1. Get buildings
    buildings = fetch_buildings(request.bbox)
    if not buildings:
        raise HTTPException(status_code=404, detail="No buildings found in this bounding box.")
    
    print(f"Found {len(buildings)} buildings.")
    
    # Coordinates array for scikit-learn
    coords = np.array([[float(b['latitude']), float(b['longitude'])] for b in buildings])
    building_ids = [b['id'] for b in buildings]
    
    # 2. Setup Clustering (KMeans)
    n_clusters = math.ceil(len(buildings) / request.capacity_per_fat)
    if n_clusters == 0:
        n_clusters = 1
        
    print(f"Running clustering for {n_clusters} FATs...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(coords)
    
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    snapped_centroids = snap_centroids_to_roads(centroids, request.bbox)
    
    # 3. Build Result
    fats = []
    connections = []
    
    for cluster_id in range(n_clusters):
        # Find which buildings belong to this cluster
        cluster_b_indices = np.where(labels == cluster_id)[0]
        cluster_building_ids = [building_ids[i] for i in cluster_b_indices]
        
        road_lat, road_lon, snapped_to_road = snapped_centroids[cluster_id]
        
        fat_id = cluster_id + 1
        
        # FAT object
        fats.append({
            "id": fat_id,
            "latitude": road_lat,
            "longitude": road_lon,
            "connected_buildings": cluster_building_ids,
            "load": len(cluster_building_ids),
            "snapped_to_road": snapped_to_road,
        })
        
        # Connections object
        for idx in cluster_b_indices:
            b_lat = coords[idx][0]
            b_lon = coords[idx][1]
            dist = calculate_distance(b_lat, b_lon, road_lat, road_lon)
            
            connections.append({
                "building_id": building_ids[idx],
                "fat_id": fat_id,
                "distance": round(dist, 2) # in meters
            })
            
    return {
        "fats": fats,
        "connections": connections,
        "metadata": {
            "building_count": len(buildings),
            "fat_count": n_clusters,
            "capacity_per_fat": request.capacity_per_fat,
            "road_snap_enabled": ENABLE_ROAD_SNAP,
        },
    }

if __name__ == "__main__":
    import uvicorn
    # Jalankan server via python fat_api.py
    uvicorn.run("fat_api:app", host="0.0.0.0", port=8000, reload=True)
