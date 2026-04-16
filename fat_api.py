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
from shapely.geometry import LineString
from shapely.wkb import loads as load_wkb

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
ENABLE_OBSTACLE_AWARE = os.getenv("ENABLE_OBSTACLE_AWARE", "true").lower() in {"1", "true", "yes", "on"}
OBSTACLE_TABLE = os.getenv("OSM_OBSTACLE_TABLE", "osm_palembang_obstacles")
OBSTACLE_PENALTIES = {
    "water": float(os.getenv("OBSTACLE_WATER_PENALTY_M", "1000")),
    "rail": float(os.getenv("OBSTACLE_RAIL_PENALTY_M", "500")),
    "major_road": float(os.getenv("OBSTACLE_MAJOR_ROAD_PENALTY_M", "250")),
    "other": float(os.getenv("OBSTACLE_OTHER_PENALTY_M", "100")),
}
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

def fetch_obstacles(bbox: Bbox):
    """Fetch local OSM obstacles from PostGIS for the requested bbox."""
    if not ENABLE_OBSTACLE_AWARE:
        return []

    lat_min, lat_max, lon_min, lon_max = normalize_bbox(bbox)
    query = f"""
        SELECT
            id,
            obstacle_type,
            COALESCE(name, '') AS name,
            ST_AsBinary(geom) AS geom_wkb
        FROM {OBSTACLE_TABLE}
        WHERE geom && ST_MakeEnvelope(%s, %s, %s, %s, 4326)
          AND ST_Intersects(geom, ST_MakeEnvelope(%s, %s, %s, %s, 4326))
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                params = (lon_min, lat_min, lon_max, lat_max) * 2
                cur.execute(query, params)
                rows = cur.fetchall()
    except Exception as e:
        print(f"Warning: obstacle query skipped: {e}")
        return []

    obstacles = []
    for row in rows:
        try:
            geom = load_wkb(bytes(row["geom_wkb"]))
        except Exception:
            continue
        if geom.is_empty:
            continue
        obstacle_type = row["obstacle_type"] or "other"
        obstacles.append({
            "id": row["id"],
            "type": obstacle_type,
            "name": row["name"],
            "geometry": geom,
            "penalty_m": OBSTACLE_PENALTIES.get(obstacle_type, OBSTACLE_PENALTIES["other"]),
        })

    print(f"Fetched {len(obstacles)} local OSM obstacles.")
    return obstacles

def snap_centroids_to_roads(centroids, bbox: Bbox):
    """Snap all centroids using one OSMnx graph request; fallback to centroids if disabled/fails."""
    if not ENABLE_ROAD_SNAP:
        return [(float(lat), float(lon), False) for lat, lon in centroids]

    try:
        ox.settings.use_cache = True
        ox.settings.requests_timeout = ROAD_SNAP_TIMEOUT

        lat_min, lat_max, lon_min, lon_max = normalize_bbox(bbox)
        pad = 0.002
        graph_bbox = (lon_min - pad, lat_min - pad, lon_max + pad, lat_max + pad)
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

def evaluate_connection(building_lat, building_lon, fat_lat, fat_lon, obstacles):
    """Calculate straight distance plus obstacle crossing penalties."""
    straight_distance = calculate_distance(building_lat, building_lon, fat_lat, fat_lon)
    if not obstacles:
        return {
            "straight_distance_m": straight_distance,
            "obstacle_penalty_m": 0.0,
            "adjusted_distance_m": straight_distance,
            "crossed_obstacles": [],
        }

    line = LineString([(float(building_lon), float(building_lat)), (float(fat_lon), float(fat_lat))])
    crossed = []
    penalty = 0.0
    for obstacle in obstacles:
        try:
            if not line.crosses(obstacle["geometry"]) and not line.intersects(obstacle["geometry"]):
                continue
        except Exception:
            continue

        penalty += obstacle["penalty_m"]
        crossed.append({
            "id": obstacle["id"],
            "type": obstacle["type"],
            "name": obstacle["name"],
            "penalty_m": round(obstacle["penalty_m"], 2),
        })

    return {
        "straight_distance_m": straight_distance,
        "obstacle_penalty_m": penalty,
        "adjusted_distance_m": straight_distance + penalty,
        "crossed_obstacles": crossed,
    }

def assign_buildings_to_fats(coords, building_ids, fat_points, capacity_per_fat, obstacles):
    """Assign buildings to FATs using distance plus obstacle penalties."""
    if not obstacles:
        assignments = {}
        for fat_index in range(len(fat_points)):
            assignments[fat_index] = []
        return assignments, {}

    assignments = {fat_index: [] for fat_index in range(len(fat_points))}
    metrics_by_pair = {}

    for building_index, (building_lat, building_lon) in enumerate(coords):
        candidates = []
        for fat_index, (fat_lat, fat_lon, _) in enumerate(fat_points):
            metrics = evaluate_connection(
                building_lat,
                building_lon,
                fat_lat,
                fat_lon,
                obstacles,
            )
            metrics_by_pair[(building_index, fat_index)] = metrics
            candidates.append((metrics["adjusted_distance_m"], fat_index))

        candidates.sort(key=lambda item: item[0])
        selected_fat = candidates[0][1]
        for _, fat_index in candidates:
            if len(assignments[fat_index]) < capacity_per_fat:
                selected_fat = fat_index
                break

        assignments[selected_fat].append(building_index)

    return assignments, metrics_by_pair

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
    obstacles = fetch_obstacles(request.bbox)
    
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
    assignments, metrics_by_pair = assign_buildings_to_fats(
        coords,
        building_ids,
        snapped_centroids,
        request.capacity_per_fat,
        obstacles,
    )
    
    # 3. Build Result
    fats = []
    connections = []
    
    for cluster_id in range(n_clusters):
        if obstacles:
            cluster_b_indices = np.array(assignments[cluster_id], dtype=int)
        else:
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
            metrics = metrics_by_pair.get((idx, cluster_id))
            if metrics is None:
                metrics = evaluate_connection(b_lat, b_lon, road_lat, road_lon, obstacles)
            
            connections.append({
                "building_id": building_ids[idx],
                "fat_id": fat_id,
                "building_latitude": float(b_lat),
                "building_longitude": float(b_lon),
                "fat_latitude": float(road_lat),
                "fat_longitude": float(road_lon),
                "distance": round(metrics["adjusted_distance_m"], 2),
                "distance_m": round(metrics["adjusted_distance_m"], 2),
                "straight_distance_m": round(metrics["straight_distance_m"], 2),
                "obstacle_penalty_m": round(metrics["obstacle_penalty_m"], 2),
                "adjusted_distance_m": round(metrics["adjusted_distance_m"], 2),
                "network_distance_m": None,
                "crossed_obstacles": metrics["crossed_obstacles"],
            })
            
    return {
        "fats": fats,
        "connections": connections,
        "metadata": {
            "building_count": len(buildings),
            "fat_count": n_clusters,
            "capacity_per_fat": request.capacity_per_fat,
            "road_snap_enabled": ENABLE_ROAD_SNAP,
            "obstacle_aware_enabled": ENABLE_OBSTACLE_AWARE,
            "obstacle_count": len(obstacles),
            "routing_mode": "straight_line_with_obstacle_penalty",
            "levels_applied": {
                "level_1_crossing_penalty": bool(obstacles),
                "level_2_obstacle_aware_reassignment": bool(obstacles),
                "level_3_local_road_routing": False,
            },
        },
    }

if __name__ == "__main__":
    import uvicorn
    # Jalankan server via python fat_api.py
    uvicorn.run("fat_api:app", host="0.0.0.0", port=8000, reload=True)
