import os
import math
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
}

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
    capacity_per_fat: int = 16

# ==========================================
# Helper Functions
# ==========================================
def get_db_connection():
    """Connect to PostgreSQL."""
    return psycopg2.connect(**DB_CONFIG)

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
                # Ensure correct ordering (south to north, west to east)
                lat_min = min(bbox.south, bbox.north)
                lat_max = max(bbox.south, bbox.north)
                lon_min = min(bbox.west, bbox.east)
                lon_max = max(bbox.west, bbox.east)

                cur.execute(query, (lat_min, lat_max, lon_min, lon_max))
                return cur.fetchall()
    except Exception as e:
        print(f"Error fetching buildings: {e}")
        raise HTTPException(status_code=500, detail="Database connection error.")

def snap_to_nearest_road(lat, lon):
    """
    Snap the given latitude and longitude to the nearest road network node using OSMnx.
    """
    try:
        # Fetch a small network around the point (e.g. 500m radius)
        G = ox.graph_from_point((lat, lon), dist=500, network_type='drive')
        
        # Find nearest node
        nearest_node = ox.distance.nearest_nodes(G, X=lon, Y=lat)
        node_data = G.nodes[nearest_node]
        
        return node_data['y'], node_data['x']  # lat, lon
    except Exception as e:
        # Fallback to the original centroid if no road found or osmnx fails
        print(f"Warning: Failed to snap to road (fallback to centroid): {e}")
        return lat, lon

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
    coords = np.array([[b['latitude'], b['longitude']] for b in buildings])
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
    
    # 3. Build Result
    fats = []
    connections = []
    
    for cluster_id in range(n_clusters):
        # Find which buildings belong to this cluster
        cluster_b_indices = np.where(labels == cluster_id)[0]
        cluster_building_ids = [building_ids[i] for i in cluster_b_indices]
        
        # Original Centroid
        centroid_lat, centroid_lon = centroids[cluster_id]
        
        # Snap Centroid to road
        road_lat, road_lon = snap_to_nearest_road(centroid_lat, centroid_lon)
        
        fat_id = cluster_id + 1
        
        # FAT object
        fats.append({
            "id": fat_id,
            "latitude": road_lat,
            "longitude": road_lon,
            "connected_buildings": cluster_building_ids,
            "load": len(cluster_building_ids)
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
        "connections": connections
    }

if __name__ == "__main__":
    import uvicorn
    # Jalankan server via python fat_api.py
    uvicorn.run("fat_api:app", host="0.0.0.0", port=8000, reload=True)
