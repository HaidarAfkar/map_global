"""
Palembang Building Data Extractor
=================================
Extracts building lat/long data from OpenStreetMap (Overpass API)
for the Palembang area, exports to CSV, and stores in PostgreSQL.
"""

import requests
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import time
import os
import sys

# ============================================================
# Configuration
# ============================================================

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# PostgreSQL connection
DB_CONFIG = {
    "host": "10.101.110.235",
    "port": 5432,
    "database": "datamap",
    "user": "haifriday",
    "password": "PWDSuperKuat88!",
}

# Output CSV path (same directory as this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_OUTPUT = os.path.join(SCRIPT_DIR, "palembang_buildings.csv")

# Table name in PostgreSQL
TABLE_NAME = "palembang_buildings"


# ============================================================
# 1. Fetch building data from Overpass API (tile-based)
# ============================================================

# Palembang bounding box
BBOX_SOUTH = -3.10
BBOX_WEST = 104.60
BBOX_NORTH = -2.85
BBOX_EAST = 104.85

# Grid size (5x5 = 25 tiles, each ~0.05° × 0.05°)
GRID_ROWS = 5
GRID_COLS = 5


def generate_tiles():
    """Split the Palembang bounding box into smaller tiles."""
    lat_step = (BBOX_NORTH - BBOX_SOUTH) / GRID_ROWS
    lon_step = (BBOX_EAST - BBOX_WEST) / GRID_COLS

    tiles = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            s = BBOX_SOUTH + r * lat_step
            n = s + lat_step
            w = BBOX_WEST + c * lon_step
            e = w + lon_step
            tiles.append((s, w, n, e))
    return tiles


def fetch_tile(south, west, north, east, tile_num, total_tiles):
    """Fetch building data for a single tile with retry logic."""
    query = f"""
    [out:json][timeout:180][maxsize:268435456];
    (
      way["building"]({south},{west},{north},{east});
      relation["building"]({south},{west},{north},{east});
    );
    out center tags;
    """

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()
            elements = data.get("elements", [])
            return elements
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            if attempt < max_retries:
                wait = 10 * attempt
                print(f"        Tile {tile_num}/{total_tiles} gagal (attempt {attempt}): {e}")
                print(f"        Menunggu {wait}s sebelum retry...")
                time.sleep(wait)
            else:
                print(f"        WARNING: Tile {tile_num}/{total_tiles} gagal setelah {max_retries} percobaan: {e}")
                return []
        except requests.exceptions.RequestException as e:
            print(f"        WARNING: Tile {tile_num}/{total_tiles} error: {e}")
            return []


def fetch_buildings():
    """Fetch building data from Overpass API using tiled queries."""
    print("[1/4] Mengambil data bangunan dari Overpass API (tiled query)...")

    tiles = generate_tiles()
    total_tiles = len(tiles)
    print(f"      Membagi area Palembang menjadi {total_tiles} tile...")

    all_elements = {}  # keyed by OSM ID to deduplicate
    for idx, (s, w, n, e) in enumerate(tiles, 1):
        print(f"      Tile {idx}/{total_tiles}: ({s:.3f},{w:.3f}) -> ({n:.3f},{e:.3f})")
        elements = fetch_tile(s, w, n, e, idx, total_tiles)
        for el in elements:
            osm_id = el.get("id")
            if osm_id not in all_elements:
                all_elements[osm_id] = el
        print(f"        -> {len(elements)} elemen (total unik: {len(all_elements)})")

        # Rate-limit: wait between tiles to be polite to the API
        if idx < total_tiles:
            time.sleep(2)

    elements_list = list(all_elements.values())
    print(f"      Total: {len(elements_list)} elemen bangunan unik dari OSM.")
    return elements_list


# ============================================================
# 2. Parse elements into structured records
# ============================================================

def parse_buildings(elements):
    """Parse raw Overpass elements into a list of dicts."""
    print("[2/4] Memproses data bangunan...")

    records = []
    for el in elements:
        # Get center coordinates
        if el["type"] == "way":
            lat = el.get("center", {}).get("lat")
            lon = el.get("center", {}).get("lon")
        elif el["type"] == "relation":
            lat = el.get("center", {}).get("lat")
            lon = el.get("center", {}).get("lon")
        else:
            # node (unlikely for buildings, but handle it)
            lat = el.get("lat")
            lon = el.get("lon")

        if lat is None or lon is None:
            continue

        tags = el.get("tags", {})

        records.append({
            "osm_id": el.get("id"),
            "name": tags.get("name", ""),
            "building_type": tags.get("building", ""),
            "latitude": lat,
            "longitude": lon,
            "street": tags.get("addr:street", ""),
            "housenumber": tags.get("addr:housenumber", ""),
            "city": tags.get("addr:city", "Palembang"),
            "postcode": tags.get("addr:postcode", ""),
        })

    print(f"      Berhasil memproses {len(records)} bangunan dengan koordinat valid.")
    return records


# ============================================================
# 3. Export to CSV
# ============================================================

def export_to_csv(records):
    """Export records to a CSV file."""
    print(f"[3/4] Menyimpan data ke CSV: {CSV_OUTPUT}")

    df = pd.DataFrame(records)
    df.to_csv(CSV_OUTPUT, index=False, encoding="utf-8-sig")

    print(f"      CSV berhasil disimpan! ({len(df)} baris)")
    return df


# ============================================================
# 4. Store in PostgreSQL
# ============================================================

def create_table(cursor):
    """Create the buildings table if it doesn't exist."""
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
        id          SERIAL PRIMARY KEY,
        osm_id      BIGINT,
        name        VARCHAR(500),
        building_type VARCHAR(255),
        latitude    DOUBLE PRECISION,
        longitude   DOUBLE PRECISION,
        street      VARCHAR(500),
        housenumber VARCHAR(50),
        city        VARCHAR(255) DEFAULT 'Palembang',
        postcode    VARCHAR(20),
        created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    cursor.execute(ddl)


def insert_records(cursor, records):
    """Bulk insert records into PostgreSQL using execute_values."""
    if not records:
        print("      Tidak ada data untuk dimasukkan.")
        return

    columns = [
        "osm_id", "name", "building_type",
        "latitude", "longitude",
        "street", "housenumber", "city", "postcode",
    ]
    values = [
        tuple(rec[col] for col in columns)
        for rec in records
    ]

    sql = f"""
        INSERT INTO {TABLE_NAME}
            ({', '.join(columns)})
        VALUES %s
    """

    # Insert in chunks of 5000
    chunk_size = 5000
    total = len(values)
    for i in range(0, total, chunk_size):
        chunk = values[i : i + chunk_size]
        execute_values(cursor, sql, chunk)
        inserted = min(i + chunk_size, total)
        print(f"      Inserted {inserted}/{total} records...")


def save_to_postgresql(records):
    """Connect to PostgreSQL, create table, and insert records."""
    print(f"[4/4] Menyimpan data ke PostgreSQL ({DB_CONFIG['host']})...")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        cursor = conn.cursor()

        # Create table
        create_table(cursor)
        conn.commit()

        # Clear existing data to avoid duplicates on re-run
        cursor.execute(f"DELETE FROM {TABLE_NAME};")
        print(f"      Tabel {TABLE_NAME} dibersihkan (data lama dihapus).")

        # Insert new data
        insert_records(cursor, records)
        conn.commit()

        # Verify
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};")
        count = cursor.fetchone()[0]
        print(f"      Berhasil! {count} baris tersimpan di tabel '{TABLE_NAME}'.")

        cursor.close()
        conn.close()

    except psycopg2.OperationalError as e:
        print(f"ERROR: Gagal koneksi ke PostgreSQL: {e}")
        print("       Pastikan server PostgreSQL aktif dan dapat diakses.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Gagal menyimpan ke database: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        sys.exit(1)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("  Palembang Building Data Extractor")
    print("=" * 60)
    print()

    start_time = time.time()

    # Step 1: Fetch from Overpass API
    elements = fetch_buildings()

    # Step 2: Parse into records
    records = parse_buildings(elements)

    if not records:
        print("Tidak ada data bangunan yang ditemukan. Keluar.")
        sys.exit(0)

    # Step 3: Export to CSV
    export_to_csv(records)

    # Step 4: Save to PostgreSQL
    save_to_postgresql(records)

    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print(f"  Selesai! Waktu total: {elapsed:.1f} detik")
    print(f"  CSV: {CSV_OUTPUT}")
    print(f"  DB:  {DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    print(f"       Tabel: {TABLE_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()