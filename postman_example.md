# Postman API Collection: FAT Placement Service

Berikut adalah contoh cara memanggil API yang telah dibuat menggunakan Postman.

## Endpoint

- **URL:** `http://localhost:8000/generate-fat`
- **Method:** `POST`
- **Headers:** 
  - `Content-Type: application/json`

## Request Body (Raw JSON)

Sesuaikan `bbox` dengan koordinat kotak (bounding box) area yang ingin kamu proses. Jika kamu masih memakai data Palembang seperti script sebelumnya, kamu bisa memakai bbox di area Palembang.

```json
{
  "bbox": {
    "north": -2.85,
    "south": -3.10,
    "east": 104.85,
    "west": 104.60
  },
  "capacity_per_fat": 16
}
```

> **Tips:** Bounding box di atas memakai koordinat seluruh kota Palembang. Untuk testing lokal dan meminimalisir waktu query/snapping jalan, sebaiknya masukkan area bbox yang lebih kecil (contoh: 1 kelurahan atau beberapa RT).

## Cara Menjalankan

1. Buka terminal, pastikan sudah di dalam folder `Data Map`.
2. Aktifkan virtual environment (jika ada).
3. Install dependencies jika belum:
   ```bash
   pip install fastapi uvicorn psycopg2-binary scikit-learn osmnx numpy pydantic
   ```
4. Jalankan server:
   ```bash
   python fat_api.py
   ```
5. Buka **Postman**, buat request baru dengan method **POST** ke `http://localhost:8000/generate-fat`.
6. Taruh payload JSON di tab **Body** -> pilih **raw** -> format **JSON**.
7. Klik **Send**.

## Contoh Response

```json
{
    "fats": [
        {
            "id": 1,
            "latitude": -3.0982701,
            "longitude": 104.6542911,
            "connected_buildings": [
                154,
                155,
                156
            ],
            "load": 3
        }
    ],
    "connections": [
        {
            "building_id": 154,
            "fat_id": 1,
            "distance": 23.51
        },
        {
            "building_id": 155,
            "fat_id": 1,
            "distance": 12.33
        },
        {
            "building_id": 156,
            "fat_id": 1,
            "distance": 45.10
        }
    ]
}
```

## Note / Edge Cases
- **Snapping Jalan:** Jika request memakan waktu lama, itu karena `osmnx` sedang mendownload data jaringan jalan di sekitar titik centroid via internet.
- **Kapasitas:** Default 1 FAT menampung 16 bangunan. Jika ada cluster yang jumlah bangunannya kurang atau rasionya terbagi, K-Means akan mendistribusikannya secara otomatis berdasarkan jarak terdekat ke Centroid.
