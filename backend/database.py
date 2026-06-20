import sqlite3
import hashlib
from datetime import datetime

DB_PATH = "backend/deepretina.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS scan_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            image_hash TEXT UNIQUE NOT NULL,
            modality TEXT NOT NULL,
            predicted_stage TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def generate_file_hash(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()

def check_cached_hash(image_hash):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT predicted_stage, confidence FROM scan_records WHERE image_hash = ?", (image_hash,))
    row = cursor.fetchone()
    conn.close()
    return row  # Returns None or cached prediction details

def save_scan_record(patient_id, image_hash, modality, stage, confidence):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute('''
            INSERT INTO scan_records (patient_id, image_hash, modality, predicted_stage, confidence, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (patient_id, image_hash, modality, stage, confidence, timestamp))
        conn.commit()
    except sqlite3.IntegrityError:
        pass  # Hash duplicate guard
    finally:
        conn.close()

def fetch_all_history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT patient_id, modality, predicted_stage, confidence, timestamp FROM scan_records ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows

init_db()  # Run auto-initialization immediately