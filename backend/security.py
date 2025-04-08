import sqlite3
import json
import uuid
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from cryptography.fernet import Fernet

# 🔑 Generate this once and store securely
# key = Fernet.generate_key()
key = b'your-32-byte-base64-key-here'  # Replace this with your actual key
fernet = Fernet(key)

def encrypt_text(text: str) -> str:
    return fernet.encrypt(text.encode()).decode()

def deidentify_and_encrypt_to_db(db_path: str, table: str, field: str):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    identifier_field = field + '_identifier'
    map_field = field + '_identifier_map'

    cursor.execute(f"PRAGMA table_info({table})")
    cols = [col[1] for col in cursor.fetchall()]
    if identifier_field not in cols:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {identifier_field} TEXT")
    if map_field not in cols:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {map_field} TEXT")

    cursor.execute(f"SELECT id, {field} FROM {table} WHERE {identifier_field} IS NULL")
    rows = cursor.fetchall()

    for row in rows:
        record_id, original_text = row
        results = analyzer.analyze(text=original_text, language='en')
        operators = {}
        entity_map = {}

        for entity in results:
            ent_type = entity.entity_type.upper()
            uid = f"{ent_type}_{uuid.uuid4().hex[:8]}"
            operators[ent_type] = OperatorConfig("replace", {"new_value": uid})
            entity_map[uid] = ent_type

        anonymized = anonymizer.anonymize(
            text=original_text,
            analyzer_results=results,
            operators=operators
        )

        encrypted_text = encrypt_text(anonymized.text)
        encrypted_map = encrypt_text(json.dumps(entity_map))

        cursor.execute(
            f"UPDATE {table} SET {identifier_field} = ?, {map_field} = ? WHERE id = ?",
            (encrypted_text, encrypted_map, record_id)
        )

    conn.commit()
    conn.close()
