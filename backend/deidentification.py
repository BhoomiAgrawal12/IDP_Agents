import sqlite3
import json
import uuid
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

# Initialize Presidio
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def deidentify_text(text: str):
    results = analyzer.analyze(text=text, language='en')

    operators = {}
    entity_map = {}

    for entity in results:
        ent_type = entity.entity_type.upper()
        uid = f"{ent_type}_{uuid.uuid4().hex[:8]}"  # Short UUID
        operators[ent_type] = OperatorConfig("replace", {"new_value": uid})
        entity_map[uid] = ent_type

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators
    )

    return anonymized.text, entity_map


conn = sqlite3.connect("pii_data.db")
cursor = conn.cursor()

original_field = 'original_text'
identifier_field = original_field + '_identifier'
map_field = original_field + '_identifier_map'

# Add new columns if they don't exist
cursor.execute("PRAGMA table_info(pii_records)")
existing_cols = [col[1] for col in cursor.fetchall()]

if identifier_field not in existing_cols:
    cursor.execute(f"ALTER TABLE pii_records ADD COLUMN {identifier_field} TEXT")
if map_field not in existing_cols:
    cursor.execute(f"ALTER TABLE pii_records ADD COLUMN {map_field} TEXT")

# Process rows with missing deidentified data
cursor.execute(f"SELECT id, {original_field} FROM pii_records WHERE {identifier_field} IS NULL")
for row in cursor.fetchall():
    record_id, original_text = row
    deid_text, entity_map = deidentify_text(original_text)
    cursor.execute(
        f"UPDATE pii_records SET {identifier_field} = ?, {map_field} = ? WHERE id = ?",
        (deid_text, json.dumps(entity_map), record_id)
    )

conn.commit()
conn.close()
