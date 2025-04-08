import sqlite3

conn = sqlite3.connect("pii_data.db")
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS pii_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_text TEXT
)
''')

cursor.executemany('''
INSERT INTO pii_records (original_text) VALUES (?)
''', [
    ("John Doe lives at 123 Main St and his phone number is 555-123-4567.",),
    ("Contact Jane at jane.doe@example.com or call her at (555) 987-6543.",),
    ("My SSN is 123-45-6789 and I live in New York.",),
    ("Michael's credit card number is 4111 1111 1111 1111.",)
])

conn.commit()
conn.close()
