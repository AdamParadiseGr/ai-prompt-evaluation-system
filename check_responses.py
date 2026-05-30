import sqlite3, json

conn = sqlite3.connect("experiments/results.db")
rows = conn.execute("""
    SELECT test_case_id, raw_response, dimensions
    FROM results
    WHERE experiment_id = '3347e312'
    AND prompt_name = 'v3_structured_output'
    AND test_case_id IN ('tc_002', 'tc_008', 'tc_009')
    ORDER BY test_case_id
""").fetchall()

for row in rows:
    dims = json.loads(row[2])
    topic = next((d for d in dims if d['name'] == 'topic_coverage'), None)
    print(f"\n=== {row[0]} | topic={topic['score'] if topic else '?'} ===")
    print(row[1][:600])
