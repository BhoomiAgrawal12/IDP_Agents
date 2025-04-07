from flask import Flask, request, jsonify
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

app = Flask(__name__)
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

@app.route("/anonymize", methods=["POST"])
def anonymize():
    data = request.json
    text = data["text"]
    policy = data.get("policy", "PERSON")
    results = analyzer.analyze(text=text, entities=policy.split(","), language="en")
    anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
    return jsonify({"text": anonymized.text, "report": {"anonymizedFields": [r.entity_type for r in results], "totalAnonymized": len(results)}})

if __name__ == "__main__":
    app.run(port=5001)