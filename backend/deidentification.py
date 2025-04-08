from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def deidentify_text(text: str):
    results = analyzer.analyze(text=text, language='en')

    operators = {}
    entity_map = {}
    tag_counts = {}

    for entity in results:
        ent_type = entity.entity_type.upper()
        tag_counts[ent_type] = tag_counts.get(ent_type, 0) + 1
        tag = f"{ent_type}_{str(tag_counts[ent_type]).zfill(3)}"
        operators[ent_type] = OperatorConfig("replace", {"new_value": tag})
        entity_map[tag] = ent_type

    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators
    )

    return anonymized.text, entity_map
