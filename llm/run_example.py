import os
import re
import time
from metaqa_kgqa_pipeline import KnowledgeGraph, KGQAPipeline

# Configuration
METAQA_DIR = "./data"  # Replace with actual path to MetaQA dataset
KB_PATH = os.path.join(METAQA_DIR, "kb.txt")

# Example questions from different hop levels
EXAMPLE_QUESTIONS = [
    # 1-hop questions
    "Who directed Insomnia?",
    "What is the genre of Love Letter?",
    # 2-hop questions
    "What are the genres of the movies directed by Michael Bay?",
    "Which actors appear in movies directed by Martin Scorsese?",
    # 3-hop questions
    "What are the genres of the movies starring actors who appeared in The Prestige?",
    "Who are the directors of movies starring actors from Inception?",
]

# More specific examples from the dataset based on the error analysis
ADDITIONAL_QUESTIONS = [
    # Questions with correct entity names from the KG
    "Who directed Dream for an Insomniac?",
    "What is the genre of Love Letters?",
    # Questions with bidirectional traversal needs
    "Which movies does William Dieterle direct?",
    "Which movies does Christian Bale appear in?",
    # Questions with intersection needs
    "What actors appear in both The Prestige and Inception?",
    "What genres are common between The Matrix and Inception?",
]


def analyze_kg(kg):
    """Analyze and print key information about the knowledge graph."""
    relation_counts = {}
    incoming_entities = set()
    outgoing_entities = set()

    # Count relation types and entity types
    for entity, relations in kg.entity_to_relations.items():
        outgoing_entities.add(entity)
        for rel, target in relations:
            if rel not in relation_counts:
                relation_counts[rel] = 0
            relation_counts[rel] += 1

    for entity in kg.entity_to_incoming:
        incoming_entities.add(entity)

    # Print analysis
    print("\nKnowledge Graph Analysis:")
    print(f"Total entities: {len(kg.entities)}")
    print(f"Entities with outgoing relations: {len(outgoing_entities)}")
    print(f"Entities with incoming relations: {len(incoming_entities)}")
    print(f"Relation types: {len(relation_counts)}")

    print("\nRelation type counts:")
    for rel, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"- {rel}: {count}")

    # Sample some entities and their relations
    print("\nSample entities and their relations:")
    sample_entities = list(kg.entities)[:5]
    for entity in sample_entities:
        print(f"\nEntity: {entity}")

        outgoing = kg.get_outgoing_relations(entity)
        if outgoing:
            print("  Outgoing relations:")
            for rel, target in outgoing[:3]:
                print(f"  - {entity} --[{rel}]--> {target}")

        incoming = kg.get_incoming_relations(entity)
        if incoming:
            print("  Incoming relations:")
            for rel, source in incoming[:3]:
                print(f"  - {source} --[{rel}]--> {entity}")


def main():
    """Run example questions through the KGQA pipeline."""
    print("Loading knowledge graph...")
    kg = KnowledgeGraph(KB_PATH)

    # Analyze knowledge graph structure
    analyze_kg(kg)

    print("\nInitializing KGQA pipeline...")
    pipeline = KGQAPipeline(kg)

    # Test entity linking
    print("\nTesting entity linking:")
    test_entities = [
        "Inception",
        "The Matrix",
        "Christopher Nolan",
        "Martin Scorsese",
        "inceptions",
        "matrix",
        "nolan",
        "scorsese",
    ]
    for entity in test_entities:
        linked = kg.improve_entity_linking(entity)
        print(f"'{entity}' -> '{linked}'")

    # Combine all questions
    all_questions = EXAMPLE_QUESTIONS + ADDITIONAL_QUESTIONS

    print("\nProcessing example questions:")
    for i, question in enumerate(all_questions):
        print(f"\n{'=' * 80}")
        print(f"Example {i + 1}: {question}")

        # Extract topic entity (if in brackets)
        topic_entity_match = re.search(r"\[(.*?)\]", question)
        topic_entity = topic_entity_match.group(1) if topic_entity_match else None

        # Clean question by removing brackets if present
        clean_question = (
            re.sub(r"\[(.*?)\]", r"\1", question) if topic_entity_match else question
        )

        # Estimate hop count based on question structure
        hop_count = None  # Let the pipeline estimate it

        start_time = time.time()

        # Answer the question
        answers = pipeline.answer_question(clean_question, topic_entity, hop_count)

        end_time = time.time()

        print(f"\nTop answers: {answers[:10]}")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
