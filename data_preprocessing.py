import os
import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt


def load_kb(kb_path):
    """
    Load the knowledge graph from kb.txt file.

    Args:
        kb_path: Path to the kb.txt file

    Returns:
        Dictionary with entities, relations, and triples
    """
    entities = set()
    relations = set()
    triples = []

    print(f"Loading knowledge graph from {kb_path}...")
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue

            subject, relation, obj = parts
            triples.append((subject, relation, obj))
            entities.add(subject)
            entities.add(obj)
            relations.add(relation)

    kb = {"entities": list(entities), "relations": list(relations), "triples": triples}

    print(
        f"Loaded {len(triples)} triples with {len(entities)} entities and {len(relations)} relations"
    )
    return kb


def create_entity_dict(kb):
    """
    Create dictionaries mapping entities to their relations.

    Args:
        kb: Knowledge base dictionary

    Returns:
        Tuple of (entity_to_outgoing, entity_to_incoming) dictionaries
    """
    entity_to_outgoing = defaultdict(list)
    entity_to_incoming = defaultdict(list)

    for s, r, o in kb["triples"]:
        entity_to_outgoing[s].append((r, o))
        entity_to_incoming[o].append((r, s))

    return entity_to_outgoing, entity_to_incoming


def analyze_kb_statistics(kb):
    """
    Analyze statistics of the knowledge graph.

    Args:
        kb: Knowledge base dictionary

    Returns:
        Dictionary with statistics
    """
    entity_to_outgoing, entity_to_incoming = create_entity_dict(kb)

    # Count outgoing and incoming relations per entity
    outgoing_counts = [len(v) for v in entity_to_outgoing.values()]
    incoming_counts = [len(v) for v in entity_to_incoming.values()]

    # Count relation frequencies
    relation_counts = defaultdict(int)
    relation_direction = defaultdict(lambda: {"outgoing": 0, "incoming": 0})

    for s, r, o in kb["triples"]:
        relation_counts[r] += 1

        # Track which type of entities use this relation in which direction
        # This helps analyze the knowledge graph structure
        s_has_other_rels = len(entity_to_outgoing.get(s, [])) > 1
        o_has_other_rels = len(entity_to_outgoing.get(o, [])) > 1

        if s_has_other_rels:
            relation_direction[r]["outgoing"] += 1
        if o_has_other_rels:
            relation_direction[r]["incoming"] += 1

    # Identify isolated entities (no connections)
    connected_entities = set(entity_to_outgoing.keys()) | set(entity_to_incoming.keys())
    isolated_entities = set(kb["entities"]) - connected_entities

    # Identify potential "entity types" based on relation patterns
    entity_types = defaultdict(int)
    for entity, relations in entity_to_outgoing.items():
        rel_types = sorted([r for r, _ in relations])
        type_signature = ",".join(rel_types)
        entity_types[type_signature] += 1

    # Sample triples for each relation
    relation_examples = {}
    for r in relation_counts.keys():
        examples = [t for t in kb["triples"] if t[1] == r][:5]
        relation_examples[r] = examples

    statistics = {
        "entity_count": len(kb["entities"]),
        "relation_count": len(kb["relations"]),
        "triple_count": len(kb["triples"]),
        "avg_outgoing_per_entity": sum(outgoing_counts) / len(kb["entities"])
        if kb["entities"]
        else 0,
        "avg_incoming_per_entity": sum(incoming_counts) / len(kb["entities"])
        if kb["entities"]
        else 0,
        "max_outgoing": max(outgoing_counts) if outgoing_counts else 0,
        "max_incoming": max(incoming_counts) if incoming_counts else 0,
        "isolated_entities_count": len(isolated_entities),
        "most_common_relations": sorted(
            relation_counts.items(), key=lambda x: x[1], reverse=True
        )[:10],
        "relation_direction": relation_direction,
        "common_entity_types": sorted(
            entity_types.items(), key=lambda x: x[1], reverse=True
        )[:10],
        "relation_examples": relation_examples,
    }

    return statistics


def visualize_kb_subset(kb, max_entities=50, max_relations=None):
    """
    Create a visualization of a subset of the knowledge graph.

    Args:
        kb: Knowledge base dictionary
        max_entities: Maximum number of entities to include
        max_relations: Optional list of relations to include

    Returns:
        NetworkX graph
    """
    G = nx.DiGraph()

    # Select a subset of entities
    entities_subset = kb["entities"][:max_entities]

    # Add nodes
    for entity in entities_subset:
        G.add_node(entity)

    # Add edges for selected entities and relations
    for s, r, o in kb["triples"]:
        if s in entities_subset and o in entities_subset:
            if max_relations is None or r in max_relations:
                G.add_edge(s, o, label=r)

    # Visualize
    plt.figure(figsize=(15, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=500,
        node_color="lightblue",
        font_size=8,
        arrows=True,
    )

    # Draw edge labels
    edge_labels = {(s, o): r for s, o, r in G.edges(data="label")}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)

    plt.title(
        f"Knowledge Graph Subset ({len(G.nodes())} entities, {len(G.edges())} relations)"
    )
    plt.tight_layout()
    plt.savefig("kg_visualization.png", dpi=300)
    plt.close()

    return G


def analyze_question_types(dataset_dir, hop=2):
    """
    Analyze types of questions in the dataset.

    Args:
        dataset_dir: Directory containing the dataset
        hop: Number of hops (1, 2, or 3)

    Returns:
        Dictionary with question type statistics
    """
    question_types = defaultdict(int)
    question_examples = defaultdict(list)

    # Load the question type file
    qtype_path = os.path.join(dataset_dir, f"{hop}-hop", "qa_train_qtype.txt")

    if not os.path.exists(qtype_path):
        print(f"Question type file not found: {qtype_path}")
        return {}

    # Load the corresponding questions
    qa_path = os.path.join(dataset_dir, f"{hop}-hop", "vanilla", "qa_train.txt")

    if not os.path.exists(qa_path):
        print(f"QA file not found: {qa_path}")
        return {}

    # Load both files
    with (
        open(qtype_path, "r", encoding="utf-8") as qtype_file,
        open(qa_path, "r", encoding="utf-8") as qa_file,
    ):
        for qtype_line, qa_line in zip(qtype_file, qa_file):
            qtype = qtype_line.strip()
            question, answer = qa_line.strip().split("\t")

            question_types[qtype] += 1

            # Store a few examples of each type
            if len(question_examples[qtype]) < 5:
                question_examples[qtype].append(
                    {"question": question, "answer": answer}
                )

    # Create statistics
    stats = {"type_counts": dict(question_types), "examples": dict(question_examples)}

    return stats


def preprocess_dataset(dataset_dir, hop=2, output_dir=None):
    """
    Preprocess the dataset for easier use.

    Args:
        dataset_dir: Directory containing the dataset
        hop: Number of hops (1, 2, or 3)
        output_dir: Directory to save preprocessed data

    Returns:
        Path to the preprocessed data
    """
    if output_dir is None:
        output_dir = os.path.join(dataset_dir, "preprocessed")

    os.makedirs(output_dir, exist_ok=True)

    # Load knowledge base
    kb_path = os.path.join(dataset_dir, "kb.txt")
    kb = load_kb(kb_path)

    # Save KB in JSON format
    kb_json_path = os.path.join(output_dir, "kb.json")
    with open(kb_json_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, indent=2)

    # Create entity dictionaries
    entity_to_outgoing, entity_to_incoming = create_entity_dict(kb)

    # Save entity dictionaries
    entity_dicts_path = os.path.join(output_dir, "entity_dicts.json")
    with open(entity_dicts_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "entity_to_outgoing": {k: v for k, v in entity_to_outgoing.items()},
                "entity_to_incoming": {k: v for k, v in entity_to_incoming.items()},
            },
            f,
            indent=2,
        )

    # Analyze KB statistics
    kb_stats = analyze_kb_statistics(kb)
    kb_stats_path = os.path.join(output_dir, "kb_stats.json")
    with open(kb_stats_path, "w", encoding="utf-8") as f:
        json.dump(kb_stats, f, indent=2)

    # Analyze question types
    qtype_stats = analyze_question_types(dataset_dir, hop)
    qtype_stats_path = os.path.join(output_dir, f"question_types_hop{hop}.json")
    with open(qtype_stats_path, "w", encoding="utf-8") as f:
        json.dump(qtype_stats, f, indent=2)

    # Visualize KB subset
    visualize_kb_subset(kb)

    print(f"Preprocessing complete. Files saved to {output_dir}")
    return output_dir


if __name__ == "__main__":
    # Example usage
    dataset_dir = "./data"  # Replace with actual path
    preprocess_dataset(dataset_dir, hop=2)
