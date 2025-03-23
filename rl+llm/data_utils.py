"""
Data loading and processing utilities for handling the MetaQA dataset
"""

import re
import os
import json
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import config
from difflib import SequenceMatcher


def load_knowledge_graph(kb_path):
    """
    Load knowledge graph data

    Parameters:
        kb_path (str): Knowledge graph file path

    Returns:
        tuple: (kg_dict, entities, relations, kg_stats, entity_to_idx, relation_to_idx)
    """
    print(f"Loading knowledge graph: {kb_path}")

    kg_dict = defaultdict(list)
    reverse_kg_dict = defaultdict(list)  # Reverse index
    entities = set()
    relations = set()
    kg_stats = {
        "triple_count": 0,
        "relation_counts": defaultdict(int),
        "entity_out_degrees": defaultdict(int),
        "entity_in_degrees": defaultdict(int),
    }

    # Read knowledge graph
    with open(kb_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading knowledge graph"):
            parts = line.strip().split("|")
            if len(parts) == 3:
                subj, rel, obj = parts
                kg_stats["triple_count"] += 1

                # Normalize entity names (remove surrounding spaces while preserving original case)
                subj = subj.strip()
                obj = obj.strip()
                rel = rel.strip()

                # Add entities and relations
                entities.add(subj)
                entities.add(obj)
                relations.add(rel)

                # Update statistics
                kg_stats["relation_counts"][rel] += 1
                kg_stats["entity_out_degrees"][subj] += 1
                kg_stats["entity_in_degrees"][obj] += 1

                # Add to knowledge graph dictionary
                kg_dict[subj].append((rel, obj))

                # Add reverse index
                inverse_rel = f"{rel}_inverse"
                relations.add(inverse_rel)
                reverse_kg_dict[obj].append((inverse_rel, subj))

    # Merge forward and reverse indices
    for entity, edges in reverse_kg_dict.items():
        if entity in kg_dict:
            kg_dict[entity].extend(edges)
        else:
            kg_dict[entity] = edges

    # Convert to list to ensure consistent ordering
    entities = sorted(list(entities))
    relations = sorted(list(relations))

    # Create index mappings
    entity_to_idx = {entity: idx for idx, entity in enumerate(entities)}
    relation_to_idx = {relation: idx for idx, relation in enumerate(relations)}

    # Process additional statistics
    kg_stats["entity_count"] = len(entities)
    kg_stats["relation_count"] = len(relations)
    kg_stats["avg_out_degree"] = (
        sum(kg_stats["entity_out_degrees"].values()) / len(entities) if entities else 0
    )

    # Save entity variant dictionary for fuzzy matching
    entity_variants = build_entity_variants(entities)

    print(
        f"Knowledge graph loaded: {len(entities)} entities, {len(relations)} relations, {kg_stats['triple_count']} triples"
    )

    return (
        kg_dict,
        entities,
        relations,
        kg_stats,
        entity_to_idx,
        relation_to_idx,
        entity_variants,
    )


def build_entity_variants(entities):
    """
    Build entity variant dictionary for fuzzy matching

    Parameters:
        entities (list): List of entities

    Returns:
        dict: Mapping from entity variant to standard name
    """
    entity_variants = {}

    for entity in entities:
        # Add original form
        entity_variants[entity] = entity

        # Add lowercase form
        entity_variants[entity.lower()] = entity

        # Add no-space form
        entity_variants[entity.replace(" ", "")] = entity
        entity_variants[entity.lower().replace(" ", "")] = entity

        # Handle special cases (e.g., "The Matrix" and "Matrix")
        if entity.lower().startswith("the "):
            no_the = entity[4:]
            entity_variants[no_the] = entity
            entity_variants[no_the.lower()] = entity

    return entity_variants


def resolve_entity(
    query, entity_variants, entities, threshold=config.ENTITY_MATCH_THRESHOLD
):
    """
    Resolve entity name with support for fuzzy matching

    Parameters:
        query (str): The queried entity name
        entity_variants (dict): Entity variant dictionary
        entities (list): List of entities
        threshold (float): Matching threshold

    Returns:
        tuple: (resolved_entity, match_score, is_exact_match)
    """
    # Normalize query
    query = query.strip()

    # Try exact match
    if query in entity_variants:
        return entity_variants[query], 1.0, True

    # Try lowercase match
    query_lower = query.lower()
    if query_lower in entity_variants:
        return entity_variants[query_lower], 0.9, False

    # Try no-space match
    query_nospace = query.replace(" ", "")
    if query_nospace in entity_variants:
        return entity_variants[query_nospace], 0.85, False

    query_lower_nospace = query_lower.replace(" ", "")
    if query_lower_nospace in entity_variants:
        return entity_variants[query_lower_nospace], 0.8, False

    # Try sequence matching (slower but more flexible)
    best_score = 0
    best_entity = None

    for entity in entities:
        # Compute sequence similarity
        score = SequenceMatcher(None, query_lower, entity.lower()).ratio()
        if score > best_score and score >= threshold:
            best_score = score
            best_entity = entity

    if best_entity:
        return best_entity, best_score, False

    # If no match is found, return the original query
    return query, 0.0, False


def extract_entity(question):
    """
    Extract subject entity from the question (content inside square brackets)

    Parameters:
        question (str): Question text

    Returns:
        str: Extracted entity; returns None if not found
    """
    match = re.search(r"\[(.*?)\]", question)
    if match:
        return match.group(1).strip()
    return None


def preprocess_question(question):
    """
    Preprocess question text

    Parameters:
        question (str): Original question

    Returns:
        tuple: (processed_question, question_type, question_focus)
    """
    # Remove square brackets
    processed = re.sub(r"\[(.*?)\]", r"\1", question)

    # Convert to lowercase and remove extra spaces
    processed = " ".join(processed.lower().split())

    # Identify question type
    question_type = "general"
    question_focus = "entity"

    if processed.startswith("who"):
        question_type = "who"
        if "direct" in processed:
            question_focus = "director"
        elif "star" in processed or "act" in processed or "play" in processed:
            question_focus = "actor"
        else:
            question_focus = "person"
    elif processed.startswith("what"):
        question_type = "what"
    elif processed.startswith("where"):
        question_type = "where"
        question_focus = "location"
    elif processed.startswith("when"):
        question_type = "when"
        question_focus = "time"
    elif "how many" in processed:
        question_type = "howmany"
        question_focus = "count"

    return processed, question_type, question_focus


def load_qa_data(qa_path, use_cache=True):
    """
    Load Q&A dataset

    Parameters:
        qa_path (str): Q&A data file path
        use_cache (bool): Whether to use cache

    Returns:
        tuple: (questions, answers, question_entities, processed_questions, question_types, question_focuses)
    """
    print(f"Loading Q&A data: {qa_path}")

    # Check cache
    cache_file = f"{qa_path}.cache.json"
    if use_cache and os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            cache = json.load(f)
            return (
                cache["questions"],
                cache["answers"],
                cache["question_entities"],
                cache["processed_questions"],
                cache["question_types"],
                cache["question_focuses"],
            )

    questions = []
    answers = []

    # Read Q&A data
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading Q&A data"):
            parts = line.strip().split("\t")
            if len(parts) == 2:
                question, answer = parts
                # Process multiple answers
                ans_list = answer.split("|")
                questions.append(question)
                answers.append(ans_list)

    # Extract entities from questions
    question_entities = [extract_entity(q) for q in questions]

    # Preprocess questions
    processed_results = [preprocess_question(q) for q in questions]
    processed_questions = [r[0] for r in processed_results]
    question_types = [r[1] for r in processed_results]
    question_focuses = [r[2] for r in processed_results]

    # Save cache
    if use_cache:
        cache = {
            "questions": questions,
            "answers": answers,
            "question_entities": question_entities,
            "processed_questions": processed_questions,
            "question_types": question_types,
            "question_focuses": question_focuses,
        }
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    print(f"Q&A data loaded: {len(questions)} questions")

    return (
        questions,
        answers,
        question_entities,
        processed_questions,
        question_types,
        question_focuses,
    )


def load_embeddings(entities, relations, pretrained_path=None, dim=config.ENTITY_DIM):
    """
    Load or initialize entity and relation embeddings

    Parameters:
        entities (list): List of entities
        relations (list): List of relations
        pretrained_path (str): Path to pretrained embeddings; if None, initialize randomly
        dim (int): Embedding dimension

    Returns:
        tuple: (entity_embeddings, relation_embeddings)
    """
    # If pretrained embeddings exist, load them
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Loading pretrained embeddings: {pretrained_path}")
        pretrained = torch.load(pretrained_path)
        entity_embeddings = pretrained["entity_embeddings"]
        relation_embeddings = pretrained["relation_embeddings"]

        # Check if dimensions and number of entities/relations match
        if entity_embeddings.size(0) != len(entities) or relation_embeddings.size(
            0
        ) != len(relations):
            print(
                "Pretrained embeddings dimensions do not match current entities/relations, reinitializing"
            )
            entity_embeddings = torch.randn(len(entities), dim)
            relation_embeddings = torch.randn(len(relations), dim)
    else:
        print(f"Randomly initializing embeddings (dim={dim})")
        # Use scaled random initialization for more stable initial values
        entity_embeddings = torch.randn(len(entities), dim) / np.sqrt(dim)
        relation_embeddings = torch.randn(len(relations), dim) / np.sqrt(dim)

    return entity_embeddings, relation_embeddings


def find_relation_by_question(question_type, question_focus, relations):
    """
    Find possible relationships based on question type and focus

    Parameters:
        question_type (str): Question type
        question_focus (str): Question focus
        relations (list): List of relations

    Returns:
        list: List of possible relations sorted by relevance
    """
    relevant_relations = []

    # Build keyword mapping
    keywords = {
        "director": ["direct", "director", "film", "movie", "directed_by"],
        "actor": ["star", "actor", "actress", "cast", "play", "starring", "played_by"],
        "location": ["location", "place", "set", "filmed", "country", "city"],
        "time": ["year", "date", "when", "time", "released"],
        "count": ["number", "total", "many", "count"],
    }

    # Get keywords for current question focus
    focus_keywords = keywords.get(question_focus, [])

    # Score relations based on relevance
    scored_relations = []
    for rel in relations:
        rel_lower = rel.lower()
        score = 0

        # Check for keyword matches
        for keyword in focus_keywords:
            if keyword in rel_lower:
                score += 2

        # Special rules
        if question_focus == "director" and "direct" in rel_lower:
            score += 3
        elif question_focus == "actor" and any(
            kw in rel_lower for kw in ["star", "play", "act"]
        ):
            score += 3

        # Only include relations with a positive score
        if score > 0:
            scored_relations.append((rel, score))

    # Sort in descending order by score
    scored_relations.sort(key=lambda x: x[1], reverse=True)

    # Return list of relations
    return [rel for rel, _ in scored_relations]
