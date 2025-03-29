import os
import json
import re
import openai
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from typing import List, Dict, Tuple, Set, Optional, Union

# Configuration (replace with your own API key)
openai.api_key = ""


class KnowledgeGraph:
    """Knowledge Graph class for MetaQA dataset."""

    def __init__(self, kb_path: str):
        """
        Initialize the Knowledge Graph from the MetaQA KB file.

        Args:
            kb_path: Path to the kb.txt file containing the knowledge graph
        """
        self.triples = []
        self.entity_to_relations = {}  # Maps entity to outgoing relations and target entities
        self.entity_to_incoming = {}  # Maps entity to incoming relations and source entities
        self.relation_types = set()  # Set of all relation types
        self.entities = set()  # Set of all entities
        self.entity_lower_to_exact = {}  # Maps lowercase entity names to exact entity names
        self.relation_counts = {}  # Counts the frequency of each relation

        self.load_kb(kb_path)
        self.analyze_kg_structure()

    def load_kb(self, kb_path: str):
        """
        Load the knowledge graph from kb.txt.

        Args:
            kb_path: Path to the kb.txt file
        """
        print(f"Loading knowledge graph from {kb_path}...")
        with open(kb_path, "r", encoding="utf-8") as f:
            for line in tqdm(f):
                # Parse the subject|relation|object format
                parts = line.strip().split("|")
                if len(parts) != 3:
                    continue

                subject, relation, obj = parts
                self.triples.append((subject, relation, obj))

                # Add to entity sets
                self.entities.add(subject)
                self.entities.add(obj)
                self.relation_types.add(relation)

                # Update relation counts
                if relation not in self.relation_counts:
                    self.relation_counts[relation] = 0
                self.relation_counts[relation] += 1

                # Map subject to (relation, object)
                if subject not in self.entity_to_relations:
                    self.entity_to_relations[subject] = []
                self.entity_to_relations[subject].append((relation, obj))

                # Map object to (relation, subject) for reverse lookup
                if obj not in self.entity_to_incoming:
                    self.entity_to_incoming[obj] = []
                self.entity_to_incoming[obj].append((relation, subject))

        # Build lowercase to exact entity mapping for better entity linking
        for entity in self.entities:
            self.entity_lower_to_exact[entity.lower()] = entity

        print(
            f"Loaded {len(self.triples)} triples with {len(self.entities)} entities and {len(self.relation_types)} relation types"
        )

    def analyze_kg_structure(self):
        """Analyze the knowledge graph structure to understand relation patterns."""
        # For each relation type, analyze how it's used
        relation_stats = {}
        for relation in self.relation_types:
            relation_stats[relation] = {
                "count": self.relation_counts.get(relation, 0),
                "subject_types": set(),
                "object_types": set(),
            }

            # Sample a few triples with this relation to understand the entity types
            sample_triples = [t for t in self.triples if t[1] == relation][:100]
            for subj, rel, obj in sample_triples:
                # For now, just collect a sample of subjects and objects
                relation_stats[relation]["subject_types"].add(subj)
                relation_stats[relation]["object_types"].add(obj)

        self.relation_stats = relation_stats

        # Print some useful statistics
        print("\nKnowledge Graph Analysis:")
        for relation, stats in sorted(
            self.relation_counts.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"- {relation}: {stats} occurrences")

    def get_outgoing_relations(self, entity: str) -> List[Tuple[str, str]]:
        """
        Get all outgoing relations and target entities for a given entity.

        Args:
            entity: Source entity

        Returns:
            List of (relation, target_entity) tuples
        """
        return self.entity_to_relations.get(entity, [])

    def get_incoming_relations(self, entity: str) -> List[Tuple[str, str]]:
        """
        Get all incoming relations and source entities for a given entity.

        Args:
            entity: Target entity

        Returns:
            List of (relation, source_entity) tuples
        """
        return self.entity_to_incoming.get(entity, [])

    def get_neighbors(self, entity: str) -> List[Tuple[str, str, bool]]:
        """
        Get all neighbor entities with relations for a given entity.

        Args:
            entity: Entity to find neighbors for

        Returns:
            List of (relation, neighbor_entity, is_outgoing) tuples
        """
        outgoing = [
            (relation, target, True)
            for relation, target in self.get_outgoing_relations(entity)
        ]
        incoming = [
            (relation, source, False)
            for relation, source in self.get_incoming_relations(entity)
        ]
        return outgoing + incoming

    def get_entities_by_name(self, name_fragment: str) -> List[str]:
        """
        Find entities containing the given name fragment.

        Args:
            name_fragment: Part of entity name to search for

        Returns:
            List of matching entity names
        """
        # Try exact match first
        if name_fragment in self.entities:
            return [name_fragment]

        # Try case-insensitive exact match
        lower_name = name_fragment.lower()
        if lower_name in self.entity_lower_to_exact:
            return [self.entity_lower_to_exact[lower_name]]

        # Try substring match
        matches = [entity for entity in self.entities if lower_name in entity.lower()]

        # If no matches or too many matches, try fuzzy matching
        if not matches or len(matches) > 10:
            fuzzy_matches = self.fuzzy_match_entity(name_fragment)
            if fuzzy_matches:
                return fuzzy_matches

        return matches

    def fuzzy_match_entity(
        self, name_fragment: str, threshold: float = 0.7
    ) -> List[str]:
        """
        Find entities that approximately match the given name fragment.

        Args:
            name_fragment: Entity name to match
            threshold: Similarity threshold (0-1)

        Returns:
            List of matching entity names
        """
        lower_name = name_fragment.lower()
        candidates = []

        # Check each entity
        for entity in self.entities:
            lower_entity = entity.lower()
            # If entity contains name or name contains entity
            if lower_name in lower_entity or lower_entity in lower_name:
                # Calculate simple similarity score
                similarity = self.calculate_similarity(lower_name, lower_entity)
                if similarity > threshold:
                    candidates.append((entity, similarity))

        # Sort by similarity score
        candidates.sort(key=lambda x: x[1], reverse=True)
        # Return top matches
        return [entity for entity, score in candidates[:5]]

    def calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate string similarity between two strings.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Similarity score (0-1)
        """
        # Simple character-based similarity
        # If one is contained in the other, that's good
        if str1 in str2:
            return len(str1) / len(str2)
        if str2 in str1:
            return len(str2) / len(str1)

        # Count matching characters in sequence
        i, j = 0, 0
        matches = 0
        while i < len(str1) and j < len(str2):
            if str1[i] == str2[j]:
                matches += 1
                i += 1
                j += 1
            elif len(str1) < len(str2):
                j += 1
            else:
                i += 1

        # Return ratio of matches to max length
        return matches / max(len(str1), len(str2))

    def extract_answer_entities(
        self, question: str, context_entities: List[str] = None
    ) -> List[str]:
        """
        Extract potential answer entities from question based on context.

        Args:
            question: Question text
            context_entities: List of entities already identified in the context

        Returns:
            List of potential answer entities
        """
        # This is a simple implementation that can be improved with entity linking methods
        # For now, we'll look for entities mentioned in the question
        potential_entities = []
        if context_entities is None:
            context_entities = []

        # Try all entities in the KG (this is inefficient but works for small KGs)
        for entity in self.entities:
            # Skip context entities which are likely question entities, not answers
            if entity in context_entities:
                continue

            # Simple string match
            if len(entity) > 3 and entity.lower() in question.lower():
                potential_entities.append(entity)

        return potential_entities

    def improve_entity_linking(self, mention: str) -> str:
        """
        Find the best match for an entity mention in the knowledge graph.

        Args:
            mention: Text mention of an entity

        Returns:
            The best matching entity from the knowledge graph
        """
        if not mention:
            return None

        # Exact match
        if mention in self.entities:
            return mention

        # Case-insensitive match
        lower_mention = mention.lower()
        if lower_mention in self.entity_lower_to_exact:
            return self.entity_lower_to_exact[lower_mention]

        # Get entities by name (partial match)
        candidates = self.get_entities_by_name(mention)
        if candidates:
            # If only one candidate, return it
            if len(candidates) == 1:
                return candidates[0]

            # If multiple candidates, select best match
            best_match = None
            highest_similarity = 0

            for entity in candidates:
                similarity = self.calculate_similarity(lower_mention, entity.lower())
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = entity

            if best_match and highest_similarity > 0.6:
                return best_match

        # If no good matches, look for entities containing mention
        candidates = [e for e in self.entities if lower_mention in e.lower()]
        if candidates:
            # Sort by length (prefer shorter matches)
            candidates.sort(key=len)
            return candidates[0]

        # If still no matches, return original mention
        return mention


class MetaQADataset:
    """Class for loading and handling MetaQA question-answer datasets."""

    def __init__(self, base_dir: str, hop: int = 1, data_type: str = "vanilla"):
        """
        Initialize the MetaQA dataset loader.

        Args:
            base_dir: Base directory containing the MetaQA dataset
            hop: Number of hops (1, 2, or 3)
            data_type: Type of data ("vanilla", "ntm", or "audio")
        """
        self.base_dir = base_dir
        self.hop = hop
        self.data_type = data_type
        self.qa_pairs = {"train": [], "dev": [], "test": []}

        self.load_data()

    def load_data(self):
        """Load the MetaQA question-answer pairs."""
        for split in ["train", "dev", "test"]:
            path = os.path.join(
                self.base_dir, f"{self.hop}-hop", self.data_type, f"qa_{split}.txt"
            )

            if not os.path.exists(path):
                print(f"Warning: File not found: {path}")
                continue

            print(f"Loading {split} data from {path}...")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) != 2:
                        continue

                    question, answer_str = parts
                    # Handle multiple answers separated by |
                    answers = answer_str.split("|")

                    # Extract topic entity from brackets [Entity]
                    topic_entity_match = re.search(r"\[(.*?)\]", question)
                    topic_entity = (
                        topic_entity_match.group(1) if topic_entity_match else None
                    )

                    # Clean question by removing brackets
                    clean_question = re.sub(r"\[(.*?)\]", r"\1", question)

                    self.qa_pairs[split].append(
                        {
                            "question": clean_question,
                            "answers": answers,
                            "topic_entity": topic_entity,
                            "original_question": question,
                            "hop": self.hop,
                        }
                    )

            print(f"Loaded {len(self.qa_pairs[split])} {split} examples")


class LLMQuestionDecomposer:
    """Class for decomposing multi-hop questions into simpler subquestions using LLM."""

    def __init__(self, model_name: str = "gpt-3.5-turbo", kg: "KnowledgeGraph" = None):
        """
        Initialize the question decomposer.

        Args:
            model_name: Name of the OpenAI model to use
            kg: Knowledge graph object (optional)
        """
        self.model_name = model_name
        self.kg = kg

    def decompose_question(
        self, question: str, topic_entity: str, hop_count: int
    ) -> List[str]:
        """
        Decompose a multi-hop question into simpler subquestions.

        Args:
            question: The original question
            topic_entity: The topic entity in the question
            hop_count: Expected number of hops required to answer the question

        Returns:
            List of subquestions
        """
        if hop_count <= 1:
            # No need to decompose 1-hop questions
            return [question]

        # Get information about relevant relations for better decomposition
        relation_info = ""
        if self.kg:
            relations = sorted(list(self.kg.relation_types))
            relation_info = "The knowledge graph has the following relation types:\n"
            relation_info += "\n".join([f"- {relation}" for relation in relations[:10]])
            relation_info += "\n\nThe topic entity has the following relations:\n"

            # Show relations for the topic entity if it exists
            outgoing = self.kg.get_outgoing_relations(topic_entity)
            incoming = self.kg.get_incoming_relations(topic_entity)

            if outgoing:
                relation_info += f"Outgoing relations from '{topic_entity}':\n"
                relation_info += "\n".join(
                    [
                        f"- {topic_entity} --[{rel}]--> {target}"
                        for rel, target in outgoing[:5]
                    ]
                )
                relation_info += "\n"

            if incoming:
                relation_info += f"Incoming relations to '{topic_entity}':\n"
                relation_info += "\n".join(
                    [
                        f"- {source} --[{rel}]--> {topic_entity}"
                        for rel, source in incoming[:5]
                    ]
                )

        prompt = f"""Given a multi-hop question about a knowledge graph in the movie domain, decompose it into {hop_count} sequential subquestions. Each subquestion should operate on the *results* of the previous subquestion.

Original question: {question}
Topic entity: {topic_entity}

{relation_info} # Optional: 關係信息可能幫助也可能干擾，可以試試有無此信息

Instructions for decomposition:
1. The first subquestion must start with the topic entity '{topic_entity}' and find intermediate entities.
2. Subsequent subquestions (Subquestion 2, Subquestion 3, etc.) must *clearly state* that they operate on the entities found in the previous step. Use placeholders like '[output of previous step]' or '[intermediate movies]' or '[intermediate actors]' to represent the results from the previous subquestion.
3. The final subquestion should aim to find the answer to the original question, using the results from the penultimate step.
4. Keep subquestions simple and focused on a single relation traversal or operation if possible.

Provide exactly {hop_count} subquestions. Format your answer strictly as:
Subquestion 1: [First subquestion starting with '{topic_entity}']
Subquestion 2: [Second subquestion operating on results of Subquestion 1, e.g., 'Who directed the movies found in step 1?']
{f"Subquestion 3: [Third subquestion operating on results of Subquestion 2]" if hop_count >= 3 else ""}

Example Decomposition for 'which person directed the movies starred by John Krasinski':
Subquestion 1: Which movies did John Krasinski star in?
Subquestion 2: Who directed the movies found in step 1?

Example Decomposition for 'genre of movies written by the director of Inception':
Subquestion 1: Who directed Inception?
Subquestion 2: Which movies were written by the director found in step 1?
Subquestion 3: What are the genres of the movies found in step 2?

Now, decompose the original question following these instructions.
Original question: {question}
"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that decomposes complex questions into simpler ones.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            result = response.choices[0].message.content.strip()
            subquestions = []

            # Extract the subquestions from the response
            for i in range(1, hop_count + 1):
                match = re.search(
                    f"Subquestion {i}: (.*?)(?:\nSubquestion|$)", result, re.DOTALL
                )
                if match:
                    subquestions.append(match.group(1).strip())

            # Ensure we have the expected number of subquestions
            if len(subquestions) != hop_count:
                print(
                    f"Warning: Expected {hop_count} subquestions but got {len(subquestions)}. Falling back to original question."
                )
                if hop_count == 2:
                    # Try to create simple 2-hop decomposition
                    return [
                        f"What are the immediate connections to {topic_entity}?",
                        question,
                    ]
                elif hop_count == 3:
                    # Try to create simple 3-hop decomposition
                    return [
                        f"What are the immediate connections to {topic_entity}?",
                        f"What entities are connected to the results from {topic_entity}?",
                        question,
                    ]
                return [question]

            return subquestions

        except Exception as e:
            print(f"Error in question decomposition: {e}")
            return [question]  # Fall back to the original question


class LLMActionSelector:
    """Class for selecting knowledge graph actions using LLM."""

    ACTION_TYPES = [
        "Basic",  # Simple relation selection
        "Filter",  # Filter results based on a criterion
        "Union",  # Combine results from different paths
        "Intersection",  # Find common elements in different paths
        "Count",  # Count the number of results
        "Comparison",  # Compare attributes of entities
        "Aggregation",  # Aggregate values (e.g., find max/min)
    ]

    def __init__(
        self, knowledge_graph: KnowledgeGraph, model_name: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the action selector.

        Args:
            knowledge_graph: The knowledge graph
            model_name: Name of the OpenAI model to use
        """
        self.kg = knowledge_graph
        self.model_name = model_name

    def select_action_type(
        self, question: str, available_types: List[str] = None
    ) -> str:
        """
        Select the appropriate action type for a given question.

        Args:
            question: The question
            available_types: List of available action types to choose from

        Returns:
            Selected action type
        """
        if available_types is None:
            available_types = self.ACTION_TYPES

        action_descriptions = {
            "Basic": "Simple relation traversal (e.g., 'Who directed The Matrix?')",
            "Filter": "Filter entities based on a criterion (e.g., 'Which comedy movies from 2010 did Adam Sandler star in?')",
            "Union": "Combine results from different paths (e.g., 'Which actors appeared in either The Matrix or Inception?')",
            "Intersection": "Find common elements (e.g., 'Which actors appeared in both The Matrix and Inception?')",
            "Count": "Count the number of results (e.g., 'How many movies did Christopher Nolan direct?')",
            "Comparison": "Compare attributes (e.g., 'Which movie directed by James Cameron had the highest rating?')",
            "Aggregation": "Aggregate over entities (e.g., 'What is the average rating of movies directed by Steven Spielberg?')",
        }

        descriptions = "\n".join(
            [
                f"- {action_type}: {action_descriptions[action_type]}"
                for action_type in available_types
            ]
        )

        prompt = f"""Given a question about a movie knowledge graph, select the most appropriate action type to answer it.

Here are some examples:
Question: Who directed The Matrix?
Action Type: Basic

Question: Which comedy movies from 2010 did Adam Sandler star in?
Action Type: Filter

Question: How many movies did Christopher Nolan direct?
Action Type: Count

Question: Which actors appeared in both The Matrix and Inception?
Action Type: Intersection

Question: Which movies did John Krasinski star in?
Action Type: Basic

---
Now, answer for the following question:
Question: {question}

Available action types:
{descriptions}

Return only the name of the most appropriate action type from the list.
"""

        # --- 開始 DEBUG ---
        print("\n--- Debug: select_action_type ---")
        print(f"Question: {question}")
        print("--- Prompt Sent to LLM ---")
        print(prompt)
        # --- 結束 DEBUG ---

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that classifies questions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=50,
                logprobs=True,  # 請求 logprobs
                top_logprobs=5,  # 獲取前 5 個 token 的概率 (應該能覆蓋我們的 action type)
            )

            result = response.choices[0].message.content.strip()
            logprobs_content = (
                response.choices[0].logprobs.content
                if response.choices[0].logprobs
                else None
            )

            print("--- LLM Response Received ---")
            print(f"Raw Content: {result}")
            print("--- Logprobs Received (Raw) ---")
            # --- 結束 DEBUG ---
            print(logprobs_content)

            # --- 開始解析 Logprobs (簡易版本) ---
            action_probs = {}
            if logprobs_content:
                # 假設 Action Type 是第一個 token
                first_token_logprobs = (
                    logprobs_content[0].top_logprobs if logprobs_content else []
                )
                print("--- Top Logprobs for First Token ---")
                for logprob_info in first_token_logprobs:
                    token_str = logprob_info.token
                    prob = np.exp(logprob_info.logprob)  # 轉換回概率
                    print(f"Token: '{token_str}', Probability: {prob:.4f}")
                    # 嘗試將 token 映射回 action type
                    # 注意：這裡只匹配了單詞完全一致的情況，\"Intersection\" 等多 token 的沒處理
                    if token_str in available_types:
                        action_probs[token_str] = prob

                # 處理多 Token 的情況 (非常簡陋的示例，可能不準確)
                # 例如，如果看到 \"Inter\"，可以猜測是 \"Intersection\"
                # 需要更健壯的映射邏輯
                # if not action_probs.get("Intersection") and any(lp.token == "Inter" for lp in first_token_logprobs):
                #     # 估算概率...
                #     pass

            print("--- Parsed Action Probabilities (Simple) ---")
            print(action_probs)
            # --- Logprobs 解析結束 ---

            # --- 決策邏輯：仍然先用最高概率的 (需要改進) ---
            selected_action = "Basic"  # 預設
            highest_prob = 0.0
            if action_probs:
                # 選擇字典中概率最高的 Action Type
                selected_action = max(action_probs, key=action_probs.get)
                highest_prob = action_probs[selected_action]
                print(
                    f"--- Action selected based on highest probability: {selected_action} (Prob: {highest_prob:.4f}) ---"
                )
            else:
                # 如果無法從 logprobs 解析，則回退到舊的基於 result 字符串匹配的方法
                print(
                    "--- Warning: Could not parse probabilities, falling back to string matching ---"
                )
                for action_type in available_types:
                    if action_type in result:
                        selected_action = action_type
                        break
                if (
                    selected_action == "Basic"
                    and "Basic" not in result
                    and not action_probs
                ):  # 避免重複打印警告
                    print(
                        f"Warning: Could not identify action type from response: '{result}'. Defaulting to Basic."
                    )

            print(f"--- Final Selected Action Type: {selected_action} ---")
            return selected_action

        except Exception as e:
            print(f"Error in action type selection: {e}")
            print("--- Defaulting Action Type: Basic ---")
            return "Basic"

    def get_all_relations(
        self, entity: str, question: str
    ) -> List[Tuple[str, str, str, float]]:
        """
        Get all relevant relations (both outgoing and incoming) for an entity.

        Args:
            entity: Entity to get relations for
            question: Question to rank relations by relevance

        Returns:
            List of (direction, relation, connected_entity, relevance_score) tuples
        """
        relations = []

        # Get outgoing relations (entity → relation → target)
        outgoing = self.kg.get_outgoing_relations(entity)
        for relation, target in outgoing:
            # Simple relevance scoring - could be improved with embeddings
            question_lower = question.lower()
            relation_lower = relation.lower()
            target_lower = target.lower()

            score = 0.5  # Base score

            # Boost score if relation or target appears in question
            if relation_lower in question_lower:
                score += 0.3
            if target_lower in question_lower:
                score += 0.2

            relations.append(("outgoing", relation, target, score))

        # Get incoming relations (source → relation → entity)
        incoming = self.kg.get_incoming_relations(entity)
        for relation, source in incoming:
            # Score incoming relations (often slightly less relevant)
            question_lower = question.lower()
            relation_lower = relation.lower()
            source_lower = source.lower()

            score = 0.4  # Lower base score for incoming

            # Boost score if relation or source appears in question
            if relation_lower in question_lower:
                score += 0.3
            if source_lower in question_lower:
                score += 0.2

            relations.append(("incoming", relation, source, score))

        # Sort by relevance score
        relations.sort(key=lambda x: x[3], reverse=True)

        return relations

    def select_relation(
        self,
        question: str,
        current_entity: str,
        previous_path: List[Tuple[str, str]] = None,
    ) -> Tuple[str, str, str]:
        """
        Select the most relevant relation and connected entity from the current entity.

        Args:
            question: The question or subquestion
            current_entity: Current entity in the traversal
            previous_path: List of (relation, entity) pairs in the path so far

        Returns:
            Tuple of (direction, selected_relation, connected_entity)
        """
        if previous_path is None:
            previous_path = []

        # Get all available relations (both directions)
        all_relations = self.get_all_relations(current_entity, question)

        if not all_relations:
            print(f"No relations found for entity: {current_entity}")
            return None, None, None

        # Limit the number of relations to avoid token limits
        if len(all_relations) > 50:
            print(
                f"Warning: Too many relations ({len(all_relations)}) for entity {current_entity}. Limiting to 50."
            )
            all_relations = all_relations[:50]

        # Format relations for prompt
        relations_text = ""
        for i, (direction, relation, connected, _) in enumerate(all_relations):
            if direction == "outgoing":
                relations_text += (
                    f"{i + 1}. {current_entity} --[{relation}]--> {connected}\n"
                )
            else:  # incoming
                relations_text += (
                    f"{i + 1}. {connected} --[{relation}]--> {current_entity}\n"
                )

        # Format the previous path
        path_text = ""
        if previous_path:
            path_steps = []
            current_path_entity = None

            for idx, (rel, ent) in enumerate(previous_path):
                if idx == 0:
                    prev_entity = (
                        current_entity
                        if idx == len(previous_path) - 1
                        else previous_path[idx + 1][1]
                    )
                    path_steps.append(f"{prev_entity} -[{rel}]-> {ent}")
                    current_path_entity = ent
                else:
                    source = (
                        previous_path[idx - 1][1] if current_path_entity else "Unknown"
                    )
                    path_steps.append(f"{source} -[{rel}]-> {ent}")
                    current_path_entity = ent

            path_text = f"\nPrevious path steps:\n" + "\n".join(path_steps)

        prompt = f"""Given a question about a movie knowledge graph and the current entity, select the most relevant relation to follow to answer the question.

Question: {question}
Current entity: {current_entity}{path_text}

Available relations involving this entity:
{relations_text}

Return only the NUMBER of the most relevant relation to follow. For example, if relation #3 is most relevant, return just the number 3.
"""
        # --- 開始 DEBUG ---
        print("\n--- Debug: select_relation ---")
        print(f"Question: {question}")
        print(f"Current Entity: {current_entity}")
        print(f"Available Relations Count: {len(all_relations)}")
        print("--- Prompt Sent to LLM ---")
        print(prompt)
        # --- 結束 DEBUG ---

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that selects relevant relations in a knowledge graph.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=50,
            )

            result = response.choices[0].message.content.strip()

            # --- 開始 DEBUG ---
            print("--- LLM Response Received ---")
            print(result)
            # --- 結束 DEBUG ---

            number_match = re.search(r"\d+", result)
            selected_relation_info = (None, None, None)  # 預設值

            if number_match:
                relation_idx = int(number_match.group(0)) - 1
                if 0 <= relation_idx < len(all_relations):
                    selected_relation_info = all_relations[relation_idx][
                        :3
                    ]  # 取 direction, relation, connected
                    print(f"--- Extracted Relation Index: {relation_idx + 1} ---")
                    print(f"--- Selected Relation Info: {selected_relation_info} ---")
                else:
                    print(
                        f"--- Warning: Extracted index {relation_idx + 1} out of range ---"
                    )
            else:
                print(
                    f"--- Warning: Could not extract number from response: '{result}' ---"
                )

            if selected_relation_info == (None, None, None) and all_relations:
                selected_relation_info = all_relations[0][:3]  # 取最高分的
                print(
                    f"--- Falling back to highest scored relation: {selected_relation_info} ---"
                )
            elif not all_relations:
                print(f"--- Error: No relations available to select from ---")

            return selected_relation_info  # 返回解析結果或最高分結果或 None

        except Exception as e:
            print(f"Error in relation selection: {e}")
            if all_relations:
                selected_relation_info = all_relations[0][:3]
                print(
                    f"--- Error Fallback to highest scored relation: {selected_relation_info} ---"
                )
                return selected_relation_info
            print(f"--- Error: No relations available after exception ---")
            return None, None, None

    def _find_relevant_relation_and_direction(
        self, question: str
    ) -> Tuple[Optional[str], bool]:
        q_lower = question.lower()
        q_tokens = q_lower.split()  # 簡單分詞

        # 查找人物 (導演、演員、編劇)
        if "who directed" in q_lower:
            return "directed_by", True  # Movie --> Director (Outgoing)
        # "who starred in X", "actors of X", "who was in X"
        if (
            "who starred" in q_lower
            or "actors" in q_tokens
            or ("who" in q_tokens and "in" in q_tokens)
        ) and ("movie" in q_lower or "film" in q_lower):
            return "starred_actors", True  # Movie --> Actor (Outgoing)
        if "who wrote" in q_lower or "writer of" in q_lower:
            return "written_by", True  # Movie --> Writer (Outgoing)

        # 查找電影
        # "movies directed by X", "films by X"
        if ("movie" in q_lower or "film" in q_lower) and (
            "directed by" in q_lower
            or (
                "by" in q_tokens
                and q_tokens.index("by") > 0
                and q_tokens[q_tokens.index("by") - 1]
                in ["directed", "films", "movies"]
            )
        ):
            return "directed_by", False  # Director --> Movie (Incoming)
        # "movies starring X", "movies with X", "movies X starred in"
        if ("movie" in q_lower or "film" in q_lower) and (
            "starring" in q_lower
            or "starred by" in q_lower
            or "with" in q_tokens
            or "starred in" in q_lower
        ):
            # Check if "starring" is used for actors, not just title
            # This logic needs careful thought based on MetaQA patterns
            if (
                "actor" not in q_lower
            ):  # Avoid "actors starring in..." which is handled above
                return "starred_actors", False  # Actor --> Movie (Incoming)
        # "movies written by X"
        if ("movie" in q_lower or "film" in q_lower) and ("written by" in q_lower):
            return "written_by", False  # Writer --> Movie (Incoming)

        # 查找屬性 (保持不變或微調)
        if "what genre" in q_lower or "genre of" in q_lower:
            return "has_genre", True
        if "what year" in q_lower or "release year" in q_lower:
            return "release_year", True
        if "what language" in q_lower:
            return "in_language", True

        print(
            f"--- Warning: Could not determine relevant relation/direction for question: {question} ---"
        )
        return None, False

    def execute_action(
        self,
        action_type: str,
        question: str,
        current_entities: List[str],
        kg: KnowledgeGraph,
    ) -> List[str]:
        """
        Execute a knowledge graph action based on the selected action type.

        Args:
            action_type: Type of action to execute
            question: The question
            current_entities: Current entities to start from
            kg: Knowledge graph

        Returns:
            List of result entities after executing the action
        """
        if not current_entities:
            print("Warning: No current entities provided for action execution")
            return []

        # Basic action - follow most relevant relation for each entity
        if action_type == "Basic":
            results = set()  # 使用 set 避免重複
            print(
                f"--- Inside Basic Action (Revised Logic for Multiple Results) for entities: {current_entities} ---"
            )

            # 嘗試從問題判斷需要的關係和方向
            wanted_relation, expect_outgoing = (
                self._find_relevant_relation_and_direction(question)
            )

            if wanted_relation is None:
                print(
                    f"--- Basic Action Failed: Cannot determine relation for question '{question}'. Falling back to old select_relation (might be wrong). ---"
                )
                # --- Fallback: 使用舊的 select_relation 邏輯 (如果上面無法判斷) ---
                # 注意：這仍然有只選一個的問題，但作為臨時後備
                fallback_results = []
                for entity in current_entities:
                    direction, relation, connected = self.select_relation(
                        question, entity
                    )
                    if connected:
                        fallback_results.append(connected)
                print(
                    f"--- Returning results from Basic Action (Fallback Logic): {fallback_results} ---"
                )
                return fallback_results
                # --- End Fallback ---

            print(
                f"--- Determined: Wanted Relation='{wanted_relation}', Expect Outgoing={expect_outgoing} ---"
            )

            for entity in current_entities:
                print(f"--- Processing entity: {entity} ---")
                neighbors = kg.get_neighbors(entity) 
                print(f"--- Found {len(neighbors)} neighbors for {entity} ---")
                for relation, neighbor_entity, is_outgoing in neighbors:
                    # 檢查關係類型是否匹配，方向是否匹配
                    if relation == wanted_relation and is_outgoing == expect_outgoing:
                        print(
                            f"--- Adding entity: {neighbor_entity} (Relation: {relation}, Outgoing: {is_outgoing}) ---"
                        )
                        results.add(neighbor_entity)

            final_results = list(results)
            print(
                f"--- Returning results from Basic Action (Revised Logic): {final_results} ---"
            )
            return final_results

        elif action_type == "Count":
            # Count 依賴於 Basic 能返回正確的集合
            basic_results = self.execute_action("Basic", question, current_entities, kg)
            count = len(basic_results)
            print(f"--- Count Action Result: {count} ---")
            return [str(count)]  # 返回計數結果，以字符串形式

        # Filter action - filter results based on criteria
        elif action_type == "Filter":
            basic_results = self.execute_action("Basic", question, current_entities, kg)
            if not basic_results:
                return []

            # Use LLM to filter the results
            filter_prompt = f"""Given a list of entities and a question, return only the entities that satisfy the conditions in the question.

Question: {question}
Entities: {", ".join(basic_results)}

Return only the entities that match the criteria in the question, separated by commas.
"""
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that filters entities based on criteria.",
                        },
                        {"role": "user", "content": filter_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=200,
                )

                result = response.choices[0].message.content.strip()
                filtered_entities = [entity.strip() for entity in result.split(",")]

                # Ensure all returned entities are in the original list
                valid_entities = [
                    entity for entity in filtered_entities if entity in basic_results
                ]
                print(
                    f"--- Filter Action executed on {len(basic_results)} basic results (LLM filtering part omitted) ---"
                )

                return valid_entities if valid_entities else basic_results

            except Exception as e:
                print(f"Error in filter action: {e}")
                return basic_results

        # Union action - combine results from multiple paths
        elif action_type == "Union":
            # Get entities mentioned in the question
            entities_in_question = kg.extract_answer_entities(
                question, current_entities
            )

            # For each entity, execute Basic action and combine results
            all_results = set()

            # Start with basic action on current entities
            basic_results = self.execute_action("Basic", question, current_entities, kg)
            all_results.update(basic_results)

            # Add results from any entities mentioned in the question
            for entity in entities_in_question:
                entity_results = self.execute_action("Basic", question, [entity], kg)
                all_results.update(entity_results)

            return list(all_results)

        # Intersection action - find common elements
        elif action_type == "Intersection":
            # 確保 execute_intersection_action 被正確調用
            try:
                print(f"--- Executing Intersection Action for question: {question} ---")
                # 將 kg 傳遞給 intersection
                return self.execute_intersection_action(question, current_entities, kg)
            except Exception as e:
                print(f"Error in intersection action: {e}")
                print(f"--- Intersection failed, falling back to Basic ---")
                return self.execute_action("Basic", question, current_entities, kg)

        # Comparison and Aggregation actions (similar logic)
        elif action_type in ["Comparison", "Aggregation"]:
            # These require understanding of entity attributes
            # For now, fall back to Basic action but could be enhanced
            print(
                f"Warning: Action type {action_type} not fully implemented. Using Basic action."
            )
            return self.execute_action("Basic", question, current_entities, kg)

        # Default fallback
        else:
            print(
                f"Warning: Unknown action type {action_type}. Falling back to Basic action."
            )
            return self.execute_action("Basic", question, current_entities, kg)

    def execute_intersection_action(
        self, question: str, current_entities: List[str], kg: KnowledgeGraph
    ) -> List[str]:
        """
        Execute an intersection action to find common entities.

        Args:
            question: The question
            current_entities: Current entities
            kg: Knowledge graph

        Returns:
            List of entities that satisfy the intersection
        """
        # Extract potential entity sets to intersect
        entity_sets = []

        # First set: Basic action on current entities
        basic_results = self.execute_action("Basic", question, current_entities, kg)
        if basic_results:
            entity_sets.append(set(basic_results))

        # Second set: Entities that appear in question and might be related
        entities_in_question = []
        for word in question.split():
            # Clean the word
            clean_word = re.sub(r"[^\w\s]", "", word).strip()
            if len(clean_word) > 3:  # Avoid short words
                # Look for entities similar to this word
                matches = kg.get_entities_by_name(clean_word)
                if matches:
                    entities_in_question.extend(matches)

        # For each mentioned entity, get related entities
        for entity in entities_in_question:
            if entity in current_entities:
                continue  # Skip current entities

            related = []
            # Get outgoing relations
            for rel, target in kg.get_outgoing_relations(entity):
                related.append(target)

            # Get incoming relations
            for rel, source in kg.get_incoming_relations(entity):
                related.append(source)

            if related:
                entity_sets.append(set(related))

        print(
            f"--- Inside Intersection: basic_results={basic_results}, num_sets={len(entity_sets)} ---"
        )  # 新增

        if len(entity_sets) < 2:  # 如果無法形成多個集合
            print(
                f"--- Intersection fallback: returning basic_results={basic_results} ---"
            )
            return basic_results  # 直接返回第一個集合的結果

        # --- 如果能正確找到多個集合 (例如，問題是 \"actors in MovieA AND MovieB\") ---
        # (假設 entities_in_question 能正確提取並生成 entity_sets[1:])
        intersection = entity_sets[0]
        for s in entity_sets[1:]:
            intersection &= s
        intersection_result = list(intersection)
        print(f"--- Intersection result: {intersection_result} ---")
        return intersection_result


class KGQAPipeline:
    """End-to-end pipeline for knowledge graph question answering."""

    def __init__(
        self, knowledge_graph: KnowledgeGraph, model_name: str = "gpt-3.5-turbo"
    ):
        """
        Initialize the KGQA pipeline.

        Args:
            knowledge_graph: The knowledge graph
            model_name: Name of the OpenAI model to use
        """
        self.kg = knowledge_graph
        self.model_name = model_name
        self.decomposer = LLMQuestionDecomposer(
            model_name=model_name, kg=knowledge_graph
        )
        self.action_selector = LLMActionSelector(
            knowledge_graph=knowledge_graph, model_name=model_name
        )

    def extract_topic_entity(self, question: str) -> str:
        """
        Extract the topic entity from a question using LLM.

        Args:
            question: The question

        Returns:
            Extracted topic entity
        """
        # If the topic entity is already in brackets, extract it
        topic_entity_match = re.search(r"\[(.*?)\]", question)
        if topic_entity_match:
            extracted = topic_entity_match.group(1)
            # Try to find the exact entity in the knowledge graph
            return self.kg.improve_entity_linking(extracted)

        # Otherwise use LLM to extract it
        prompt = f"""From the following question about a movie knowledge graph, identify the main entity that the question is about.

Question: {question}

The answer should be a single entity name that exists in a movie knowledge graph, such as a movie title, actor name, director name, etc.

Return only the entity name without any additional text.
"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts entities from questions.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=50,
            )

            extracted_entity = response.choices[0].message.content.strip()

            # Use improved entity linking
            linked_entity = self.kg.improve_entity_linking(extracted_entity)
            if linked_entity != extracted_entity:
                print(
                    f"Linked entity '{extracted_entity}' to '{linked_entity}' in knowledge graph"
                )

            return linked_entity

        except Exception as e:
            print(f"Error in topic entity extraction: {e}")
            # Try simple extraction as fallback
            words = question.split()
            for word in words:
                clean_word = re.sub(r"[^\w\s]", "", word).strip()
                if len(clean_word) > 3:  # Skip short words
                    entity = self.kg.improve_entity_linking(clean_word)
                    if entity:
                        return entity
            return None

    def estimate_hop_count(self, question: str) -> int:
        """
        Estimate the number of hops needed to answer a question using LLM.

        Args:
            question: The question

        Returns:
            Estimated number of hops (1, 2, or 3)
        """
        # First, use heuristic rules based on question structure
        lower_question = question.lower()

        # Check for common patterns suggesting multi-hop questions
        if any(
            pattern in lower_question
            for pattern in [
                "actor",
                "appear",
                "star",
                "genre",
                "movies directed by",
                "movies starring",
                "actors who",
                "directors of",
                "genre of movies",
            ]
        ):
            # Check for 3-hop patterns
            if any(
                pattern in lower_question
                for pattern in [
                    "actors who appeared in",
                    "genre of movies starring actors",
                    "directors of movies starring",
                    "actors from",
                ]
            ):
                return 3

            # Check for 2-hop patterns
            if any(
                pattern in lower_question
                for pattern in [
                    "directed by",
                    "starring",
                    "appeared in",
                    "acted in",
                    "genre of",
                    "movies did",
                    "movies by",
                ]
            ):
                return 2

        # Default to LLM-based estimation if heuristics don't match
        prompt = f"""Given a question about a movie knowledge graph, estimate how many steps (hops) would be needed to answer it.

Question: {question}

For example:
- "Who directed The Matrix?" requires 1 hop: The Matrix --[directed_by]--> Director
- "Which actors appeared in movies directed by Christopher Nolan?" requires 2 hops: Christopher Nolan --[directed]--> Movies --[has_actor]--> Actors
- "What is the genre of movies starring actors who appeared in The Matrix?" requires 3 hops: The Matrix --[has_actor]--> Actors --[starred_in]--> Movies --[has_genre]--> Genre

Return only the number (1, 2, or 3) without any additional text.
"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that analyzes question complexity.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=10,
            )

            result = response.choices[0].message.content.strip()

            # Extract the hop count
            hop_match = re.search(r"[1-3]", result)
            if hop_match:
                return int(hop_match.group(0))
            else:
                print(
                    f"Warning: Could not extract hop count from response: '{result}'. Defaulting to 1."
                )
                return 1

        except Exception as e:
            print(f"Error in hop count estimation: {e}")
            return 1  # Default to 1 hop

    def validate_intermediate_results(
        self, subquestion: str, current_entities: List[str], result_entities: List[str]
    ) -> bool:
        """
        Validate that the intermediate results make sense for the subquestion.

        Args:
            subquestion: The subquestion
            current_entities: Input entities
            result_entities: Output entities

        Returns:
            True if results seem valid, False otherwise
        """
        # 1. Check if we have any results
        if not result_entities:
            return False

        # 2. Check if results are just repeating the input
        if set(result_entities) == set(current_entities):
            return False

        # 3. Use LLM to validate if results make sense
        if len(result_entities) > 5:
            # Too many results to validate individually
            return True

        prompt = f"""Given a question and some potential answers, rate how likely these answers are to be correct (0-10 scale).

Question: {subquestion}
Starting with entities: {", ".join(current_entities)}
Potential answers: {", ".join(result_entities)}

Rate each answer on a scale of 0-10, where 0 means completely irrelevant and 10 means definitely correct.
"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that evaluates answers.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=150,
            )

            validation_text = response.choices[0].message.content.strip()

            # Extract ratings
            ratings = []
            for entity in result_entities:
                # Look for a rating of this entity
                rating_match = re.search(
                    rf"{re.escape(entity)}.*?(\d+)(?:/10)?",
                    validation_text,
                    re.IGNORECASE | re.DOTALL,
                )
                if rating_match:
                    rating = int(rating_match.group(1))
                    ratings.append(rating)

            # If we found ratings, check if they're high enough
            if ratings:
                avg_rating = sum(ratings) / len(ratings)
                return avg_rating >= 5

            # No ratings found
            return True

        except Exception as e:
            print(f"Error in result validation: {e}")
            return True  # Default to accepting results

    def try_alternative_path(self, question: str, entity: str) -> List[str]:
        """
        Try an alternative path when the primary path fails.

        Args:
            question: The question
            entity: Starting entity

        Returns:
            List of alternative results
        """
        print(f"Trying alternative path for entity: {entity}")

        # 1. Try incoming relations instead of outgoing
        incoming = self.kg.get_incoming_relations(entity)
        if incoming:
            print(f"Found {len(incoming)} incoming relations")
            # Pick most relevant incoming relation
            for relation, source in incoming:
                if (
                    relation.lower() in question.lower()
                    or source.lower() in question.lower()
                ):
                    print(
                        f"Following incoming relation: {source} --[{relation}]--> {entity}"
                    )
                    # Now try to continue from the source
                    action_type = self.action_selector.select_action_type(question)
                    results = self.action_selector.execute_action(
                        action_type, question, [source], self.kg
                    )
                    if results:
                        return results

        # 2. Try alternate entity linking
        words = question.split()
        for word in words:
            if len(word) > 3 and word.lower() not in [
                "what",
                "which",
                "where",
                "when",
                "who",
                "how",
                "did",
                "does",
            ]:
                alternate_entity = self.kg.improve_entity_linking(word)
                if alternate_entity and alternate_entity != entity:
                    print(f"Trying alternate entity: {alternate_entity}")
                    action_type = self.action_selector.select_action_type(question)
                    results = self.action_selector.execute_action(
                        action_type, question, [alternate_entity], self.kg
                    )
                    if results:
                        return results

        # No alternatives worked
        return []

    def answer_question(
        self, question: str, topic_entity: str = None, hop_count: int = None
    ) -> List[str]:
        """
        Answer a question using the knowledge graph.

        Args:
            question: The question
            topic_entity: Topic entity (if already extracted)
            hop_count: Number of hops (if already estimated)

        Returns:
            List of answer entities
        """
        # Extract topic entity if not provided
        if topic_entity is None:
            topic_entity = self.extract_topic_entity(question)

        if topic_entity is None:
            print("Error: Could not extract topic entity from question")
            return []

        # Use improved entity linking
        exact_topic_entity = self.kg.improve_entity_linking(topic_entity)

        # Estimate hop count if not provided
        if hop_count is None:
            hop_count = self.estimate_hop_count(question)

        print(f"Answering question: '{question}'")
        print(f"Topic entity: {exact_topic_entity}")
        print(f"Estimated hop count: {hop_count}")

        # Decompose question into subquestions
        subquestions = self.decomposer.decompose_question(
            question, exact_topic_entity, hop_count
        )
        print(f"Decomposed into {len(subquestions)} subquestions: {subquestions}")

        # Answer each subquestion in sequence
        current_entities = [exact_topic_entity]
        path_history = []

        for i, subquestion in enumerate(subquestions):
            print(f"Processing subquestion {i + 1}: '{subquestion}'")
            print(f"Current entities: {current_entities}")

            # Select action type
            action_type = self.action_selector.select_action_type(subquestion)
            print(f"Selected action type: {action_type}")

            # Execute action
            result_entities = self.action_selector.execute_action(
                action_type, subquestion, current_entities, self.kg
            )
            print(f"Results after action: {result_entities}")

            # Validate results
            if not result_entities or not self.validate_intermediate_results(
                subquestion, current_entities, result_entities
            ):
                print(f"Warning: No valid results found for subquestion {i + 1}")

                # Try alternative approaches for each current entity
                alternative_results = []
                for entity in current_entities:
                    alt_results = self.try_alternative_path(subquestion, entity)
                    alternative_results.extend(alt_results)

                if alternative_results:
                    print(f"Found alternative results: {alternative_results}")
                    result_entities = alternative_results
                elif i == len(subquestions) - 1:
                    # If this is the final subquestion, return current entities as fallback
                    print("Using current entities as fallback for final answer")
                    return current_entities
                else:
                    # If we can't recover, break the loop
                    print("Could not recover from failed subquestion")
                    break

            # Update current entities for next hop
            current_entities = result_entities
            path_history.append((subquestion, action_type, result_entities))

        return current_entities

    def evaluate(
        self, dataset: MetaQADataset, split: str = "test", max_samples: int = None
    ) -> Dict:
        """
        Evaluate the pipeline on a dataset.

        Args:
            dataset: The dataset to evaluate on
            split: Data split to use ('train', 'dev', or 'test')
            max_samples: Maximum number of samples to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        results = {
            "total": 0,
            "correct": 0,
            "partial": 0,
            "wrong": 0,
            "accuracy": 0.0,
            "mrr": 0.0,
            "per_hop": {
                1: {"total": 0, "correct": 0},
                2: {"total": 0, "correct": 0},
                3: {"total": 0, "correct": 0},
            },
        }

        examples = dataset.qa_pairs[split]
        if max_samples is not None and max_samples < len(examples):
            examples = examples[:max_samples]

        print(f"Evaluating on {len(examples)} examples from {split} set")

        for i, example in enumerate(examples):
            print(f"\nExample {i + 1}/{len(examples)}")
            question = example["question"]
            gold_answers = example["answers"]
            topic_entity = example["topic_entity"]
            hop = example["hop"]

            # Answer the question
            predicted_answers = self.answer_question(question, topic_entity, hop)

            # Compute metrics
            results["total"] += 1
            results["per_hop"][hop]["total"] += 1

            # Check for exact matches
            exact_matches = set(predicted_answers).intersection(set(gold_answers))
            if exact_matches:
                if len(exact_matches) == len(gold_answers):
                    results["correct"] += 1
                    results["per_hop"][hop]["correct"] += 1
                else:
                    results["partial"] += 1
            else:
                results["wrong"] += 1

            # Compute MRR (Mean Reciprocal Rank)
            reciprocal_rank = 0
            for i, pred in enumerate(
                predicted_answers[:10]
            ):  # Consider top 10 predictions
                if pred in gold_answers:
                    reciprocal_rank = 1.0 / (i + 1)
                    break
            results["mrr"] += reciprocal_rank

            print(f"Question: {question}")
            print(f"Gold answers: {gold_answers}")
            print(f"Predicted answers: {predicted_answers}")
            print(
                f"Status: {'Correct' if exact_matches and len(exact_matches) == len(gold_answers) else 'Partial' if exact_matches else 'Wrong'}"
            )

        # Compute final metrics
        if results["total"] > 0:
            results["accuracy"] = results["correct"] / results["total"]
            results["mrr"] = results["mrr"] / results["total"]

            # Per-hop accuracy
            for hop in [1, 2, 3]:
                if results["per_hop"][hop]["total"] > 0:
                    results["per_hop"][hop]["accuracy"] = (
                        results["per_hop"][hop]["correct"]
                        / results["per_hop"][hop]["total"]
                    )
                else:
                    results["per_hop"][hop]["accuracy"] = 0

        print(f"\nEvaluation results on {split} set:")
        print(f"Total examples: {results['total']}")
        print(f"Correct: {results['correct']} ({results['accuracy'] * 100:.2f}%)")
        print(
            f"Partial: {results['partial']} ({results['partial'] / results['total'] * 100:.2f}%)"
        )
        print(
            f"Wrong: {results['wrong']} ({results['wrong'] / results['total'] * 100:.2f}%)"
        )
        print(f"MRR: {results['mrr']:.4f}")

        # Print per-hop results
        for hop in [1, 2, 3]:
            hop_total = results["per_hop"][hop]["total"]
            if hop_total > 0:
                hop_acc = results["per_hop"][hop]["accuracy"] * 100
                print(
                    f"{hop}-hop: {results['per_hop'][hop]['correct']}/{hop_total} ({hop_acc:.2f}%)"
                )

        return results


def main(metaqa_base_dir, hop=2, max_samples=10):
    """
    Run the KGQA pipeline on MetaQA dataset.

    Args:
        metaqa_base_dir: Base directory for MetaQA dataset
        hop: Number of hops to evaluate (1, 2, or 3)
        max_samples: Maximum number of samples to evaluate
    """
    # Load knowledge graph
    kb_path = os.path.join(metaqa_base_dir, "kb.txt")
    kg = KnowledgeGraph(kb_path)

    # Load dataset
    dataset = MetaQADataset(metaqa_base_dir, hop=hop)

    # Initialize pipeline
    pipeline = KGQAPipeline(kg)

    # Evaluate pipeline
    results = pipeline.evaluate(dataset, split="test", max_samples=max_samples)

    # Save results
    result_path = f"evaluation_results_hop{hop}.json"
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {result_path}")


if __name__ == "__main__":
    # Example usage
    metaqa_base_dir = "./data"  # Replace with actual path
    main(metaqa_base_dir, hop=2, max_samples=1)
