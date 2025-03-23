"""
Knowledge Graph Environment for interaction between RL agent and the knowledge graph
"""

import torch
import numpy as np
import random
from collections import defaultdict
import config
import json
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("kg_env.log"), logging.StreamHandler()],
)
logger = logging.getLogger("KGEnv")


class KnowledgeGraphEnv:
    """Knowledge Graph Environment class, handling interaction between agent and KG"""

    def __init__(
        self,
        kg_dict,
        entities,
        relations,
        entity_to_idx,
        relation_to_idx,
        entity_variants=None,
    ):
        """
        Initialize the knowledge graph environment

        Parameters:
            kg_dict (dict): Knowledge graph dictionary {entity: [(relation, target_entity), ...]}
            entities (list): List of entities
            relations (list): List of relations
            entity_to_idx (dict): Mapping from entity to index
            relation_to_idx (dict): Mapping from relation to index
            entity_variants (dict): Dictionary for entity name variants
        """
        self.kg_dict = kg_dict
        self.entities = entities
        self.relations = relations
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        self.idx_to_entity = {v: k for k, v in entity_to_idx.items()}
        self.idx_to_relation = {v: k for k, v in relation_to_idx.items()}
        self.entity_variants = entity_variants or {}

        # Map action types to processing functions
        self.action_processors = {
            "basic": self._process_basic_action,
            "filter": self._process_filter_action,
            "union": self._process_union_action,
            "aggregation": self._process_aggregation_action,
            "ordinal": self._process_ordinal_action,
            "stop": self._process_stop_action,
        }

        # Statistics
        self.stats = defaultdict(int)
        self.action_history = []

        # Debug information
        self.debug_info = []

    def reset(self, question, question_entity, question_type=None, question_focus=None):
        """
        Reset the environment to start a new question

        Parameters:
            question (str): The question text
            question_entity (str): The main entity mentioned in the question
            question_type (str): Type of question (e.g., "who", "what")
            question_focus (str): Focus of the question (e.g., "director", "actor")

        Returns:
            dict: Initial state
        """
        # Parse the entity
        self.original_question_entity = question_entity
        if question_entity in self.entity_variants:
            resolved_entity = self.entity_variants[question_entity]
            match_score = 1.0
            is_exact = True
        else:
            # Attempt to find in the entity list
            from difflib import SequenceMatcher

            best_score = 0
            best_entity = None
            for entity in self.entities:
                score = SequenceMatcher(
                    None, question_entity.lower(), entity.lower()
                ).ratio()
                if score > best_score:
                    best_score = score
                    best_entity = entity

            if best_score >= config.ENTITY_MATCH_THRESHOLD:
                resolved_entity = best_entity
                match_score = best_score
                is_exact = False
                logger.info(
                    f"Entity '{question_entity}' fuzzy matched as '{resolved_entity}', score: {match_score:.2f}"
                )
            else:
                resolved_entity = question_entity
                match_score = 0.0
                is_exact = False
                logger.warning(
                    f"Entity '{question_entity}' did not find a close match in the knowledge graph"
                )

        # Set the parsed entity as the current entity
        self.question_entity = resolved_entity
        self.current_entities = [resolved_entity]

        # Save question information
        self.question = question
        self.question_type = question_type
        self.question_focus = question_focus

        # Reset state
        self.path_history = []
        self.done = False
        self.step_count = 0
        self.debug_info = []
        self.action_history = []

        # Record entity resolution information
        self.debug_info.append(
            {
                "event": "entity_resolution",
                "original_entity": question_entity,
                "resolved_entity": resolved_entity,
                "match_score": match_score,
                "is_exact_match": is_exact,
            }
        )

        # Check if the entity has outgoing edges
        has_outgoing_edges = (
            resolved_entity in self.kg_dict and len(self.kg_dict[resolved_entity]) > 0
        )
        self.debug_info.append(
            {
                "event": "entity_check",
                "entity": resolved_entity,
                "has_outgoing_edges": has_outgoing_edges,
                "edge_count": len(self.kg_dict.get(resolved_entity, [])),
            }
        )

        if resolved_entity in self.kg_dict:
            logger.info(
                f"Entity '{resolved_entity}' found in KG with {len(self.kg_dict[resolved_entity])} outgoing edges"
            )
        else:
            logger.warning(f"Entity '{resolved_entity}' has no outgoing edges in KG")

        return self._get_state()

    def step(self, action_type, action_params=None):
        """
        Execute one step/action

        Parameters:
            action_type (str): The type of action to perform
            action_params: Parameters for the action

        Returns:
            tuple: (next_state, reward, done, info)
        """
        if self.done:
            return (
                self._get_state(),
                0,
                True,
                {"error": "Environment has already finished"},
            )

        self.step_count += 1

        # Record action
        self.action_history.append((action_type, action_params))

        # Check if action type is valid
        if action_type not in self.action_processors:
            self.stats["invalid_action_type"] += 1
            logger.warning(f"Invalid action type: {action_type}")
            return (
                self._get_state(),
                -0.1,
                False,
                {"error": f"Invalid action type: {action_type}"},
            )

        # Save the current state before processing the action
        prev_entities = self.current_entities.copy()

        # Process the action
        reward, done, info = self.action_processors[action_type](action_params)

        # Record debug information
        self.debug_info.append(
            {
                "event": "action_execution",
                "step": self.step_count,
                "action_type": action_type,
                "action_params": str(action_params),
                "prev_entities": prev_entities,
                "next_entities": self.current_entities,
                "reward": reward,
                "done": done,
                "info": info,
            }
        )

        # Force finish if maximum steps reached
        if self.step_count >= config.MAX_STEPS:
            done = True
            info["max_steps_reached"] = True
            logger.info(f"Maximum steps reached ({config.MAX_STEPS}), force finish")

        self.done = done
        return self._get_state(), reward, done, info

    def _process_basic_action(self, relation):
        """
        Process a basic path action

        Parameters:
            relation (str): The relation to follow

        Returns:
            tuple: (reward, done, info)
        """
        if relation not in self.relation_to_idx:
            self.stats["unknown_relation"] += 1
            logger.warning(f"Unknown relation: {relation}")
            return -0.1, False, {"error": f"Unknown relation: {relation}"}

        new_entities = []

        for entity in self.current_entities:
            if entity in self.kg_dict:
                for rel, obj in self.kg_dict[entity]:
                    if rel == relation:
                        new_entities.append(obj)

        if new_entities:
            self.current_entities = new_entities
            self.path_history.append(("basic", relation))

            # A higher reward is given when fewer entities are found
            if len(new_entities) <= 5:
                reward = 0.2
            else:
                reward = 0.1

            logger.info(
                f"Moved along relation '{relation}', found {len(new_entities)} new entities"
            )
            return reward, False, {"new_entities_count": len(new_entities)}
        else:
            self.stats["no_path_found"] += 1
            logger.warning(f"No path found along relation '{relation}'")
            return -0.1, False, {"error": "No path found"}

    def _process_filter_action(self, filter_params):
        """
        Process a filter action

        Parameters:
            filter_params (tuple): (filter_relation, filter_value, operator)

        Returns:
            tuple: (reward, done, info)
        """
        if not isinstance(filter_params, tuple) or len(filter_params) != 3:
            self.stats["invalid_filter_params"] += 1
            logger.warning(f"Invalid filter parameters: {filter_params}")
            return -0.1, False, {"error": "Invalid filter parameters"}

        filter_relation, filter_value, operator = filter_params

        if filter_relation not in self.relation_to_idx:
            self.stats["unknown_filter_relation"] += 1
            logger.warning(f"Unknown filter relation: {filter_relation}")
            return -0.1, False, {"error": f"Unknown filter relation: {filter_relation}"}

        filtered_entities = []

        for entity in self.current_entities:
            if entity in self.kg_dict:
                values = []
                for rel, obj in self.kg_dict[entity]:
                    if rel == filter_relation:
                        values.append(obj)

                # Filter based on the operator
                if operator == "=" and filter_value in values:
                    filtered_entities.append(entity)
                elif operator == "!=" and filter_value not in values:
                    filtered_entities.append(entity)
                elif operator == ">" and any(
                    self._compare_values(v, filter_value, ">") for v in values
                ):
                    filtered_entities.append(entity)
                elif operator == ">=" and any(
                    self._compare_values(v, filter_value, ">=") for v in values
                ):
                    filtered_entities.append(entity)
                elif operator == "<" and any(
                    self._compare_values(v, filter_value, "<") for v in values
                ):
                    filtered_entities.append(entity)
                elif operator == "<=" and any(
                    self._compare_values(v, filter_value, "<=") for v in values
                ):
                    filtered_entities.append(entity)

        if filtered_entities:
            self.current_entities = filtered_entities
            self.path_history.append(("filter", filter_params))
            logger.info(
                f"Applied filter '{filter_relation} {operator} {filter_value}', retained {len(filtered_entities)} entities"
            )
            return 0.1, False, {"filtered_entities_count": len(filtered_entities)}
        else:
            self.stats["no_entities_after_filter"] += 1
            logger.warning(
                f"After applying filter '{filter_relation} {operator} {filter_value}', no entities remain"
            )
            return -0.1, False, {"error": "No entities remain after filtering"}

    def _process_union_action(self, union_params):
        """
        Process a union action

        Parameters:
            union_params (tuple): (relation1, relation2)

        Returns:
            tuple: (reward, done, info)
        """
        if not isinstance(union_params, tuple) or len(union_params) != 2:
            self.stats["invalid_union_params"] += 1
            logger.warning(f"Invalid union parameters: {union_params}")
            return -0.1, False, {"error": "Invalid union parameters"}

        relation1, relation2 = union_params

        if (
            relation1 not in self.relation_to_idx
            or relation2 not in self.relation_to_idx
        ):
            self.stats["unknown_union_relation"] += 1
            logger.warning(f"Unknown union relation: {relation1} or {relation2}")
            return -0.1, False, {"error": "Unknown union relation"}

        union_entities = set()

        # Process the first relation
        for entity in self.current_entities:
            if entity in self.kg_dict:
                for rel, obj in self.kg_dict[entity]:
                    if rel == relation1:
                        union_entities.add(obj)

        # Process the second relation
        for entity in self.current_entities:
            if entity in self.kg_dict:
                for rel, obj in self.kg_dict[entity]:
                    if rel == relation2:
                        union_entities.add(obj)

        if union_entities:
            self.current_entities = list(union_entities)
            self.path_history.append(("union", union_params))
            logger.info(
                f"Union of relations '{relation1}' and '{relation2}' resulted in {len(union_entities)} entities"
            )
            return 0.1, False, {"union_entities_count": len(union_entities)}
        else:
            self.stats["no_entities_after_union"] += 1
            logger.warning(
                f"Union of relations '{relation1}' and '{relation2}' resulted in no entities"
            )
            return -0.1, False, {"error": "No entities after union"}

    def _process_aggregation_action(self, agg_type):
        """
        Process an aggregation action

        Parameters:
            agg_type (str): Aggregation type ('count', 'min', 'max', 'sum', 'avg')

        Returns:
            tuple: (reward, done, info)
        """
        if agg_type not in ["count", "min", "max", "sum", "avg"]:
            self.stats["invalid_aggregation_type"] += 1
            logger.warning(f"Invalid aggregation type: {agg_type}")
            return -0.1, False, {"error": f"Invalid aggregation type: {agg_type}"}

        # Check if there are entities to aggregate
        if not self.current_entities:
            self.stats["no_entities_to_aggregate"] += 1
            logger.warning("No entities to aggregate")
            return -0.1, False, {"error": "No entities to aggregate"}

        # Check if the question type is suitable for aggregation
        is_count_question = (
            self.question_type == "howmany" or "how many" in self.question.lower()
        )

        # Give a higher reward for 'count' aggregation on count questions
        if is_count_question and agg_type == "count":
            reward_multiplier = 2.0
        else:
            reward_multiplier = 1.0

        # Perform aggregation
        if agg_type == "count":
            result = str(len(self.current_entities))
            logger.info(f"Performed count aggregation, result: {result}")
        elif agg_type in ["min", "max", "sum", "avg"]:
            # Attempt to convert entities to numbers
            numeric_values = []
            for entity in self.current_entities:
                try:
                    numeric_values.append(float(entity))
                except (ValueError, TypeError):
                    pass

            if not numeric_values:
                self.stats["no_numeric_entities"] += 1
                logger.warning("No numeric entities available for aggregation")
                return (
                    -0.1,
                    False,
                    {"error": "No numeric entities available for aggregation"},
                )

            if agg_type == "min":
                result = str(min(numeric_values))
            elif agg_type == "max":
                result = str(max(numeric_values))
            elif agg_type == "sum":
                result = str(sum(numeric_values))
            elif agg_type == "avg":
                result = str(sum(numeric_values) / len(numeric_values))

            logger.info(f"Performed {agg_type} aggregation, result: {result}")

        # Replace current entities with the aggregation result
        self.current_entities = [result]
        self.path_history.append(("aggregation", agg_type))

        # Base reward
        base_reward = 0.1

        # Adjust reward based on question type
        if self.question_type == "howmany" and agg_type == "count":
            # Using count for "how many" questions is appropriate
            adjusted_reward = base_reward * 2
            logger.info("Aggregation matches the question type, giving double reward")
        else:
            # Base reward for other cases
            adjusted_reward = base_reward

        return (
            adjusted_reward * reward_multiplier,
            False,
            {"aggregation_result": result},
        )

    def _process_ordinal_action(self, ordinal_params):
        """
        Process an ordinal action

        Parameters:
            ordinal_params (tuple): (sort_relation, order_type, position)

        Returns:
            tuple: (reward, done, info)
        """
        if not isinstance(ordinal_params, tuple) or len(ordinal_params) != 3:
            self.stats["invalid_ordinal_params"] += 1
            logger.warning(f"Invalid ordinal parameters: {ordinal_params}")
            return -0.1, False, {"error": "Invalid ordinal parameters"}

        sort_relation, order_type, position = ordinal_params

        if sort_relation not in self.relation_to_idx:
            self.stats["unknown_ordinal_relation"] += 1
            logger.warning(f"Unknown sorting relation: {sort_relation}")
            return -0.1, False, {"error": f"Unknown sorting relation: {sort_relation}"}

        if order_type not in ["asc", "desc"]:
            self.stats["invalid_order_type"] += 1
            logger.warning(f"Invalid order type: {order_type}")
            return -0.1, False, {"error": f"Invalid order type: {order_type}"}

        try:
            position = int(position)
        except (ValueError, TypeError):
            self.stats["invalid_position"] += 1
            logger.warning(f"Invalid position: {position}")
            return -0.1, False, {"error": f"Invalid position: {position}"}

        # Get sorting values for each entity
        entity_values = []
        for entity in self.current_entities:
            if entity in self.kg_dict:
                values = []
                for rel, obj in self.kg_dict[entity]:
                    if rel == sort_relation:
                        try:
                            values.append((entity, float(obj)))
                        except (ValueError, TypeError):
                            values.append((entity, obj))

                if values:
                    # If multiple values exist, take the first one
                    entity_values.append(values[0])

        if not entity_values:
            self.stats["no_entities_to_sort"] += 1
            logger.warning("No entities available for sorting")
            return -0.1, False, {"error": "No entities available for sorting"}

        # Sorting
        try:
            sorted_entities = sorted(
                entity_values, key=lambda x: x[1], reverse=(order_type == "desc")
            )
        except TypeError:
            # If values cannot be compared, use string sorting
            sorted_entities = sorted(
                entity_values, key=lambda x: str(x[1]), reverse=(order_type == "desc")
            )

        # Check if the position is valid
        if position < 0:
            position = len(sorted_entities) + position  # Support negative index

        if position < 0 or position >= len(sorted_entities):
            self.stats["invalid_position_range"] += 1
            logger.warning(f"Position out of range: {position}")
            return -0.1, False, {"error": f"Position out of range: {position}"}

        # Select the entity at the specified position
        self.current_entities = [sorted_entities[position][0]]
        self.path_history.append(("ordinal", ordinal_params))

        logger.info(
            f"Sorted entities by {sort_relation} ({order_type}); selected entity at position {position}: {self.current_entities[0]}"
        )

        return 0.2, False, {"selected_entity": self.current_entities[0]}

    def _process_stop_action(self, _):
        """
        Process a stop action

        Parameters:
            _ (None): Not used

        Returns:
            tuple: (reward, done, info)
        """
        # Check if the current state is suitable to stop
        if not self.current_entities:
            logger.warning("Stopping with no entities available")
            return 0, True, {"final_entities": []}

        # Stopping at the first step is usually not a good choice
        if (
            self.step_count == 1
            and len(self.current_entities) == 1
            and self.current_entities[0] == self.question_entity
        ):
            logger.warning("Stopping at the first step may not be a good choice")
            # For 'who' type questions, stopping at the main entity is generally wrong
            if self.question_type == "who":
                return (
                    -0.2,
                    True,
                    {
                        "final_entities": self.current_entities,
                        "warning": "Stopping at the first step may not be a good choice",
                    },
                )

        logger.info(f"Stop action executed, final entities: {self.current_entities}")
        return 0, True, {"final_entities": self.current_entities}

    def _compare_values(self, v1, v2, op):
        """
        Compare two values

        Parameters:
            v1: The first value
            v2: The second value
            op (str): Comparison operator

        Returns:
            bool: Comparison result
        """
        # Attempt to convert values to numbers for comparison
        try:
            v1_num = float(v1)
            v2_num = float(v2)

            if op == ">":
                return v1_num > v2_num
            elif op == ">=":
                return v1_num >= v2_num
            elif op == "<":
                return v1_num < v2_num
            elif op == "<=":
                return v1_num <= v2_num
        except (ValueError, TypeError):
            # If conversion fails, use string comparison
            if op == ">":
                return v1 > v2
            elif op == ">=":
                return v1 >= v2
            elif op == "<":
                return v1 < v2
            elif op == "<=":
                return v1 <= v2

        return False

    def _get_state(self):
        """
        Get the current state

        Returns:
            dict: The current state
        """
        return {
            "question": self.question,
            "question_type": getattr(self, "question_type", None),
            "question_focus": getattr(self, "question_focus", None),
            "current_entities": self.current_entities,
            "path_history": self.path_history,
            "step_count": self.step_count,
            "original_entity": self.original_question_entity,
            "resolved_entity": self.question_entity,
        }

    def get_valid_actions(self):
        """
        Get valid actions in the current state

        Returns:
            list: List of valid actions [(action_type, action_params), ...]
        """
        valid_actions = []

        # In the first step and if the question type is "who directed" or "who starred",
        # directly add the related basic actions
        if self.step_count == 0 and hasattr(self, "question_focus"):
            if self.question_focus == "director":
                # Find all relations related to "director"
                for rel in self.relations:
                    if "direct" in rel.lower():
                        valid_actions.append(("basic", rel))

            elif self.question_focus == "actor":
                # Find all relations related to "actor"
                for rel in self.relations:
                    if any(
                        term in rel.lower() for term in ["star", "act", "play", "cast"]
                    ):
                        valid_actions.append(("basic", rel))

        # Stop action is always allowed, but on the first step we might wish to avoid it
        if (
            self.step_count > 0
            or not hasattr(self, "question_focus")
            or self.question_focus not in ["director", "actor", "person"]
        ):
            valid_actions.append(("stop", None))

        # If no entities are present, no further actions can be executed
        if not self.current_entities:
            return valid_actions

        # Basic path action
        for entity in self.current_entities[
            :5
        ]:  # Limit number of entities to avoid huge action space
            if entity in self.kg_dict:
                seen_relations = set()  # Avoid duplicate relations
                for rel, _ in self.kg_dict[entity]:
                    if rel not in seen_relations:
                        valid_actions.append(("basic", rel))
                        seen_relations.add(rel)

        # Only consider filtering and aggregation when there are multiple entities
        if len(self.current_entities) > 1:
            # Filter actions
            # Only add some potentially useful filters
            common_years = ["1997", "2000", "2010", "2020"]
            common_countries = ["USA", "UK", "France", "Japan"]
            common_ratings = ["PG", "PG-13", "R"]

            if self.step_count > 0:  # Consider filtering only after the first step
                # Add filter actions based on domain knowledge
                if hasattr(self, "question_type"):
                    if (
                        "year" in self.question.lower()
                        or "when" in self.question.lower()
                    ):
                        # For year-related questions
                        filter_relations = [
                            r
                            for r in self.relations
                            if any(
                                term in r.lower()
                                for term in ["year", "date", "release"]
                            )
                        ]
                        for rel in filter_relations[:3]:  # Limit to 3 relations
                            for year in common_years:
                                valid_actions.append(("filter", (rel, year, "=")))

                    elif (
                        "where" in self.question.lower()
                        or "country" in self.question.lower()
                    ):
                        # For location-related questions
                        filter_relations = [
                            r
                            for r in self.relations
                            if any(
                                term in r.lower()
                                for term in ["country", "location", "place"]
                            )
                        ]
                        for rel in filter_relations[:3]:
                            for country in common_countries:
                                valid_actions.append(("filter", (rel, country, "=")))

            # Aggregation actions
            if "how many" in self.question.lower() or "count" in self.question.lower():
                valid_actions.append(("aggregation", "count"))

            # Other aggregation actions
            has_numeric_entities = any(
                self._is_numeric(e) for e in self.current_entities
            )
            if has_numeric_entities:
                for agg_type in ["min", "max", "avg"]:
                    valid_actions.append(("aggregation", agg_type))

        # Ordinal action
        if len(self.current_entities) > 1 and self.step_count > 0:
            if "first" in self.question.lower() or "latest" in self.question.lower():
                # Find relations suitable for sorting
                sort_relations = [
                    r
                    for r in self.relations
                    if any(
                        term in r.lower() for term in ["year", "date", "time", "rating"]
                    )
                ]
                for rel in sort_relations[:2]:  # Limit to 2 relations
                    for order in ["asc", "desc"]:
                        valid_actions.append(
                            ("ordinal", (rel, order, 0))
                        )  # First element
                        valid_actions.append(
                            ("ordinal", (rel, order, -1))
                        )  # Last element

        # Union action (more advanced, use sparingly)
        if self.step_count > 0 and "and" in self.question.lower():
            # Only add union action if the question contains "and"
            relations_seen = set()
            for entity in self.current_entities[:3]:
                if entity in self.kg_dict:
                    for rel, _ in self.kg_dict[entity][:5]:
                        relations_seen.add(rel)

            # Add up to 3 union actions
            relations_list = list(relations_seen)
            if len(relations_list) >= 2:
                for i in range(min(3, len(relations_list))):
                    for j in range(i + 1, min(4, len(relations_list))):
                        valid_actions.append(
                            ("union", (relations_list[i], relations_list[j]))
                        )

        return valid_actions

    def _is_numeric(self, value):
        """Check if the value is numeric"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def calculate_reward(self, true_answers):
        """
        Calculate the final reward

        Parameters:
            true_answers (list): List of true answers

        Returns:
            float: Reward value
        """
        predicted_answers = set(self.current_entities)
        true_answers_set = set(true_answers)

        # Calculate precision, recall, and F1 score
        if len(predicted_answers) == 0:
            precision = 0
        else:
            precision = len(predicted_answers.intersection(true_answers_set)) / len(
                predicted_answers
            )

        if len(true_answers_set) == 0:
            recall = 0
        else:
            recall = len(predicted_answers.intersection(true_answers_set)) / len(
                true_answers_set
            )

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # Use F1 as the reward
        reward = f1

        # Give extra reward if the prediction is completely correct
        if predicted_answers == true_answers_set and len(predicted_answers) > 0:
            reward += 0.5

        # Record evaluation information
        self.debug_info.append(
            {
                "event": "reward_calculation",
                "predicted_answers": list(predicted_answers),
                "true_answers": list(true_answers_set),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "reward": reward,
            }
        )

        logger.info(
            f"Reward calculation: precision={precision:.4f}, recall={recall:.4f}, F1={f1:.4f}, final reward={reward:.4f}"
        )

        return reward

    def get_debug_info(self):
        """
        Get debugging information

        Returns:
            list: List of debug information entries
        """
        return self.debug_info

    def get_stats(self):
        """
        Get environment statistics

        Returns:
            dict: Environment statistics
        """
        return dict(self.stats)

    def save_debug_info(self, path="kg_env_debug.json"):
        """
        Save debugging information to a file

        Parameters:
            path (str): Path to save the debugging information
        """
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "debug_info": self.debug_info,
                    "stats": self.get_stats(),
                    "action_history": self.action_history,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        logger.info(f"Debug information saved to {path}")
