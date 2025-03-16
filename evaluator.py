"""
Evaluation logic, implementing model evaluation and analysis
"""

import json
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import config
import logging
import os
from difflib import SequenceMatcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evaluator.log"), logging.StreamHandler()],
)
logger = logging.getLogger("Evaluator")


class KGQAEvaluator:
    """
    KGQA Evaluator
    """

    def __init__(
        self,
        agent,
        env,
        llm_reward_fn,
        entity_to_idx,
        relation_to_idx,
        entity_variants=None,
    ):
        """
        Initialize evaluator

        Parameters:
            agent (KGQAUnifiedAgent): RL agent
            env (KnowledgeGraphEnv): Knowledge graph environment
            llm_reward_fn (LLMRewardFunction): LLM reward function
            entity_to_idx (dict): Mapping from entity to index
            relation_to_idx (dict): Mapping from relation to index
            entity_variants (dict): Entity variant dictionary
        """
        self.agent = agent
        self.env = env
        self.llm_reward_fn = llm_reward_fn
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx
        self.entity_variants = entity_variants or {}

        # Evaluation results
        self.results = None

        logger.info("KGQA Evaluator initialization complete")

    def evaluate(
        self,
        questions,
        answers,
        question_entities,
        question_types=None,
        question_focuses=None,
        limit=None,
    ):
        """
        Evaluate the model

        Parameters:
            questions (list): List of questions
            answers (list): List of answers
            question_entities (list): List of question entities
            question_types (list): List of question types
            question_focuses (list): List of question focuses
            limit (int): Limit on the number of questions to evaluate

        Returns:
            dict: Evaluation results
        """
        logger.info(f"Starting evaluation of {len(questions)} questions...")

        results = {
            "correct": 0,
            "total": len(questions),
            "precisions": [],
            "recalls": [],
            "f1s": [],
            "path_lengths": [],
            "action_types": defaultdict(int),
            "detailed_results": [],
        }

        # Limit the number of questions for evaluation
        if limit and limit < results["total"]:
            indices = np.random.choice(results["total"], limit, replace=False)
            eval_questions = [questions[i] for i in indices]
            eval_answers = [answers[i] for i in indices]
            eval_entities = [question_entities[i] for i in indices]
            if question_types:
                eval_types = [question_types[i] for i in indices]
            else:
                eval_types = [None] * len(eval_questions)
            if question_focuses:
                eval_focuses = [question_focuses[i] for i in indices]
            else:
                eval_focuses = [None] * len(eval_questions)
            results["total"] = limit
        else:
            eval_questions = questions
            eval_answers = answers
            eval_entities = question_entities
            if question_types:
                eval_types = question_types
            else:
                eval_types = [None] * len(eval_questions)
            if question_focuses:
                eval_focuses = question_focuses
            else:
                eval_focuses = [None] * len(eval_questions)

        # Evaluate each question
        for i, (question, true_answers, entity, q_type, q_focus) in enumerate(
            tqdm(
                zip(
                    eval_questions,
                    eval_answers,
                    eval_entities,
                    eval_types,
                    eval_focuses,
                ),
                total=results["total"],
                desc="Evaluating",
            )
        ):
            # Execute evaluation
            eval_result = self.evaluate_single_question(
                question, true_answers, entity, q_type, q_focus
            )

            # Statistics
            if eval_result["correct"]:
                results["correct"] += 1

            results["precisions"].append(eval_result["precision"])
            results["recalls"].append(eval_result["recall"])
            results["f1s"].append(eval_result["f1"])
            results["path_lengths"].append(len(eval_result["actions"]))

            # Count action types
            for action_type, _ in eval_result["actions"]:
                results["action_types"][action_type] += 1

            # Record detailed results
            results["detailed_results"].append(eval_result)

        # Calculate overall metrics
        results["accuracy"] = results["correct"] / results["total"]
        results["avg_precision"] = np.mean(results["precisions"])
        results["avg_recall"] = np.mean(results["recalls"])
        results["avg_f1"] = np.mean(results["f1s"])
        results["avg_path_length"] = np.mean(results["path_lengths"])

        logger.info(
            f"Evaluation complete: Accuracy = {results['accuracy']:.4f}, F1 Score = {results['avg_f1']:.4f}"
        )

        # Save results
        self.results = results

        return results

    def evaluate_single_question(
        self, question, true_answers, entity, question_type=None, question_focus=None
    ):
        """
        Evaluate a single question

        Parameters:
            question (str): The question
            true_answers (list): True answers
            entity (str): Question entity
            question_type (str): Question type
            question_focus (str): Question focus

        Returns:
            dict: Evaluation result
        """
        # Reset the environment
        state = self.env.reset(question, entity, question_type, question_focus)

        # Record action trajectory
        actions_taken = []
        explanations = []

        done = False
        step = 0

        while not done and step < config.MAX_STEPS:
            step += 1

            # Get valid actions
            valid_actions = self.env.get_valid_actions()

            if not valid_actions:
                break

            # Get LLM guidance
            try:
                action_scores = self.llm_reward_fn.get_action_guidance(
                    state, valid_actions
                )
            except Exception as e:
                logger.error(f"Error getting LLM guidance: {e}")
                action_scores = None

            # Choose action (without exploration)
            action_type, action_params, _ = self.agent.act(
                state,
                valid_actions,
                self.entity_to_idx,
                self.relation_to_idx,
                epsilon=0,
                llm_guidance=action_scores,
            )

            # Get action explanation
            try:
                explanation = self.llm_reward_fn.get_action_explanation(
                    state, action_type, action_params
                )
            except Exception as e:
                logger.error(f"Error getting action explanation: {e}")
                explanation = "Unable to retrieve explanation."

            # Record action and explanation
            actions_taken.append((action_type, action_params))
            explanations.append(explanation)

            # Execute action
            next_state, _, done, _ = self.env.step(action_type, action_params)

            # Update state
            state = next_state

        # Predicted result
        predicted_answers = self.env.current_entities

        # Calculate precision, recall, and F1 score
        true_answers_set = set(true_answers)
        predicted_answers_set = set(predicted_answers)

        # Attempt matching using entity variants
        if self.entity_variants:
            true_variants = set()
            for ans in true_answers:
                # Add original answer
                true_variants.add(ans)
                # Add variants
                for variant, original in self.entity_variants.items():
                    if original == ans:
                        true_variants.add(variant)
                    # Use similarity matching
                    elif (
                        SequenceMatcher(None, variant.lower(), ans.lower()).ratio()
                        > 0.9
                    ):
                        true_variants.add(variant)

            # Extended true answer set
            expanded_true_set = true_answers_set.union(true_variants)

            # If the expanded set improves matching, use it
            if len(predicted_answers_set.intersection(expanded_true_set)) > len(
                predicted_answers_set.intersection(true_answers_set)
            ):
                true_answers_set = expanded_true_set
                logger.info(
                    f"Using entity variants to expand true answers set: {true_answers} -> {true_answers_set}"
                )

        if predicted_answers_set:
            precision = len(predicted_answers_set.intersection(true_answers_set)) / len(
                predicted_answers_set
            )
        else:
            precision = 0

        if true_answers_set:
            recall = len(predicted_answers_set.intersection(true_answers_set)) / len(
                true_answers_set
            )
        else:
            recall = 0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0

        # Determine if correct
        correct = (
            len(predicted_answers_set.intersection(true_answers_set)) > 0
            and len(predicted_answers_set) > 0
        )

        # Process action text
        action_texts = []
        for action_type, action_params in actions_taken:
            if action_type == "basic":
                action_texts.append(f"Proceed along relation '{action_params}'")
            elif action_type == "filter":
                rel, val, op = action_params
                action_texts.append(f"Filter condition '{rel} {op} {val}'")
            elif action_type == "union":
                rel1, rel2 = action_params
                action_texts.append(f"Union results of relations '{rel1}' and '{rel2}'")
            elif action_type == "aggregation":
                action_texts.append(f"Aggregation operation '{action_params}'")
            elif action_type == "ordinal":
                sort_rel, order, pos = action_params
                action_texts.append(
                    f"Sort by '{sort_rel}' ({order}) and select position {pos}"
                )
            elif action_type == "stop":
                action_texts.append(f"Stop and return current entities as answer")

        # Return detailed result
        return {
            "question": question,
            "question_type": question_type,
            "question_focus": question_focus,
            "entity": entity,
            "true_answers": true_answers,
            "predicted_answers": predicted_answers,
            "actions": actions_taken,
            "action_texts": action_texts,
            "explanations": explanations,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "correct": correct,
        }

    def analyze_results(self, results=None, save_path=None):
        """
        Analyze evaluation results

        Parameters:
            results (dict): Evaluation results; if None, use last evaluation results
            save_path (str): Path to save the analysis results

        Returns:
            dict: Analysis results
        """
        if results is None:
            if self.results is None:
                raise ValueError(
                    "No evaluation results available. Please call evaluate() first."
                )
            results = self.results

        analysis = {
            "overall_metrics": {
                "accuracy": results["accuracy"],
                "precision": results["avg_precision"],
                "recall": results["avg_recall"],
                "f1": results["avg_f1"],
                "avg_path_length": results["avg_path_length"],
            },
            "path_length_analysis": {},
            "action_type_analysis": dict(results["action_types"]),
            "question_type_analysis": {},
            "error_analysis": {},
        }

        # Analyze performance by path length
        path_lengths = defaultdict(list)
        for i, length in enumerate(results["path_lengths"]):
            path_lengths[length].append(results["f1s"][i])

        for length, f1s in path_lengths.items():
            analysis["path_length_analysis"][length] = {
                "count": len(f1s),
                "avg_f1": np.mean(f1s),
            }

        # Analyze performance by question type
        question_types = defaultdict(list)
        for result in results["detailed_results"]:
            q_type = result.get("question_type", "unknown")
            question_types[q_type].append(result["f1"])

        for q_type, f1s in question_types.items():
            analysis["question_type_analysis"][q_type] = {
                "count": len(f1s),
                "avg_f1": np.mean(f1s),
            }

        # Error analysis
        error_cases = []
        for result in results["detailed_results"]:
            if not result["correct"]:
                error_cases.append(
                    {
                        "question": result["question"],
                        "true_answers": result["true_answers"],
                        "predicted_answers": result["predicted_answers"],
                        "actions": result["action_texts"],
                        "f1": result["f1"],
                    }
                )

        analysis["error_analysis"]["count"] = len(error_cases)

        if error_cases:
            # Sort by F1 score, select the worst 10 cases
            worst_cases = sorted(error_cases, key=lambda x: x["f1"])[:10]
            analysis["error_analysis"]["worst_cases"] = worst_cases

        # Visualization
        self.visualize_results(results, save_path)

        # Save analysis results
        if save_path:
            analysis_path = save_path.replace(".png", "_analysis.json")
            with open(analysis_path, "w", encoding="utf-8") as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            logger.info(f"Analysis results saved: {analysis_path}")

        return analysis

    def visualize_results(self, results, save_path=None):
        """
        Visualize evaluation results

        Parameters:
            results (dict): Evaluation results
            save_path (str): Path to save the image
        """
        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Plot precision, recall, and F1 distributions
        axs[0, 0].hist(results["precisions"], bins=10, alpha=0.5, label="Precision")
        axs[0, 0].hist(results["recalls"], bins=10, alpha=0.5, label="Recall")
        axs[0, 0].hist(results["f1s"], bins=10, alpha=0.5, label="F1")
        axs[0, 0].legend()
        axs[0, 0].set_title("Precision, Recall, and F1 Distribution")
        axs[0, 0].set_xlabel("Score")
        axs[0, 0].set_ylabel("Count")

        # Plot path length distribution
        axs[0, 1].hist(
            results["path_lengths"],
            bins=range(max(results["path_lengths"]) + 2),
            alpha=0.7,
        )
        axs[0, 1].set_title("Path Length Distribution")
        axs[0, 1].set_xlabel("Path Length")
        axs[0, 1].set_ylabel("Count")

        # Plot F1 score by path length
        path_lengths = defaultdict(list)
        for i, length in enumerate(results["path_lengths"]):
            path_lengths[length].append(results["f1s"][i])

        lengths = sorted(path_lengths.keys())
        avg_f1s = [np.mean(path_lengths[l]) for l in lengths]

        axs[1, 0].bar(lengths, avg_f1s)
        axs[1, 0].set_title("Average F1 by Path Length")
        axs[1, 0].set_xlabel("Path Length")
        axs[1, 0].set_ylabel("Average F1")

        # Plot action type distribution
        action_types = results["action_types"]
        types = list(action_types.keys())
        counts = list(action_types.values())

        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        sorted_types = [types[i] for i in sorted_indices]
        sorted_counts = [counts[i] for i in sorted_indices]

        axs[1, 1].bar(sorted_types, sorted_counts)
        axs[1, 1].set_title("Action Type Distribution")
        axs[1, 1].set_xlabel("Action Type")
        axs[1, 1].set_ylabel("Count")
        plt.setp(axs[1, 1].xaxis.get_majorticklabels(), rotation=45)

        # Adjust layout
        plt.tight_layout()

        # Save or display image
        if save_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Image saved: {save_path}")
        else:
            plt.show()

    def explain_reasoning(
        self,
        question,
        question_entity,
        question_type=None,
        question_focus=None,
        verbose=True,
    ):
        """
        Explain the model's reasoning process

        Parameters:
            question (str): The question
            question_entity (str): The topic entity of the question
            question_type (str): The question type
            question_focus (str): The question focus
            verbose (bool): Whether to output detailed information

        Returns:
            dict: Reasoning explanation
        """
        if verbose:
            print(f"Question: {question}")
            print(f"Topic entity: {question_entity}")

        # Reset the environment
        state = self.env.reset(question, question_entity, question_type, question_focus)

        reasoning_steps = []

        done = False
        step = 1

        while not done and step <= config.MAX_STEPS:
            if verbose:
                print(f"\nStep {step}:")
                print(f"Current entities: {', '.join(state['current_entities'][:5])}")

            # Get valid actions
            valid_actions = self.env.get_valid_actions()

            # In the first step, filter out the "stop" action to ensure at least one meaningful action is executed
            if step == 1 and len(self.env.path_history) == 0:
                original_action_count = len(valid_actions)
                valid_actions = [(at, ap) for at, ap in valid_actions if at != "stop"]
                filtered_action_count = len(valid_actions)

                if verbose and original_action_count != filtered_action_count:
                    print(
                        f"Step 1: Filtered out stop action, {original_action_count} -> {filtered_action_count} valid actions"
                    )

            if not valid_actions:
                if verbose:
                    print("No valid actions, ending reasoning")
                break

            if verbose:
                print(f"Available actions: {len(valid_actions)}")

            # Get LLM guidance
            try:
                action_scores = self.llm_reward_fn.get_action_guidance(
                    state, valid_actions
                )
            except Exception as e:
                logger.error(f"Error getting LLM guidance: {e}")
                action_scores = None
                if verbose:
                    print(f"Error getting LLM guidance: {e}")

            # Choose action (without exploration)
            action_type, action_params, _ = self.agent.act(
                state,
                valid_actions,
                self.entity_to_idx,
                self.relation_to_idx,
                epsilon=0,
                llm_guidance=action_scores,
            )

            # Format action text
            if action_type == "basic":
                action_text = f"Proceed along relation '{action_params}'"
            elif action_type == "filter":
                rel, val, op = action_params
                action_text = f"Filter condition '{rel} {op} {val}'"
            elif action_type == "union":
                rel1, rel2 = action_params
                action_text = f"Union results of relations '{rel1}' and '{rel2}'"
            elif action_type == "aggregation":
                action_text = f"Aggregation operation '{action_params}'"
            elif action_type == "ordinal":
                sort_rel, order, pos = action_params
                action_text = (
                    f"Sort by '{sort_rel}' ({order}) and select position {pos}"
                )
            elif action_type == "stop":
                action_text = "Stop and return current entities as answer"
            else:
                action_text = f"Unknown action '{action_type}'"

            if verbose:
                print(f"Chosen action: {action_text}")

            # Get action explanation
            try:
                explanation = self.llm_reward_fn.get_action_explanation(
                    state, action_type, action_params
                )
                if verbose:
                    print(f"Explanation: {explanation}")
            except Exception as e:
                logger.error(f"Error getting action explanation: {e}")
                explanation = "Unable to retrieve explanation."
                if verbose:
                    print(f"Error getting action explanation: {e}")

            # Execute action
            next_state, reward, done, info = self.env.step(action_type, action_params)

            # Record step
            reasoning_steps.append(
                {
                    "step": step,
                    "current_entities": state["current_entities"][:5],
                    "action_type": action_type,
                    "action_params": action_params,
                    "action_text": action_text,
                    "explanation": explanation,
                    "next_entities": next_state["current_entities"][:5],
                    "reward": reward,
                    "info": info,
                }
            )

            # Update state
            state = next_state
            step += 1

        # Get environment debug info
        debug_info = self.env.get_debug_info()

        # Final answer
        if verbose:
            print("\nReasoning complete!")
            print(f"Final answer: {', '.join(state['current_entities'])}")

        return {
            "question": question,
            "topic_entity": question_entity,
            "question_type": question_type,
            "question_focus": question_focus,
            "steps": reasoning_steps,
            "final_answer": state["current_entities"],
            "debug_info": debug_info,
        }

    def save_explanation(self, explanation, path=None):
        """
        Save the reasoning explanation to a file

        Parameters:
            explanation (dict): Reasoning explanation
            path (str): Save path; if None, a filename will be generated automatically
        """
        if path is None:
            # Generate filename
            question_snippet = (
                explanation["question"][:30]
                .replace(" ", "_")
                .replace("?", "")
                .replace("/", "_")
            )
            path = os.path.join(config.LOG_DIR, f"explanation_{question_snippet}.json")

        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save to file
        with open(path, "w", encoding="utf-8") as f:
            json.dump(explanation, f, ensure_ascii=False, indent=2)

        logger.info(f"Reasoning explanation saved to: {path}")
        return path
