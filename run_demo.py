"""
Demo script for the knowledge graph question answering system driven by reinforcement learning and LLM
"""

import os
import argparse
import torch
import logging
import json
from datetime import datetime

import config
from data_utils import (
    load_knowledge_graph,
    load_qa_data,
    load_embeddings,
    preprocess_question,
)
from kg_env import KnowledgeGraphEnv
from model import KGQAUnifiedAgent
from llm_reward import LLMRewardFunction
from evaluator import KGQAEvaluator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("demo.log"), logging.StreamHandler()],
)
logger = logging.getLogger("Demo")


def setup_environment():
    """Set up the environment and load necessary components"""
    print("Setting up environment...")

    # Load knowledge graph
    kb_path = os.path.join(config.DATA_DIR, "kb.txt")
    try:
        (
            kg_dict,
            entities,
            relations,
            kg_stats,
            entity_to_idx,
            relation_to_idx,
            entity_variants,
        ) = load_knowledge_graph(kb_path)
        print(
            f"v Knowledge graph loaded successfully: {len(entities)} entities, {len(relations)} relations"
        )
    except Exception as e:
        logger.error(f"Knowledge graph loading failed: {e}")
        print(f"x Knowledge graph loading failed: {e}")
        exit(1)

    # Create environment
    env = KnowledgeGraphEnv(
        kg_dict, entities, relations, entity_to_idx, relation_to_idx, entity_variants
    )
    print(f"v Knowledge graph environment created successfully")

    # Create LLM reward function
    try:
        llm_reward_fn = LLMRewardFunction()
        print(
            f"v LLM reward function created successfully, using model: {config.LLM_MODEL}"
        )
    except Exception as e:
        logger.error(f"LLM reward function creation failed: {e}")
        print(f"x LLM reward function creation failed: {e}")
        exit(1)

    # Load or create model
    model_path = os.path.join(config.MODEL_SAVE_DIR, "initial_model.pt")
    if os.path.exists(model_path):
        try:
            agent = KGQAUnifiedAgent.load(model_path, len(entities), len(relations))
            print(f"v Model loaded successfully: {model_path}")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            print(f"x Model loading failed: {e}")

            # Create a new model
            entity_embeddings, relation_embeddings = load_embeddings(
                entities, relations
            )
            agent = KGQAUnifiedAgent(entity_embeddings, relation_embeddings)
            print(f"v New model created successfully")
    else:
        # Create a new model
        entity_embeddings, relation_embeddings = load_embeddings(entities, relations)
        agent = KGQAUnifiedAgent(entity_embeddings, relation_embeddings)
        print(f"v New model created successfully")

    # Create evaluator
    evaluator = KGQAEvaluator(
        agent, env, llm_reward_fn, entity_to_idx, relation_to_idx, entity_variants
    )
    print(f"v Evaluator created successfully")

    return (
        kg_dict,
        entities,
        relations,
        env,
        llm_reward_fn,
        agent,
        evaluator,
        entity_to_idx,
        relation_to_idx,
    )


def run_single_question(evaluator, question, entity=None):
    """Run a demo for a single question"""
    print("\n" + "=" * 80)
    print(f"Question: {question}")

    if entity is None:
        # Try to extract entity from the question
        entity_match = None
        for char in ["[", "]", '"', "'"]:
            if char in question:
                # May contain entity markers
                import re

                match = re.search(r"[\[\"\'](.+?)[\]\"\']", question)
                if match:
                    entity_match = match.group(1)
                    break

        if entity_match:
            entity = entity_match
            print(f"Extracted entity from the question: {entity}")
        else:
            print(
                "Unable to extract entity from the question, please specify one manually:"
            )
            entity = input("> ")

    print(f"Entity: {entity}")

    # Process question type and focus
    processed, question_type, question_focus = preprocess_question(question)
    print(f"Question Type: {question_type}")
    print(f"Question Focus: {question_focus}")

    # Execute reasoning process
    print("\nStarting reasoning process...")
    explanation = evaluator.explain_reasoning(
        question, entity, question_type, question_focus
    )

    # Save the result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(config.LOG_DIR, f"explanation_{timestamp}.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(explanation, f, ensure_ascii=False, indent=2)

    print(f"\nReasoning process saved to: {result_path}")
    print("=" * 80)

    return explanation


def run_demo_questions(evaluator):
    """Run a series of demo questions"""
    demo_questions = [
        # Director-related questions
        {"question": "Who directed Titanic?", "entity": "Titanic"},
        {"question": "Who directed The Matrix?", "entity": "The Matrix"},
        # Actor-related questions
        {"question": "Who starred in Inception?", "entity": "Inception"},
        {
            "question": "Who acted in The Lord of the Rings?",
            "entity": "The Lord of the Rings",
        },
        # Quantity-related questions
        {
            "question": "How many movies did James Cameron direct?",
            "entity": "James Cameron",
        },
        {"question": "How many films did Tom Hanks star in?", "entity": "Tom Hanks"},
    ]

    results = []
    for q_info in demo_questions:
        result = run_single_question(evaluator, q_info["question"], q_info["entity"])
        results.append(result)

    # Save all results
    all_results_path = os.path.join(
        config.LOG_DIR, f"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(all_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nAll demo results saved to: {all_results_path}")

    return results


def interactive_demo(evaluator):
    """Interactive demo"""
    print("\n" + "=" * 80)
    print("Welcome to the interactive demo of the knowledge graph QA system")
    print("Enter a question and press enter. Type 'exit' or 'quit' to exit")
    print("=" * 80)

    while True:
        print("\nPlease enter a question:")
        question = input("> ")

        if question.lower() in ["exit", "quit", "q"]:
            break

        print(
            "Please enter an entity (leave blank to attempt extraction from the question):"
        )
        entity = input("> ")

        if not entity:
            entity = None

        try:
            run_single_question(evaluator, question, entity)
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            print(f"Error processing question: {e}")

    print("\nThank you for using the system! Goodbye!")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Knowledge Graph QA System Demo")
    parser.add_argument("--question", type=str, default=None, help="Question to answer")
    parser.add_argument(
        "--entity", type=str, default=None, help="Entity mentioned in the question"
    )
    parser.add_argument(
        "--run_examples", action="store_true", help="Run example questions"
    )
    parser.add_argument("--interactive", action="store_true", help="Interactive demo")

    args = parser.parse_args()

    # Set up environment
    _, _, _, env, llm_reward_fn, agent, evaluator, _, _ = setup_environment()

    # Run demo based on mode
    if args.question:
        run_single_question(evaluator, args.question, args.entity)
    elif args.run_examples:
        run_demo_questions(evaluator)
    elif args.interactive:
        interactive_demo(evaluator)
    else:
        # By default, run one sample question
        run_single_question(evaluator, "Who directed Titanic?", "Titanic")


if __name__ == "__main__":
    main()
