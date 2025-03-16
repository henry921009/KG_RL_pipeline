"""
System testing and quality assurance script
"""

import os
import sys
import json
import time
import logging
import torch
import numpy as np
from tqdm import tqdm
import config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("testing.log"), logging.StreamHandler()],
)
logger = logging.getLogger("Testing")


def test_kg_loading():
    """Test knowledge graph loading"""
    print("Testing knowledge graph loading...")

    from data_utils import load_knowledge_graph

    kb_path = os.path.join(config.DATA_DIR, "kb.txt")
    start_time = time.time()

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
        elapsed = time.time() - start_time

        print(f"✓ Knowledge graph loaded successfully ({elapsed:.2f} seconds)")
        print(f"  - Number of entities: {len(entities)}")
        print(f"  - Number of relations: {len(relations)}")
        print(f"  - Number of triples: {kg_stats['triple_count']}")

        # Test specific entities
        test_entities = ["Titanic", "The Matrix", "James Cameron", "Tom Hanks"]
        print("\nTesting specific entities:")

        for entity in test_entities:
            if entity in kg_dict:
                edges = kg_dict[entity]
                print(f"  ✓ '{entity}' found in KG with {len(edges)} edges")
                if edges:
                    print(f"    Sample edges: {edges[:3]}")
            else:
                print(f"  ✗ '{entity}' not found in KG")

                # Try to find in entity variants
                if entity_variants:
                    variants = []
                    for variant, original in entity_variants.items():
                        if (
                            original.lower() == entity.lower()
                            or entity.lower() in original.lower()
                        ):
                            variants.append((variant, original))

                    if variants:
                        print(f"    Possible variants: {variants[:5]}")

        return True, {
            "kg_dict": kg_dict,
            "entities": entities,
            "relations": relations,
            "kg_stats": kg_stats,
            "entity_to_idx": entity_to_idx,
            "relation_to_idx": relation_to_idx,
            "entity_variants": entity_variants,
        }

    except Exception as e:
        print(f"✗ Knowledge graph loading failed: {e}")
        return False, str(e)


def test_qa_data_loading():
    """Test QA data loading"""
    print("\nTesting QA data loading...")

    from data_utils import load_qa_data

    test_files = [
        os.path.join(config.DATA_DIR, "1-hop/vanilla/qa_train.txt"),
        os.path.join(config.DATA_DIR, "1-hop/vanilla/qa_dev.txt"),
        os.path.join(config.DATA_DIR, "1-hop/vanilla/qa_test.txt"),
    ]

    results = {}

    for file_path in test_files:
        if not os.path.exists(file_path):
            print(f"✗ File not found: {file_path}")
            continue

        try:
            start_time = time.time()
            (
                questions,
                answers,
                question_entities,
                processed_questions,
                question_types,
                question_focuses,
            ) = load_qa_data(file_path)
            elapsed = time.time() - start_time

            print(f"✓ Loaded successfully: {file_path} ({elapsed:.2f} seconds)")
            print(f"  - Number of questions: {len(questions)}")
            print(f"  - Sample question: {questions[0]}")
            print(f"  - Sample answer: {answers[0]}")
            print(f"  - Sample entity: {question_entities[0]}")

            # Count question types
            type_counts = {}
            for q_type in question_types:
                if q_type not in type_counts:
                    type_counts[q_type] = 0
                type_counts[q_type] += 1

            print(f"  - Question type distribution: {type_counts}")

            results[file_path] = {"count": len(questions), "types": type_counts}

        except Exception as e:
            print(f"✗ Loading failed: {file_path}, error: {e}")

    return len(results) > 0, results


def test_llm_reward_function():
    """Test LLM reward function"""
    print("\nTesting LLM reward function...")

    from llm_reward import LLMRewardFunction

    try:
        # Check API key
        api_key = os.environ.get("OPENAI_API_KEY") or config.OPENAI_API_KEY
        if not api_key:
            print(f"✗ OpenAI API key not set")
            return False, "API key not set"

        # Create reward function
        llm_reward_fn = LLMRewardFunction()
        print(f"✓ LLM reward function created successfully")

        # Test action guidance
        test_state = {
            "question": "Who directed Titanic?",
            "current_entities": ["Titanic"],
            "path_history": [],
        }

        test_actions = [("basic", "directed_by"), ("basic", "starring"), ("stop", None)]

        start_time = time.time()
        try:
            scores = llm_reward_fn.get_action_guidance(test_state, test_actions)
            elapsed = time.time() - start_time

            print(f"✓ Action guidance obtained successfully ({elapsed:.2f} seconds)")
            print(f"  - Actions: {test_actions}")
            print(f"  - Scores: {scores}")

            # Test explanation
            explanation = llm_reward_fn.get_action_explanation(
                test_state, "basic", "directed_by"
            )
            print(f"✓ Action explanation obtained successfully")
            print(f"  - Explanation: {explanation}")

            return True, {"scores": scores, "explanation": explanation}

        except Exception as e:
            print(f"✗ Action guidance failed: {e}")
            return False, str(e)

    except Exception as e:
        print(f"✗ LLM reward function creation failed: {e}")
        return False, str(e)


def test_model_initialization():
    """Test model initialization"""
    print("\nTesting model initialization...")

    from model import KGQAUnifiedAgent
    from data_utils import load_embeddings

    try:
        # Get entities and relations
        kg_success, kg_data = test_kg_loading()
        if not kg_success:
            print(
                "✗ Unable to test model initialization, knowledge graph loading failed"
            )
            return False, "Knowledge graph loading failed"

        entities = kg_data["entities"]
        relations = kg_data["relations"]

        # Load or create embeddings
        start_time = time.time()
        entity_embeddings, relation_embeddings = load_embeddings(entities, relations)
        elapsed = time.time() - start_time

        print(f"✓ Embeddings loaded successfully ({elapsed:.2f} seconds)")
        print(f"  - Entity embeddings shape: {entity_embeddings.shape}")
        print(f"  - Relation embeddings shape: {relation_embeddings.shape}")

        # Create model
        start_time = time.time()
        agent = KGQAUnifiedAgent(entity_embeddings, relation_embeddings)
        elapsed = time.time() - start_time

        print(f"✓ Model created successfully ({elapsed:.2f} seconds)")

        # Test save and load
        model_path = os.path.join(config.MODEL_SAVE_DIR, "test_model.pt")
        agent.save(model_path)
        print(f"✓ Model saved successfully: {model_path}")

        loaded_agent = KGQAUnifiedAgent.load(model_path, len(entities), len(relations))
        print(f"✓ Model loaded successfully")

        return True, {"model_path": model_path}

    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return False, str(e)


def test_environment():
    """Test environment"""
    print("\nTesting environment...")

    from kg_env import KnowledgeGraphEnv

    try:
        # Get knowledge graph data
        kg_success, kg_data = test_kg_loading()
        if not kg_success:
            print("✗ Unable to test environment, knowledge graph loading failed")
            return False, "Knowledge graph loading failed"

        kg_dict = kg_data["kg_dict"]
        entities = kg_data["entities"]
        relations = kg_data["relations"]
        entity_to_idx = kg_data["entity_to_idx"]
        relation_to_idx = kg_data["relation_to_idx"]
        entity_variants = kg_data["entity_variants"]

        # Create environment
        env = KnowledgeGraphEnv(
            kg_dict,
            entities,
            relations,
            entity_to_idx,
            relation_to_idx,
            entity_variants,
        )
        print(f"✓ Environment created successfully")

        # Test reset
        test_question = "Who directed Titanic?"
        test_entity = "Titanic"

        state = env.reset(test_question, test_entity)
        print(f"✓ Environment reset successfully")
        print(f"  - State: {state}")

        # Test valid actions retrieval
        valid_actions = env.get_valid_actions()
        print(
            f"✓ Retrieved valid actions successfully, found {len(valid_actions)} actions"
        )

        if valid_actions:
            # Test executing action
            action_type, action_params = valid_actions[0]
            next_state, reward, done, info = env.step(action_type, action_params)

            print(f"✓ Action executed successfully: {action_type}, {action_params}")
            print(f"  - Reward: {reward}")
            print(f"  - Done: {done}")
            print(f"  - Next state entity count: {len(next_state['current_entities'])}")

        return True, {
            "question": test_question,
            "entity": test_entity,
            "initial_state": state,
            "valid_actions_count": len(valid_actions),
        }

    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False, str(e)


def test_evaluator():
    """Test evaluator"""
    print("\nTesting evaluator...")

    from kg_env import KnowledgeGraphEnv
    from model import KGQAUnifiedAgent
    from llm_reward import LLMRewardFunction
    from evaluator import KGQAEvaluator
    from data_utils import load_embeddings

    try:
        # Get knowledge graph data
        kg_success, kg_data = test_kg_loading()
        if not kg_success:
            print("✗ Unable to test evaluator, knowledge graph loading failed")
            return False, "Knowledge graph loading failed"

        kg_dict = kg_data["kg_dict"]
        entities = kg_data["entities"]
        relations = kg_data["relations"]
        entity_to_idx = kg_data["entity_to_idx"]
        relation_to_idx = kg_data["relation_to_idx"]
        entity_variants = kg_data["entity_variants"]

        # Create environment
        env = KnowledgeGraphEnv(
            kg_dict,
            entities,
            relations,
            entity_to_idx,
            relation_to_idx,
            entity_variants,
        )

        # Create agent
        entity_embeddings, relation_embeddings = load_embeddings(entities, relations)
        agent = KGQAUnifiedAgent(entity_embeddings, relation_embeddings)

        # Create LLM reward function
        llm_reward_fn = LLMRewardFunction()

        # Create evaluator
        evaluator = KGQAEvaluator(
            agent, env, llm_reward_fn, entity_to_idx, relation_to_idx, entity_variants
        )
        print(f"✓ Evaluator created successfully")

        # Test single question evaluation
        test_question = "Who directed Titanic?"
        test_entity = "Titanic"

        try:
            start_time = time.time()
            result = evaluator.evaluate_single_question(
                test_question,
                ["James Cameron"],  # Assumed correct answer
                test_entity,
            )
            elapsed = time.time() - start_time

            print(f"✓ Single question evaluation succeeded ({elapsed:.2f} seconds)")
            print(f"  - Predicted answers: {result['predicted_answers']}")
            print(f"  - F1 score: {result['f1']}")
            print(f"  - Number of steps: {len(result['actions'])}")

            return True, {
                "question": test_question,
                "entity": test_entity,
                "evaluation_result": result,
            }

        except Exception as e:
            print(f"✗ Single question evaluation failed: {e}")
            return False, str(e)

    except Exception as e:
        print(f"✗ Evaluator test failed: {e}")
        return False, str(e)


def run_demo_test():
    """Test demo script"""
    print("\nTesting demo script...")

    import subprocess

    try:
        # Test default demo
        cmd = [
            "python",
            "run_demo.py",
            "--question",
            "Who directed Titanic?",
            "--entity",
            "Titanic",
        ]
        print(f"Executing command: {' '.join(cmd)}")

        start_time = time.time()
        process = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start_time

        if process.returncode == 0:
            print(f"✓ Demo script ran successfully ({elapsed:.2f} seconds)")

            # Check output
            output = process.stdout
            if "推理完成" in output:
                print(f"✓ Completion marker found")
            else:
                print(f"✗ Completion marker not found")

            return True, {"elapsed": elapsed, "output_length": len(output)}
        else:
            print(f"✗ Demo script failed, return code: {process.returncode}")
            print(f"Error output: {process.stderr}")
            return False, process.stderr

    except Exception as e:
        print(f"✗ Demo script test failed: {e}")
        return False, str(e)


def run_all_tests():
    """Run all tests"""
    print("=" * 80)
    print("Starting comprehensive system testing")
    print("=" * 80)

    results = {}

    # Get system information
    import platform
    import torch

    system_info = {
        "os": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device": str(config.DEVICE),
    }

    print(f"System information:")
    print(f"  - Operating system: {system_info['os']}")
    print(f"  - Python version: {system_info['python']}")
    print(f"  - PyTorch version: {system_info['torch']}")
    print(f"  - CUDA available: {system_info['cuda']}")
    print(f"  - Device: {system_info['device']}")

    results["system_info"] = system_info

    # Test knowledge graph loading
    kg_success, kg_result = test_kg_loading()
    results["kg_loading"] = {"success": kg_success, "details": kg_result}

    # Test QA data loading
    qa_success, qa_result = test_qa_data_loading()
    results["qa_loading"] = {"success": qa_success, "details": qa_result}

    # Test LLM reward function
    llm_success, llm_result = test_llm_reward_function()
    results["llm_reward"] = {"success": llm_success, "details": llm_result}

    # Test model initialization
    model_success, model_result = test_model_initialization()
    results["model_init"] = {"success": model_success, "details": model_result}

    # Test environment
    env_success, env_result = test_environment()
    results["environment"] = {"success": env_success, "details": env_result}

    # Test evaluator
    eval_success, eval_result = test_evaluator()
    results["evaluator"] = {"success": eval_success, "details": eval_result}

    # Test demo script
    demo_success, demo_result = run_demo_test()
    results["demo"] = {"success": demo_success, "details": demo_result}

    # Save test results
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_path = os.path.join(config.LOG_DIR, f"system_test_{timestamp}.json")

    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"Testing complete, results saved to: {result_path}")

    # Calculate overall success rate
    success_count = sum(
        1 for k, v in results.items() if k != "system_info" and v.get("success", False)
    )
    total_tests = len(results) - 1  # Exclude system_info

    print(
        f"Test success rate: {success_count}/{total_tests} ({success_count / total_tests * 100:.1f}%)"
    )
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_all_tests()
