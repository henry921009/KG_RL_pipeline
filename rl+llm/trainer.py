"""
Training logic: Implements the model training process
"""

import os
import time
import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import config


class KGQATrainer:
    """
    KGQA Trainer
    """

    def __init__(
        self, agent, env, llm_reward_fn, entity_to_idx, relation_to_idx, optimizer=None
    ):
        """
        Initialize the trainer

        Parameters:
            agent (KGQAUnifiedAgent): RL agent
            env (KnowledgeGraphEnv): Knowledge graph environment
            llm_reward_fn (LLMRewardFunction): LLM reward function
            entity_to_idx (dict): Mapping from entity to index
            relation_to_idx (dict): Mapping from relation to index
            optimizer (torch.optim.Optimizer): Optimizer
        """
        self.agent = agent
        self.env = env
        self.llm_reward_fn = llm_reward_fn
        self.entity_to_idx = entity_to_idx
        self.relation_to_idx = relation_to_idx

        # Create an optimizer if none is provided
        if optimizer is None:
            self.optimizer = optim.Adam(agent.parameters(), lr=config.LEARNING_RATE)
        else:
            self.optimizer = optimizer

        # Training statistics
        self.stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "correct_episodes": [],
            "loss_history": [],
            "epsilon_history": [],
        }

        # Best model performance
        self.best_accuracy = 0.0
        self.best_model_path = None

        # Create model save directory if it does not exist
        if not os.path.exists(config.MODEL_SAVE_DIR):
            os.makedirs(config.MODEL_SAVE_DIR)

    def train_episode(self, question, question_entity, true_answers, epsilon):
        """
        Train one episode

        Parameters:
            question (str): The question
            question_entity (str): The entity mentioned in the question
            true_answers (list): The ground truth answers
            epsilon (float): Exploration rate

        Returns:
            dict: Episode statistics
        """
        # Reset the environment
        state = self.env.reset(question, question_entity)

        # Record trajectory
        log_probs = []
        rewards = []

        done = False
        episode_length = 0

        while not done:
            # Retrieve valid actions
            valid_actions = self.env.get_valid_actions()

            if not valid_actions:
                # No valid actions, end the episode early
                break

            # Get LLM guidance
            action_scores = self.llm_reward_fn.get_action_guidance(state, valid_actions)

            # Choose action
            action_type, action_params, log_prob = self.agent.act(
                state,
                valid_actions,
                self.entity_to_idx,
                self.relation_to_idx,
                epsilon=epsilon,
                llm_guidance=action_scores,
            )

            # Execute action
            next_state, reward, done, info = self.env.step(action_type, action_params)

            # Record log probability and reward
            log_probs.append(log_prob)
            rewards.append(reward)

            # Update state
            state = next_state
            episode_length += 1

            # End episode if maximum steps reached
            if episode_length >= config.MAX_STEPS:
                done = True

        # Compute final reward
        final_reward = self.env.calculate_reward(true_answers)

        # Replace the last reward with final reward if the episode did not end explicitly
        if rewards:
            rewards[-1] = final_reward

        # Compute discounted cumulative rewards
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + config.GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns).to(config.DEVICE)

        # Return early if returns are empty
        if len(returns) == 0:
            return {"reward": 0, "length": 0, "correct": False, "loss": 0}

        # Normalize returns if there is more than one value
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute policy loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)

        policy_loss = torch.stack(policy_loss).sum()

        # Update the model
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # Check if the episode is correct
        correct = set(self.env.current_entities).intersection(set(true_answers)) == set(
            true_answers
        )

        return {
            "reward": final_reward,
            "length": episode_length,
            "correct": correct,
            "loss": policy_loss.item(),
        }

    def train(
        self,
        train_questions,
        train_answers,
        train_entities,
        dev_questions=None,
        dev_answers=None,
        dev_entities=None,
        num_episodes=None,
        batch_size=1,
    ):
        """
        Train the model

        Parameters:
            train_questions (list): Training questions
            train_answers (list): Training answers
            train_entities (list): Training question entities
            dev_questions (list): Development questions
            dev_answers (list): Development answers
            dev_entities (list): Development question entities
            num_episodes (int): Number of training episodes
            batch_size (int): Batch size

        Returns:
            dict: Training statistics
        """
        if num_episodes is None:
            num_episodes = config.MAX_EPISODES

        print(f"Starting training: {num_episodes} episodes, batch size = {batch_size}")

        # Initialize progress bar
        pbar = tqdm(total=num_episodes)

        # Training loop
        episode_count = 0
        epoch = 0
        best_model_path = None

        while episode_count < num_episodes:
            epoch += 1
            epoch_rewards = []
            epoch_lengths = []
            epoch_correct = []
            epoch_losses = []

            # Compute current exploration rate
            epsilon = max(
                config.EPSILON_END,
                config.EPSILON_START - episode_count / config.EPSILON_DECAY,
            )

            # Create batch
            indices = np.random.permutation(len(train_questions))
            for start_idx in range(0, len(indices), batch_size):
                if episode_count >= num_episodes:
                    break

                # Get batch indices
                end_idx = min(start_idx + batch_size, len(indices))
                batch_indices = indices[start_idx:end_idx]

                # Retrieve batch data
                batch_questions = [train_questions[i] for i in batch_indices]
                batch_answers = [train_answers[i] for i in batch_indices]
                batch_entities = [train_entities[i] for i in batch_indices]

                # Train on batch
                batch_stats = []
                for question, answers, entity in zip(
                    batch_questions, batch_answers, batch_entities
                ):
                    # Train one episode
                    episode_stats = self.train_episode(
                        question, entity, answers, epsilon
                    )
                    batch_stats.append(episode_stats)

                    # Update statistics
                    epoch_rewards.append(episode_stats["reward"])
                    epoch_lengths.append(episode_stats["length"])
                    epoch_correct.append(episode_stats["correct"])
                    epoch_losses.append(episode_stats["loss"])

                    # Update progress bar
                    pbar.update(1)
                    episode_count += 1

                # Accumulate statistics
                self.stats["episode_rewards"].extend(epoch_rewards)
                self.stats["episode_lengths"].extend(epoch_lengths)
                self.stats["correct_episodes"].extend(epoch_correct)
                self.stats["loss_history"].extend(epoch_losses)
                self.stats["epsilon_history"].append(epsilon)

                # Update progress bar description
                avg_reward = np.mean(epoch_rewards[-batch_size:])
                avg_length = np.mean(epoch_lengths[-batch_size:])
                avg_correct = np.mean(epoch_correct[-batch_size:])
                avg_loss = np.mean(epoch_losses[-batch_size:])

                pbar.set_description(
                    f"Epoch {epoch} | "
                    f"Reward: {avg_reward:.4f} | "
                    f"Length: {avg_length:.2f} | "
                    f"Correct: {avg_correct:.4f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Epsilon: {epsilon:.4f}"
                )

                # Save the model periodically
                if episode_count % config.SAVE_INTERVAL == 0:
                    model_path = os.path.join(
                        config.MODEL_SAVE_DIR, f"model_episode_{episode_count}.pt"
                    )
                    self.agent.save(model_path)
                    print(f"\nModel saved: {model_path}")

                # Evaluate the model periodically
                if dev_questions and episode_count % config.LOG_INTERVAL == 0:
                    print("\nEvaluating model...")
                    accuracy = self.evaluate(dev_questions, dev_answers, dev_entities)
                    print(f"Validation accuracy: {accuracy:.4f}")

                    # Save the best model
                    if accuracy > self.best_accuracy:
                        self.best_accuracy = accuracy
                        best_model_path = os.path.join(
                            config.MODEL_SAVE_DIR, f"best_model_acc_{accuracy:.4f}.pt"
                        )
                        self.agent.save(best_model_path)
                        print(f"New best model saved: {best_model_path}")
                        self.best_model_path = best_model_path

        pbar.close()

        # Save the final model
        final_model_path = os.path.join(config.MODEL_SAVE_DIR, "final_model.pt")
        self.agent.save(final_model_path)
        print(f"Final model saved: {final_model_path}")

        # If there is a best model, display its information
        if best_model_path:
            print(f"Best model: {best_model_path}, Accuracy: {self.best_accuracy:.4f}")

        return self.stats

    def evaluate(self, questions, answers, question_entities, limit=None):
        """
        Evaluate the model

        Parameters:
            questions (list): List of questions
            answers (list): List of answers
            question_entities (list): List of question entities
            limit (int): Limit number of questions for evaluation

        Returns:
            float: Accuracy
        """
        correct = 0
        total = len(questions)

        # Limit the number of questions for evaluation
        if limit and limit < total:
            indices = np.random.choice(total, limit, replace=False)
            eval_questions = [questions[i] for i in indices]
            eval_answers = [answers[i] for i in indices]
            eval_entities = [question_entities[i] for i in indices]
            total = limit
        else:
            eval_questions = questions
            eval_answers = answers
            eval_entities = question_entities

        # Evaluate each question
        for question, true_answers, entity in tqdm(
            zip(eval_questions, eval_answers, eval_entities), total=total, desc="Evaluation"
        ):
            # Reset the environment
            state = self.env.reset(question, entity)

            done = False
            while not done:
                # Retrieve valid actions
                valid_actions = self.env.get_valid_actions()

                if not valid_actions:
                    break

                # Get LLM guidance
                action_scores = self.llm_reward_fn.get_action_guidance(
                    state, valid_actions
                )

                # Choose action (without exploration)
                action_type, action_params, _ = self.agent.act(
                    state,
                    valid_actions,
                    self.entity_to_idx,
                    self.relation_to_idx,
                    epsilon=0,
                    llm_guidance=action_scores,
                )

                # Execute action
                state, _, done, _ = self.env.step(action_type, action_params)

            # Check if correct (all true answers are found)
            predicted_answers = set(self.env.current_entities)
            true_answers_set = set(true_answers)

            if predicted_answers and true_answers_set.issubset(predicted_answers):
                correct += 1

        # Compute accuracy
        accuracy = correct / total
        return accuracy

    def visualize_training(self, save_path=None):
        """
        Visualize the training process

        Parameters:
            save_path (str): Save path for the image. If None, the image is displayed.
        """
        # Create the figure
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        # Smooth function for curves
        def smooth(y, window=10):
            box = np.ones(window) / window
            y_smooth = np.convolve(y, box, mode="same")
            return y_smooth

        # Plot reward curve
        axs[0, 0].plot(smooth(self.stats["episode_rewards"]))
        axs[0, 0].set_title("Average Reward")
        axs[0, 0].set_xlabel("Episode")
        axs[0, 0].set_ylabel("Reward")

        # Plot episode length curve
        axs[0, 1].plot(smooth(self.stats["episode_lengths"]))
        axs[0, 1].set_title("Average Episode Length")
        axs[0, 1].set_xlabel("Episode")
        axs[0, 1].set_ylabel("Length")

        # Plot correct rate curve
        correct_rate = []
        window_size = 100
        for i in range(0, len(self.stats["correct_episodes"]), window_size):
            window = self.stats["correct_episodes"][i : i + window_size]
            if window:
                correct_rate.append(sum(window) / len(window))

        axs[1, 0].plot(correct_rate)
        axs[1, 0].set_title("Correct Rate (Window Size = 100)")
        axs[1, 0].set_xlabel("Window")
        axs[1, 0].set_ylabel("Correct Rate")

        # Plot loss curve
        axs[1, 1].plot(smooth(self.stats["loss_history"]))
        axs[1, 1].set_title("Loss")
        axs[1, 1].set_xlabel("Episode")
        axs[1, 1].set_ylabel("Loss")

        # Adjust layout
        plt.tight_layout()

        # Save or display the image
        if save_path:
            plt.savefig(save_path)
            print(f"Image saved: {save_path}")
        else:
            plt.show()
