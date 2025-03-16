"""
RL agent model, implements a unified RL strategy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel, BertTokenizer
import config
import logging
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("model.log"), logging.StreamHandler()],
)
logger = logging.getLogger("Model")


class QuestionEncoder(nn.Module):
    """
    Question encoder, uses BERT model to encode questions
    """

    def __init__(
        self, pretrained_model="bert-base-uncased", output_dim=config.HIDDEN_DIM
    ):
        """
        Initialize the question encoder

        Parameters:
            pretrained_model (str): Pretrained BERT model name
            output_dim (int): Output dimension
        """
        super(QuestionEncoder, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
        self.bert = BertModel.from_pretrained(pretrained_model)

        # Freeze BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Projection layer
        self.projection = nn.Linear(self.bert.config.hidden_size, output_dim)

    def forward(self, question):
        """
        Encode the question

        Parameters:
            question (str): Question text

        Returns:
            tensor: Vector representation of the question
        """
        # Tokenize the question
        inputs = self.tokenizer(
            question, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(next(self.bert.parameters()).device)

        # Get BERT output
        with torch.no_grad():
            outputs = self.bert(**inputs)

        # Use the representation of the [CLS] token
        cls_output = outputs.last_hidden_state[:, 0, :]

        # Project to the specified dimension
        return self.projection(cls_output)


class PathEncoder(nn.Module):
    """
    Path encoder, encodes historical paths
    """

    def __init__(self, relation_dim, hidden_dim):
        """
        Initialize the path encoder

        Parameters:
            relation_dim (int): Relation embedding dimension
            hidden_dim (int): Hidden dimension
        """
        super(PathEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=relation_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, path_relations):
        """
        Encode the path

        Parameters:
            path_relations (tensor): Path relation embeddings [batch_size, seq_len, relation_dim]

        Returns:
            tensor: Vector representation of the path [batch_size, hidden_dim]
        """
        # Path may be empty
        if path_relations.size(1) == 0:
            batch_size = path_relations.size(0)
            device = path_relations.device
            return torch.zeros(batch_size, self.gru.hidden_size).to(device)

        # Encode the path using GRU
        _, hidden = self.gru(path_relations)

        # Return the hidden state from the last layer
        return hidden.squeeze(0)


class KGQAUnifiedAgent(nn.Module):
    """
    Unified KGQA agent, integrates high-level and low-level decision making
    """

    def __init__(
        self, entity_embeddings, relation_embeddings, hidden_dim=config.HIDDEN_DIM
    ):
        """
        Initialize the KGQA agent

        Parameters:
            entity_embeddings (tensor): Entity embeddings matrix
            relation_embeddings (tensor): Relation embeddings matrix
            hidden_dim (int): Hidden dimension
        """
        super(KGQAUnifiedAgent, self).__init__()

        self.entity_dim = entity_embeddings.size(1)
        self.relation_dim = relation_embeddings.size(1)
        self.hidden_dim = hidden_dim

        # Embedding layers
        self.entity_embedding = nn.Embedding.from_pretrained(
            entity_embeddings, freeze=False
        )
        self.relation_embedding = nn.Embedding.from_pretrained(
            relation_embeddings, freeze=False
        )

        # Question encoder
        self.question_encoder = QuestionEncoder(output_dim=hidden_dim)

        # Path encoder
        self.path_encoder = PathEncoder(self.relation_dim, hidden_dim)

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + self.entity_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
        )

        # Action type prediction head
        self.action_type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim, len(config.ACTION_TYPES)),
        )

        # Relation prediction head (for basic action)
        self.relation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim, self.relation_dim),
        )

        # Parameters prediction head (for other actions)
        self.params_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Move model to the configured device
        self.to(config.DEVICE)

        logger.info(
            f"KGQA agent initialized, entity embedding dimension: {self.entity_dim}, relation embedding dimension: {self.relation_dim}"
        )

    def encode_state(self, state, entity_to_idx, relation_to_idx):
        """
        Encode the state

        Parameters:
            state (dict): State
            entity_to_idx (dict): Mapping from entities to indices
            relation_to_idx (dict): Mapping from relations to indices

        Returns:
            tensor: Vector representation of the state
        """
        # Encode question
        question_vec = self.question_encoder(state["question"]).squeeze(0)

        # Encode current entities
        entity_idxs = []
        for e in state["current_entities"][:5]:  # Limit the number of entities
            if e in entity_to_idx:
                entity_idxs.append(entity_to_idx[e])
            else:
                # If entity is not in the dictionary, use a special index
                entity_idxs.append(0)

        if entity_idxs:
            entity_vecs = self.entity_embedding(
                torch.tensor(entity_idxs).to(config.DEVICE)
            )
            entity_vec = torch.mean(entity_vecs, dim=0)
        else:
            entity_vec = torch.zeros(self.entity_dim).to(config.DEVICE)

        # Encode path history
        path_relations = []
        for action_type, params in state["path_history"]:
            if action_type == "basic":
                rel = params
                if rel in relation_to_idx:
                    path_relations.append(relation_to_idx[rel])
            # Other action types... only encode relations for basic paths

        if path_relations:
            path_relation_idxs = torch.tensor(path_relations).to(config.DEVICE)
            path_relation_vecs = self.relation_embedding(path_relation_idxs).unsqueeze(
                0
            )
            path_vec = self.path_encoder(path_relation_vecs).squeeze(0)
        else:
            path_vec = torch.zeros(self.hidden_dim).to(config.DEVICE)

        # Combine all features
        combined_vec = torch.cat([question_vec, entity_vec, path_vec])
        state_vec = self.state_encoder(combined_vec)

        return state_vec

    def forward(self, state, entity_to_idx, relation_to_idx):
        """
        Forward propagation

        Parameters:
            state (dict): State
            entity_to_idx (dict): Mapping from entities to indices
            relation_to_idx (dict): Mapping from relations to indices

        Returns:
            tuple: (action_type_logits, relation_logits, params_vec)
        """
        state_vec = self.encode_state(state, entity_to_idx, relation_to_idx)

        # Predict action type
        action_type_logits = self.action_type_head(state_vec)

        # Predict relation (for basic action)
        relation_vec = self.relation_head(state_vec)
        relation_logits = torch.matmul(relation_vec, self.relation_embedding.weight.t())

        # Predict parameters (for other actions)
        params_vec = self.params_head(state_vec)

        return action_type_logits, relation_logits, params_vec

    def act(
        self,
        state,
        valid_actions,
        entity_to_idx,
        relation_to_idx,
        epsilon=0.1,
        llm_guidance=None,
    ):
        """
        Select action

        Parameters:
            state (dict): State
            valid_actions (list): List of valid actions [(action_type, action_params), ...]
            entity_to_idx (dict): Mapping from entities to indices
            relation_to_idx (dict): Mapping from relations to indices
            epsilon (float): Exploration probability
            llm_guidance (list or None): Action guidance scores provided by LLM

        Returns:
            tuple: (action_type, action_params, log_prob)
        """
        # If no valid actions, return 'stop'
        if not valid_actions:
            return "stop", None, torch.tensor(0.0).to(config.DEVICE)

        # Random exploration
        if np.random.random() < epsilon:
            action_idx = np.random.choice(len(valid_actions))
            action_type, action_params = valid_actions[action_idx]
            return action_type, action_params, torch.tensor(0.0).to(config.DEVICE)

        # Forward propagation
        action_type_logits, relation_logits, _ = self(
            state, entity_to_idx, relation_to_idx
        )

        # Convert valid actions to mask
        action_type_mask = torch.zeros(len(config.ACTION_TYPES)).to(config.DEVICE)
        relation_mask = torch.zeros(self.relation_embedding.num_embeddings).to(
            config.DEVICE
        )

        # Fill in the mask
        valid_action_types = set()
        valid_relations = set()

        for action_type, action_params in valid_actions:
            if action_type in config.ACTION_TYPES:
                idx = config.ACTION_TYPES.index(action_type)
                action_type_mask[idx] = 1.0
                valid_action_types.add(action_type)

            if action_type == "basic" and action_params in relation_to_idx:
                idx = relation_to_idx[action_params]
                relation_mask[idx] = 1.0
                valid_relations.add(action_params)

        # Apply mask
        masked_action_type_logits = action_type_logits.clone()
        masked_action_type_logits[action_type_mask == 0] = float("-inf")

        masked_relation_logits = relation_logits.clone()
        masked_relation_logits[relation_mask == 0] = float("-inf")

        # Compute probabilities
        action_type_probs = F.softmax(masked_action_type_logits, dim=0)
        relation_probs = F.softmax(masked_relation_logits, dim=0)

        # Identify question type and focus
        question_type = state.get("question_type", "general")
        question_focus = state.get("question_focus", "entity")

        # If LLM guidance is available, combine with it
        if llm_guidance is not None and len(llm_guidance) == len(valid_actions):
            action_scores = []

            for i, (action_type, action_params) in enumerate(valid_actions):
                if action_type in config.ACTION_TYPES:
                    type_idx = config.ACTION_TYPES.index(action_type)
                    type_prob = action_type_probs[type_idx].item()

                    # Compute relation probability (if applicable)
                    if action_type == "basic" and action_params in relation_to_idx:
                        rel_idx = relation_to_idx[action_params]
                        rel_prob = relation_probs[rel_idx].item()
                        prob = type_prob * rel_prob
                    else:
                        prob = type_prob

                    # Combine with LLM guidance
                    llm_weight = config.LLM_REWARD_WEIGHT

                    # Adjust LLM weight based on question type
                    if question_type == "who" and question_focus in [
                        "director",
                        "actor",
                        "person",
                    ]:
                        # For 'who' questions, increase LLM guidance weight
                        llm_weight = min(0.8, llm_weight * 1.2)
                    elif question_type == "howmany" and action_type == "aggregation":
                        # For 'howmany' questions with aggregation, increase LLM guidance weight
                        llm_weight = min(0.8, llm_weight * 1.2)

                    combined_score = (
                        1 - llm_weight
                    ) * prob + llm_weight * llm_guidance[i]
                    action_scores.append((i, combined_score))
                else:
                    action_scores.append((i, 0.0))

            # Select the action with the highest score
            best_action_idx = max(action_scores, key=lambda x: x[1])[0]
            action_type, action_params = valid_actions[best_action_idx]

            # Compute logarithmic probability
            if action_type in config.ACTION_TYPES:
                type_idx = config.ACTION_TYPES.index(action_type)
                log_prob = torch.log(action_type_probs[type_idx] + 1e-10)

                if action_type == "basic" and action_params in relation_to_idx:
                    rel_idx = relation_to_idx[action_params]
                    log_prob += torch.log(relation_probs[rel_idx] + 1e-10)
            else:
                log_prob = torch.tensor(0.0).to(config.DEVICE)
        else:
            # Without LLM guidance, use the policy network to select the action

            # First select the action type
            action_type_idx = torch.argmax(action_type_probs).item()
            action_type = config.ACTION_TYPES[action_type_idx]

            # Then select the parameter
            if action_type == "basic":
                relation_idx = torch.argmax(relation_probs).item()
                # Find the matching relation from valid actions
                for at, ap in valid_actions:
                    if (
                        at == "basic"
                        and ap in relation_to_idx
                        and relation_to_idx[ap] == relation_idx
                    ):
                        action_params = ap
                        break
                else:
                    # If no matching relation is found, choose the first valid basic action
                    for at, ap in valid_actions:
                        if at == "basic":
                            action_params = ap
                            break
                    else:
                        # If no valid basic action, revert to 'stop'
                        action_type = "stop"
                        action_params = None
            else:
                # For other action types, find the first matching action
                for at, ap in valid_actions:
                    if at == action_type:
                        action_params = ap
                        break
                else:
                    # If no matching action, revert to 'stop'
                    action_type = "stop"
                    action_params = None

            # Compute logarithmic probability
            if action_type in config.ACTION_TYPES:
                type_idx = config.ACTION_TYPES.index(action_type)
                log_prob = torch.log(action_type_probs[type_idx] + 1e-10)

                if action_type == "basic" and action_params in relation_to_idx:
                    rel_idx = relation_to_idx[action_params]
                    log_prob += torch.log(relation_probs[rel_idx] + 1e-10)
            else:
                log_prob = torch.tensor(0.0).to(config.DEVICE)

        # Log the selected action
        logger.debug(
            f"Selected action: {action_type}, {action_params}, log_prob={log_prob.item():.4f}"
        )

        return action_type, action_params, log_prob

    def save(self, path):
        """
        Save the model

        Parameters:
            path (str): Path to save the model
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "entity_embedding": self.entity_embedding.weight.data,
                "relation_embedding": self.relation_embedding.weight.data,
                "config": {
                    "entity_dim": self.entity_dim,
                    "relation_dim": self.relation_dim,
                    "hidden_dim": self.hidden_dim,
                },
            },
            path,
        )

        logger.info(f"Model saved to: {path}")

    @classmethod
    def load(cls, path, entity_count, relation_count):
        """
        Load the model

        Parameters:
            path (str): Path from which to load the model
            entity_count (int): Number of entities
            relation_count (int): Number of relations

        Returns:
            KGQAUnifiedAgent: The loaded model
        """
        checkpoint = torch.load(path, map_location=config.DEVICE)

        # Check if embedding sizes match
        entity_embeddings = checkpoint["entity_embedding"]
        relation_embeddings = checkpoint["relation_embedding"]

        if entity_embeddings.size(0) != entity_count:
            # If sizes don't match, reinitialize embeddings
            entity_embeddings = torch.randn(entity_count, entity_embeddings.size(1))
            logger.warning(
                f"Entity embedding size mismatch, reinitializing: {entity_embeddings.size(0)} vs {entity_count}"
            )

        if relation_embeddings.size(0) != relation_count:
            # If sizes don't match, reinitialize embeddings
            relation_embeddings = torch.randn(
                relation_count, relation_embeddings.size(1)
            )
            logger.warning(
                f"Relation embedding size mismatch, reinitializing: {relation_embeddings.size(0)} vs {relation_count}"
            )

        # Create the model
        model = cls(entity_embeddings, relation_embeddings)

        # Load the state dictionary
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            logger.info("Model state loaded successfully")
        except Exception as e:
            logger.error(f"Error loading state dictionary: {e}")
            logger.info("Using randomly initialized model")

        return model
