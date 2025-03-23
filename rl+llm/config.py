"""
Configuration file that contains all hyperparameters and settings
"""

import os
import torch

# Path configuration
DATA_DIR = "./data"  # data directory
KB_PATH = os.path.join(DATA_DIR, "kb.txt")
TRAIN_QA_PATH = os.path.join(DATA_DIR, "1-hop/vanilla/qa_train.txt")
DEV_QA_PATH = os.path.join(DATA_DIR, "1-hop/vanilla/qa_dev.txt")
TEST_QA_PATH = os.path.join(DATA_DIR, "1-hop/vanilla/qa_test.txt")

# Model configuration
ENTITY_DIM = 200  # entity embedding dimension
RELATION_DIM = 200  # relation embedding dimension
HIDDEN_DIM = 256  # hidden layer dimension
DROPOUT = 0.1  # dropout rate

# Training configuration
LEARNING_RATE = 0.001
BATCH_SIZE = 32
MAX_EPISODES = 10000  # maximum number of training episodes
GAMMA = 0.99  # reward discount factor
EPSILON_START = 0.9  # initial exploration rate
EPSILON_END = 0.05  # final exploration rate
EPSILON_DECAY = 5000  # steps for exploration rate decay
MAX_STEPS = 3  # maximum steps per episode

# LLM configuration
OPENAI_API_KEY = os.environ.get(
    "OPENAI_API_KEY", ""
)  # read API key from environment variable
LLM_MODEL = "gpt-3.5-turbo"  # or other available models
LLM_TEMPERATURE = 0  # generation temperature, 0 indicates deterministic output
LLM_REWARD_WEIGHT = 0.7  # LLM reward weight (increased weight)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Operation types
ACTION_TYPES = ["basic", "filter", "union", "aggregation", "ordinal", "stop"]

# Logging configuration
LOG_INTERVAL = 10  # log every number of episodes
SAVE_INTERVAL = 100  # save the model every number of episodes
MODEL_SAVE_DIR = "./models"  # model save directory
LOG_DIR = "./logs"  # log save directory

# Entity resolution settings
ENTITY_MATCH_THRESHOLD = 0.8  # entity matching threshold
MAX_ENTITY_CANDIDATES = 5  # maximum number of entity candidates

# Debug settings
DEBUG_MODE = True  # enable debug mode
VERBOSE_OUTPUT = True  # verbose output

# Create necessary directories
for dir_path in [DATA_DIR, MODEL_SAVE_DIR, LOG_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
