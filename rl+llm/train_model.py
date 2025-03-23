import os
import torch
from datetime import datetime

import config
from data_utils import load_knowledge_graph, load_qa_data, load_embeddings
from kg_env import KnowledgeGraphEnv
from model import KGQAUnifiedAgent
from llm_reward import LLMRewardFunction
from trainer import KGQATrainer

# 加載知識圖譜
print("加載知識圖譜...")
kb_path = os.path.join(config.DATA_DIR, "kb.txt")
(
    kg_dict,
    entities,
    relations,
    kg_stats,
    entity_to_idx,
    relation_to_idx,
    entity_variants,
) = load_knowledge_graph(kb_path)

# 加載訓練數據
print("加載訓練數據...")
train_qa_path = os.path.join(config.DATA_DIR, "1-hop/vanilla/qa_train.txt")
(
    train_questions,
    train_answers,
    train_entities,
    processed_questions,
    question_types,
    question_focuses,
) = load_qa_data(train_qa_path)

# 加載驗證數據
print("加載驗證數據...")
dev_qa_path = os.path.join(config.DATA_DIR, "1-hop/vanilla/qa_dev.txt")
dev_questions, dev_answers, dev_entities, dev_processed, dev_types, dev_focuses = (
    load_qa_data(dev_qa_path)
)

# 創建環境
print("創建環境...")
env = KnowledgeGraphEnv(
    kg_dict, entities, relations, entity_to_idx, relation_to_idx, entity_variants
)

# 創建模型
print("創建模型...")
entity_embeddings, relation_embeddings = load_embeddings(entities, relations)
agent = KGQAUnifiedAgent(entity_embeddings, relation_embeddings)

# 創建LLM獎勵函數
print("創建LLM獎勵函數...")
llm_reward_fn = LLMRewardFunction()

# 創建訓練器
print("創建訓練器...")
trainer = KGQATrainer(agent, env, llm_reward_fn, entity_to_idx, relation_to_idx)

# 訓練模型
print("開始訓練...")
stats = trainer.train(
    train_questions[:10000],  # 可以限制樣本數量加快訓練
    train_answers[:10000],
    train_entities[:10000],
    dev_questions[:1000],
    dev_answers[:1000],
    dev_entities[:1000],
    num_episodes=config.MAX_EPISODES,
    batch_size=config.BATCH_SIZE,
)

# 保存訓練結果圖表
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
vis_path = os.path.join(config.LOG_DIR, f"training_vis_{timestamp}.png")
trainer.visualize_training(save_path=vis_path)

print(f"訓練完成！結果已保存到: {vis_path}")
