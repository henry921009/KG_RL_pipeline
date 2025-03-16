MetaQA data: https://drive.google.com/drive/folders/0B-36Uca2AvwhTWVFSUZqRXVtbUE?resourcekey=0-kdv6ho5KcpEXdI2aUdLn_g&usp=sharing

# 3/16
## 檔案結構
- `config.py` - 配置參數
- `data_utils.py` - 數據加載與處理
- `kg_env.py` - 知識圖譜環境
- `model.py` - RL模型
- `llm_reward.py` - LLM獎勵函數
- `evaluator.py` - 評估邏輯
- `test_system.py` - 系統測試script
- `run_demo.py` - demo script
- `run_demo.sh` - 快速run script
------------------------------------------------------
(先產code，尚未正式進行訓練)
- `trainer.py` - 訓練器
- `train_model.py` - 訓練script

## 目前實現的操作
- **Basic** - 沿關係路徑前進
- **Filter** - 基於條件過濾實體
- **Union** - 合併多個關係的結果
- **Aggregation** - 執行聚合操作（如count）
- **Ordinal** - 執行排序操作
- **Stop** - 終止推理並返回答案
