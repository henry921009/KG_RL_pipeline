"""
LLM reward function, used to evaluate actions and generate rewards
"""

import json
import time
import openai
import torch
import numpy as np
import logging
from tqdm import tqdm
import config
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("llm_reward.log"), logging.StreamHandler()],
)
logger = logging.getLogger("LLMReward")


class LLMRewardFunction:
    """
    Use LLM as reward function and guidance
    """

    def __init__(
        self, api_key=None, model=None, temperature=None, max_retries=3, retry_delay=5
    ):
        """
        Initialize LLM reward function

        Parameters:
            api_key (str): OpenAI API key
            model (str): Model name to use
            temperature (float): Generation temperature
            max_retries (int): Maximum number of retries
            retry_delay (int): Retry delay (seconds)
        """
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.LLM_MODEL
        self.temperature = temperature or config.LLM_TEMPERATURE
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Set API key
        openai.api_key = self.api_key

        # Cache
        self.guidance_cache = {}
        self.reward_cache = {}
        self.explanation_cache = {}

        # Statistics
        self.api_calls = 0
        self.cache_hits = 0
        self.api_errors = 0

        # Cache file paths
        self.cache_dir = os.path.join(config.LOG_DIR, "llm_cache")
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        self.guidance_cache_file = os.path.join(self.cache_dir, "guidance_cache.json")
        self.reward_cache_file = os.path.join(self.cache_dir, "reward_cache.json")

        # Load cache
        self._load_cache()

        logger.info(f"LLM reward function initialized, using model: {self.model}")

    def _load_cache(self):
        """Load cache files"""
        if os.path.exists(self.guidance_cache_file):
            try:
                with open(self.guidance_cache_file, "r", encoding="utf-8") as f:
                    # Convert string keys to tuple keys
                    str_cache = json.load(f)
                    for k, v in str_cache.items():
                        try:
                            # Attempt to convert string key to tuple
                            parts = k.split("|||||")
                            if len(parts) >= 3:
                                question = parts[0]
                                entities = tuple(parts[1].split(","))
                                history = tuple(eval(parts[2]))
                                actions = (
                                    tuple(eval(parts[3])) if len(parts) > 3 else ()
                                )

                                key = (question, entities, history, actions)
                                self.guidance_cache[key] = v
                        except Exception as e:
                            logger.warning(
                                f"Unable to parse cache key: {k}, error: {e}"
                            )
                logger.info(
                    f"Loaded {len(self.guidance_cache)} guidance cache entries from {self.guidance_cache_file}"
                )
            except Exception as e:
                logger.error(f"Error loading guidance cache: {e}")

        if os.path.exists(self.reward_cache_file):
            try:
                with open(self.reward_cache_file, "r", encoding="utf-8") as f:
                    str_cache = json.load(f)
                    for k, v in str_cache.items():
                        try:
                            parts = k.split("|||||")
                            if len(parts) >= 2:
                                question = parts[0]
                                answers = tuple(sorted(parts[1].split(",")))

                                key = (question, answers)
                                self.reward_cache[key] = v
                        except Exception as e:
                            logger.warning(
                                f"Unable to parse cache key: {k}, error: {e}"
                            )
                logger.info(
                    f"Loaded {len(self.reward_cache)} reward cache entries from {self.reward_cache_file}"
                )
            except Exception as e:
                logger.error(f"Error loading reward cache: {e}")

    def _save_cache(self):
        """Save cache to file"""
        try:
            # Convert tuple keys to string keys
            str_guidance_cache = {}
            for k, v in self.guidance_cache.items():
                try:
                    question, entities, history, actions = k
                    key = f"{question}|||||{','.join(entities)}|||||{history}|||||{actions}"
                    str_guidance_cache[key] = v
                except Exception as e:
                    logger.warning(f"Unable to serialize cache key: {k}, error: {e}")

            with open(self.guidance_cache_file, "w", encoding="utf-8") as f:
                json.dump(str_guidance_cache, f, ensure_ascii=False)

            str_reward_cache = {}
            for k, v in self.reward_cache.items():
                try:
                    question, answers = k
                    key = f"{question}|||||{','.join(answers)}"
                    str_reward_cache[key] = v
                except Exception as e:
                    logger.warning(f"Unable to serialize cache key: {k}, error: {e}")

            with open(self.reward_cache_file, "w", encoding="utf-8") as f:
                json.dump(str_reward_cache, f, ensure_ascii=False)

            logger.info(
                f"Cache saved to file: {len(self.guidance_cache)} guidance entries, {len(self.reward_cache)} reward entries"
            )
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    def get_action_guidance(self, state, valid_actions):
        """
        Get LLM guidance on actions

        Parameters:
            state (dict): Current state
            valid_actions (list): List of valid actions [(action_type, action_params), ...]

        Returns:
            list: A score (0-1) for each action
        """
        # If no valid actions, return an empty list
        if not valid_actions:
            return []

        # Construct cache key
        current_entities = state.get("current_entities", [])[
            :5
        ]  # Limit number of entities to avoid overly long cache key
        path_history = state.get("path_history", [])

        # Convert complex objects to hashable types
        cache_key = (
            state["question"],
            tuple(current_entities),
            tuple((a_type, str(a_params)) for a_type, a_params in path_history),
            tuple((a_type, str(a_params)) for a_type, a_params in valid_actions),
        )

        # Check cache
        if cache_key in self.guidance_cache:
            self.cache_hits += 1
            logger.debug(f"Guidance cache hit: {cache_key[0][:30]}...")
            return self.guidance_cache[cache_key]

        # Identify question type and focus
        question_type = state.get(
            "question_type", self._identify_question_type(state["question"])
        )
        question_focus = state.get(
            "question_focus", self._identify_question_focus(state["question"])
        )

        # Build text representation of the question history
        path_text = ""
        for i, (action_type, params) in enumerate(path_history):
            if action_type == "basic":
                path_text += f"Step {i + 1}: move along relation '{params}'\n"
            elif action_type == "filter":
                rel, val, op = params
                path_text += f"Step {i + 1}: filter condition '{rel} {op} {val}'\n"
            elif action_type == "union":
                rel1, rel2 = params
                path_text += (
                    f"Step {i + 1}: union of relations '{rel1}' and '{rel2}' results\n"
                )
            elif action_type == "aggregation":
                path_text += f"Step {i + 1}: aggregation operation '{params}'\n"
            elif action_type == "ordinal":
                sort_rel, order, pos = params
                path_text += f"Step {i + 1}: sort '{sort_rel}' ({order}) and select position {pos}\n"

        # Build text representation of actions
        action_texts = []
        for i, (action_type, params) in enumerate(valid_actions):
            if action_type == "basic":
                action_texts.append(f"{i + 1}. move along relation '{params}'")
            elif action_type == "filter":
                rel, val, op = params
                action_texts.append(f"{i + 1}. filter condition '{rel} {op} {val}'")
            elif action_type == "union":
                rel1, rel2 = params
                action_texts.append(
                    f"{i + 1}. union of relations '{rel1}' and '{rel2}' results"
                )
            elif action_type == "aggregation":
                action_texts.append(f"{i + 1}. aggregation operation '{params}'")
            elif action_type == "ordinal":
                sort_rel, order, pos = params
                action_texts.append(
                    f"{i + 1}. sort '{sort_rel}' ({order}) and select position {pos}"
                )
            elif action_type == "stop":
                action_texts.append(
                    f"{i + 1}. stop and return current entity as answer"
                )

        # Build prompt based on specific question type
        type_guidance = ""
        if question_type == "who":
            type_guidance = """
對於"who"類型問題:
1. 最有用的操作是沿著指向人物的關系前進，如"directed_by"、"starring"等。
2. 如果當前實體是電影，應該尋找指向人物的關系。
3. 如果當前實體已經是人物，可以考慮停止操作。
4. 聚合操作對"who"問題通常沒有幫助，因為我們需要找到具體人物而非數量。
"""
        elif question_type == "howmany":
            type_guidance = """
對於"how many"類型問題:
1. 首先需要沿著相關關系前進，找到符合條件的實體集合。
2. 然後使用"count"聚合操作來計算數量。
3. 過濾操作可能有用，用於篩選符合特定條件的實體。
4. 直接停止通常不是好選擇，除非已經執行了計數操作。
"""

        # Build guidance based on current step
        step_guidance = ""
        if len(path_history) == 0:
            step_guidance = """
第一步推薦:
1. 幾乎總是應該選擇"basic"操作來沿關系前進，尤其是與問題焦點相關的關系。
2. 避免在第一步就選擇"stop"或"aggregation"操作，這通常不會得到正確答案。
"""

        # Build prompt
        prompt = f"""
你是一個知識圖譜問答系統的專家，幫助我們理解如何在知識圖譜上找到答案。

問題: {state["question"]}
問題類型: {question_type}
問題焦點: {question_focus}

當前實體: {", ".join(current_entities)}

已走過的路徑:
{path_text if path_text else "（尚未采取任何動作）"}

可能的下一步動作:
{chr(10).join(action_texts)}

{type_guidance}
{step_guidance}

關於如何回答問題的重要知識:
1. 基本路徑操作(basic)用於沿著關係在知識圖譜中移動，是找到答案的關鍵步驟。
2. 過濾操作(filter)用於根據特定條件篩選實體。
3. 聚合操作(aggregation)用於對實體集合進行統計，如count、min、max等。
4. 停止操作(stop)用於返回當前實體作為最終答案，應在確信已找到答案時使用。

請為每個動作根據它們對回答上述問題的有用性給出0到1之間的分數。例如，與問題不相關的動作應該得分低，而直接指向答案的動作應該得分高。

只返回JSON格式的分數列表，例如:
{{"scores": [0.8, 0.2, 0.6, 0.1]}}
"""

        # Call API
        for attempt in range(self.max_retries):
            try:
                self.api_calls += 1
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一個專業的知識圖譜推理專家，擅長評估操作的相關性。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=150,
                )

                # Parse response
                response_text = response.choices[0].message.content.strip()
                logger.debug(f"LLM响应: {response_text}")

                # Try to parse JSON
                try:
                    scores_data = json.loads(response_text)
                    scores = scores_data.get("scores", [])

                    # Ensure scores count matches action count
                    if len(scores) != len(valid_actions):
                        # If not matching, try to fix
                        if len(scores) < len(valid_actions):
                            scores.extend([0.5] * (len(valid_actions) - len(scores)))
                        else:
                            scores = scores[: len(valid_actions)]

                    # Ensure all scores are in 0-1 range
                    scores = [max(0, min(1, s)) for s in scores]

                    # Adjust scores based on question type and step
                    adjusted_scores = self._adjust_scores_by_context(
                        scores,
                        valid_actions,
                        question_type,
                        question_focus,
                        len(path_history),
                    )

                    # Cache result
                    self.guidance_cache[cache_key] = adjusted_scores

                    # Periodically save cache
                    if self.api_calls % 10 == 0:
                        self._save_cache()

                    return adjusted_scores

                except json.JSONDecodeError:
                    # If JSON cannot be parsed, retry
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        # Last attempt, return default scores
                        default_scores = [0.5] * len(valid_actions)
                        self.guidance_cache[cache_key] = default_scores
                        return default_scores

            except Exception as e:
                self.api_errors += 1
                logger.error(
                    f"Error calling LLM API (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    # Final attempt failed, return default scores
                    default_scores = [0.5] * len(valid_actions)
                    self.guidance_cache[cache_key] = default_scores
                    return default_scores

    def _adjust_scores_by_context(
        self, scores, valid_actions, question_type, question_focus, path_length
    ):
        """
        Adjust scores based on context

        Parameters:
            scores (list): Original scores
            valid_actions (list): List of valid actions
            question_type (str): Type of the question
            question_focus (str): Focus of the question
            path_length (int): Current path length

        Returns:
            list: Adjusted scores
        """
        adjusted_scores = scores.copy()

        # Special handling for first step
        if path_length == 0:
            for i, (action_type, _) in enumerate(valid_actions):
                # Avoid selecting 'stop' in the first step
                if action_type == "stop":
                    adjusted_scores[i] *= 0.1  # Greatly reduce stop score

                # For "who" questions, enhance basic actions
                if question_type == "who" and action_type == "basic":
                    action_params = valid_actions[i][1]
                    if isinstance(action_params, str):
                        if (
                            question_focus == "director"
                            and "direct" in action_params.lower()
                        ):
                            adjusted_scores[i] = min(1.0, adjusted_scores[i] * 1.5)
                        elif question_focus == "actor" and any(
                            term in action_params.lower()
                            for term in ["star", "act", "play"]
                        ):
                            adjusted_scores[i] = min(1.0, adjusted_scores[i] * 1.5)

                # For "howmany" questions, do not directly use aggregation in the first step
                if question_type == "howmany" and action_type == "aggregation":
                    adjusted_scores[i] *= 0.5

        # Handling for subsequent steps
        else:
            for i, (action_type, action_params) in enumerate(valid_actions):
                # For "howmany" questions, increase weight for count aggregation in later steps
                if (
                    question_type == "howmany"
                    and action_type == "aggregation"
                    and action_params == "count"
                ):
                    adjusted_scores[i] = min(1.0, adjusted_scores[i] * 1.3)

        return adjusted_scores

    def get_final_reward(self, question, predicted_answers, true_answers=None):
        """
        Get final reward

        Parameters:
            question (str): The question
            predicted_answers (list): Predicted answers
            true_answers (list or None): True answers, if available

        Returns:
            float: Reward value (0-1)
        """
        # If true answers are provided, compute the F1 score directly
        if true_answers:
            predicted_set = set(predicted_answers)
            true_set = set(true_answers)

            if not predicted_set or not true_set:
                return 0.0

            precision = len(predicted_set.intersection(true_set)) / len(predicted_set)
            recall = len(predicted_set.intersection(true_set)) / len(true_set)

            if precision + recall == 0:
                return 0.0

            f1 = 2 * precision * recall / (precision + recall)
            logger.info(
                f"Final reward calculation: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}"
            )
            return f1

        # Construct cache key
        cache_key = (
            question,
            tuple(sorted(predicted_answers[:10])),
        )  # Use only first 10 answers to avoid overly long cache key

        # Check cache
        if cache_key in self.reward_cache:
            self.cache_hits += 1
            logger.debug(f"Reward cache hit: {cache_key[0][:30]}...")
            return self.reward_cache[cache_key]

        # Identify question type
        question_type = self._identify_question_type(question)
        question_focus = self._identify_question_focus(question)

        # Build prompt based on question type
        type_specific_prompt = ""
        if question_type == "who":
            type_specific_prompt = """
這是一個詢問人物的問題。好的答案應該是人名。
如果答案是電影名稱或其他非人物實體，評分應該較低。
"""
        elif question_type == "howmany":
            type_specific_prompt = """
這是一個詢問數量的問題。好的答案應該是一個數字。
如果答案是實體名稱而非數字，評分應該較低。
"""

        # Evaluation based on answer type
        answer_type_evaluation = ""
        # Check if predicted answers match the expected answer type
        is_numeric_answer = all(self._is_numeric(a) for a in predicted_answers)

        if question_type == "howmany" and not is_numeric_answer:
            answer_type_evaluation = "注意: 這個'how many'問題的答案應該是數字，但得到的是實體名稱，可能不正確。"
        elif (
            question_type != "howmany"
            and is_numeric_answer
            and len(predicted_answers) == 1
        ):
            answer_type_evaluation = "注意: 這個問題期望的答案是實體名稱，但得到的是數字，可能是聚合操作的結果，這對非計數問題可能不合適。"

        # Build prompt
        prompt = f"""
請評估以下問題的預測答案質量，給出0到1之間的分數。

問題: {question}
問題類型: {question_type}
問題焦點: {question_focus}
預測答案: {", ".join(predicted_answers[:10])}

{type_specific_prompt}
{answer_type_evaluation}

評分標準:
1. 答案是否直接回答了問題
2. 答案類型是否符合問題類型的預期
3. 答案的完整性和準確性

請給出0到1之間的一個數字作為評分，不需要任何解釋。
"""

        # Call API
        for attempt in range(self.max_retries):
            try:
                self.api_calls += 1
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "你是一个专业的问答系统评估者，擅长评估答案的质量和相关性。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=10,
                )

                # Parse response
                response_text = response.choices[0].message.content.strip()
                logger.debug(f"LLM响应: {response_text}")

                # Try to extract number
                try:
                    import re

                    score_match = re.search(r"(\d+(\.\d+)?)", response_text)
                    if score_match:
                        reward = float(score_match.group(1))
                        reward = float(score_match.group(1))
                        # Ensure reward is within 0-1 range
                        reward = max(0, min(1, reward))
                    else:
                        # Use default value if no number is found
                        reward = 0.5
                        logger.warning(
                            f"Unable to extract number from response: {response_text}"
                        )

                    # Cache result
                    self.reward_cache[cache_key] = reward

                    # Periodically save cache
                    if self.api_calls % 10 == 0:
                        self._save_cache()

                    return reward

                except ValueError:
                    # Retry if unable to parse as float
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        # Final attempt failed, return default reward
                        self.reward_cache[cache_key] = 0.5
                        return 0.5

            except Exception as e:
                self.api_errors += 1
                logger.error(
                    f"Error calling LLM API (attempt {attempt + 1}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    # Final attempt failed, return default reward
                    self.reward_cache[cache_key] = 0.5
                    return 0.5

    def get_action_explanation(self, state, action_type, action_params):
        """
        Get LLM explanation for an action

        Parameters:
            state (dict): Current state
            action_type (str): Type of action
            action_params: Parameters of the action

        Returns:
            str: Explanation text
        """
        # Construct cache key
        current_entities = state.get("current_entities", [])[:5]
        cache_key = (
            state["question"],
            tuple(current_entities),
            action_type,
            str(action_params),
        )

        # Check cache
        if cache_key in self.explanation_cache:
            self.cache_hits += 1
            return self.explanation_cache[cache_key]

        # Build action text
        if action_type == "basic":
            action_text = f"沿關係 '{action_params}' 前進"
        elif action_type == "filter":
            rel, val, op = action_params
            action_text = f"過濾條件 '{rel} {op} {val}'"
        elif action_type == "union":
            rel1, rel2 = action_params
            action_text = f"合併關係 '{rel1}' 和 '{rel2}' 的結果"
        elif action_type == "aggregation":
            action_text = f"聚合操作 '{action_params}'"
        elif action_type == "ordinal":
            sort_rel, order, pos = action_params
            action_text = f"排序 '{sort_rel}' ({order}) 並選擇位置 {pos}"
        elif action_type == "stop":
            action_text = "停止並返回當前實體作為答案"
        else:
            action_text = f"未知操作 '{action_type}'"

        # Build text representation of path history
        path_text = ""
        for i, (a_type, params) in enumerate(state.get("path_history", [])):
            if a_type == "basic":
                path_text += f"步驟{i + 1}: 沿關係 '{params}' 前進\n"
            elif a_type == "filter":
                rel, val, op = params
                path_text += f"步驟{i + 1}: 過濾條件 '{rel} {op} {val}'\n"
            elif a_type == "union":
                rel1, rel2 = params
                path_text += f"步驟{i + 1}: 合併關係 '{rel1}' 和 '{rel2}' 的結果\n"
            elif a_type == "aggregation":
                path_text += f"步驟{i + 1}: 聚合操作 '{params}'\n"
            elif a_type == "ordinal":
                sort_rel, order, pos = params
                path_text += (
                    f"步驟{i + 1}: 排序 '{sort_rel}' ({order}) 並選擇位置 {pos}\n"
                )

        # Build prompt
        prompt = f"""
請解釋為什麽以下動作對回答問題有幫助或沒有幫助。

問題: {state["question"]}

當前實體: {", ".join(current_entities)}

已走過的路徑:
{path_text if path_text else "（尚未采取任何動作）"}

選擇的動作: {action_text}

請詳細解釋這個動作如何幫助或阻礙回答問題。分析其合理性、相關性和可能的效果。
"""

        # Call API
        try:
            self.api_calls += 1
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一個專業的知識圖譜推理專家，擅長解釋操作的意義和合理性。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=150,
            )

            # Parse response
            explanation = response.choices[0].message.content.strip()

            # Cache result
            self.explanation_cache[cache_key] = explanation

            return explanation

        except Exception as e:
            self.api_errors += 1
            logger.error(f"Error calling LLM API: {e}")
            return "無法獲取解釋。"

    def _identify_question_type(self, question):
        """
        Identify the question type

        Parameters:
            question (str): The question text

        Returns:
            str: The question type
        """
        question_lower = question.lower()

        if question_lower.startswith("who"):
            return "who"
        elif question_lower.startswith("what"):
            return "what"
        elif question_lower.startswith("where"):
            return "where"
        elif question_lower.startswith("when"):
            return "when"
        elif "how many" in question_lower:
            return "howmany"
        elif question_lower.startswith("which"):
            return "which"

        return "general"

    def _identify_question_focus(self, question):
        """
        Identify the question focus

        Parameters:
            question (str): The question text

        Returns:
            str: The question focus
        """
        question_lower = question.lower()

        if "direct" in question_lower:
            return "director"
        elif any(term in question_lower for term in ["star", "act in", "play in"]):
            return "actor"
        elif any(term in question_lower for term in ["where", "location", "country"]):
            return "location"
        elif any(term in question_lower for term in ["when", "year", "date"]):
            return "time"
        elif "how many" in question_lower:
            return "count"

        # Infer focus based on question type
        if question_lower.startswith("who"):
            return "person"
        elif question_lower.startswith("where"):
            return "location"
        elif question_lower.startswith("when"):
            return "time"

        return "entity"

    def _is_numeric(self, value):
        """Check if the value is numeric"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def precompute_guidance(
        self,
        env,
        questions,
        question_entities,
        question_types=None,
        question_focuses=None,
        batch_size=10,
        max_questions=None,
    ):
        """
        Precompute LLM guidance for a batch of questions to accelerate training

        Parameters:
            env (KnowledgeGraphEnv): Knowledge graph environment
            questions (list): List of questions
            question_entities (list): List of question entities
            question_types (list): List of question types
            question_focuses (list): List of question focuses
            batch_size (int): Batch size
            max_questions (int): Maximum number of questions

        Returns:
            dict: Precomputed guidance cache
        """
        if max_questions and max_questions < len(questions):
            indices = np.random.choice(len(questions), max_questions, replace=False)
            questions = [questions[i] for i in indices]
            question_entities = [question_entities[i] for i in indices]
            if question_types:
                question_types = [question_types[i] for i in indices]
            if question_focuses:
                question_focuses = [question_focuses[i] for i in indices]

        print(f"預計算LLM指導 for {len(questions)} questions...")

        results = []

        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=min(batch_size, 5)) as executor:
            futures = []

            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i : i + batch_size]
                batch_entities = question_entities[i : i + batch_size]
                batch_types = (
                    question_types[i : i + batch_size]
                    if question_types
                    else [None] * len(batch_questions)
                )
                batch_focuses = (
                    question_focuses[i : i + batch_size]
                    if question_focuses
                    else [None] * len(batch_questions)
                )

                for j, (question, entity, q_type, q_focus) in enumerate(
                    zip(batch_questions, batch_entities, batch_types, batch_focuses)
                ):
                    # Submit task to thread pool
                    future = executor.submit(
                        self._precompute_single_question,
                        env,
                        question,
                        entity,
                        q_type,
                        q_focus,
                        i + j,
                    )
                    futures.append(future)

            # Collect results
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="預計算LLM指導"
            ):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing question: {e}")

        # Save cache
        self._save_cache()

        print(f"Precomputation complete, cache size: {len(self.guidance_cache)}")
        return results

    def _precompute_single_question(
        self, env, question, entity, question_type, question_focus, idx
    ):
        """Process precomputation for a single question"""
        try:
            # Reset environment
            state = env.reset(question, entity, question_type, question_focus)

            # Get valid actions
            valid_actions = env.get_valid_actions()

            # Get LLM guidance
            action_scores = self.get_action_guidance(state, valid_actions)

            return {
                "index": idx,
                "question": question,
                "entity": entity,
                "valid_actions_count": len(valid_actions),
                "action_scores": action_scores,
            }
        except Exception as e:
            logger.error(f"Error processing question {idx}: {e}")
            return {
                "index": idx,
                "question": question,
                "entity": entity,
                "error": str(e),
            }

    def get_stats(self):
        """
        Get LLM usage statistics

        Returns:
            dict: Statistics information
        """
        return {
            "api_calls": self.api_calls,
            "cache_hits": self.cache_hits,
            "api_errors": self.api_errors,
            "guidance_cache_size": len(self.guidance_cache),
            "reward_cache_size": len(self.reward_cache),
            "explanation_cache_size": len(self.explanation_cache),
        }
