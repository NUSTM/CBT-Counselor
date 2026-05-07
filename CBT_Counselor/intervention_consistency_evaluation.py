from CBT_Counselor.clinical_safety_evaluation import get_response_by_chatgpt
from typing import List
import json
import re
import os



MAX_RETRIES = 2


PROMPT_PATH = os.path.join(os.path.dirname(__file__), "intervention_consistency_evaluation_prompt.txt")
with open(PROMPT_PATH, "r") as f:
    CBT_CONSISTENCY_SYSTEM_PROMPT = f.read()


CBT_CONSISTENCY_USER_PROMPT_TEMPLATE = """Please evaluate the cross-turn intervention consistency of the counselor's current response.

Presenting Problem: {presenting_problem}
Dialogue Stage: {dialogue_stage}

Dialogue History:
{dialogue_history}

Current Counselor Response to Evaluate:
CBT Skill: {predicted_skill}
Dialogue Strategy: {predicted_strategy}
Response: {generated_response}

Return ONLY JSON, no extra text."""


CBT_CONSISTENCY_BATCH_USER_PROMPT_TEMPLATE = """Please evaluate the cross-turn intervention consistency for the following {n} counselor responses. They all share the same dialogue context.

Presenting Problem: {presenting_problem}
Dialogue Stage: {dialogue_stage}

Dialogue History:
{dialogue_history}

Responses to Evaluate:
{responses_block}

Return ONLY a JSON array of {n} objects (one per response, in the same order), no extra text.
Each object must follow this schema exactly:
{{"core_belief_continuity": {{"score": <int 0-10>, "identified_beliefs": ["<belief>"], "rationale": "<brief>"}}, "stage_appropriateness": {{"score": <int 0-10>, "expected_skills": ["<skill>"], "rationale": "<brief>"}}, "strategy_coherence": {{"score": <int 0-10>, "prev_strategies": ["<strategy>"], "rationale": "<brief>"}}}}"""


def parse_consistency_judgment(raw_response: str) -> dict:
    try:
        result = json.loads(raw_response)
        return result
    except json.JSONDecodeError:
        json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        brace_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
    return {"parse_error": True}


def calculate_consistency_reward(
    presenting_problem: str,
    dialogue_stage: str,
    dialogue_history: str,
    predicted_skill: str,
    predicted_strategy: str,
    generated_response: str,
    weights: dict = None,
    max_retries: int = MAX_RETRIES
    ) -> dict:

    if weights is None:
        weights = {
            "core_belief_continuity": 1 / 3,
            "stage_appropriateness": 1 / 3,
            "strategy_coherence": 1 / 3
        }

    user_message = CBT_CONSISTENCY_USER_PROMPT_TEMPLATE.format(
        presenting_problem=presenting_problem,
        dialogue_stage=dialogue_stage,
        dialogue_history=dialogue_history,
        predicted_skill=predicted_skill,
        predicted_strategy=predicted_strategy,
        generated_response=generated_response
    )

    last_raw_response = ""
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_response = get_response_by_chatgpt(
                system_content=CBT_CONSISTENCY_SYSTEM_PROMPT,
                user_message=user_message
            )
            last_raw_response = raw_response

            judgment = parse_consistency_judgment(raw_response)

            if judgment.get("parse_error", False):
                last_error = f"Attempt {attempt}: JSON parse failed"
                print(f"[Consistency Reward] {last_error}, retrying...")
                continue

            
            scores = extract_scores(judgment)


            if any(v is None for v in scores.values()):
                last_error = f"Attempt {attempt}: missing scores {scores}"
                print(f"[Consistency Reward] {last_error}, retrying...")
                continue


            dim_rewards = {k: v / 10.0 for k, v in scores.items()}
            total_reward = sum(dim_rewards[k] * weights[k] for k in weights)

            return {
                "reward": round(total_reward, 4),
                "dim_scores": scores,
                "dim_rewards": {k: round(v, 4) for k, v in dim_rewards.items()},
                "judgment": judgment,
                "raw_response": raw_response,
                "attempts": attempt
            }

        except Exception as e:
            last_error = f"Attempt {attempt}: {e}"
            print(f"[Consistency Reward] {last_error}, retrying...")


    print(f"[Consistency Reward] All {max_retries} attempts failed. Last error: {last_error}")
    return {
        "reward": 0.5,
        "dim_scores": {"core_belief_continuity": None, "stage_appropriateness": None, "strategy_coherence": None},
        "dim_rewards": {"core_belief_continuity": None, "stage_appropriateness": None, "strategy_coherence": None},
        "judgment": {"parse_error": True, "last_error": str(last_error)},
        "raw_response": last_raw_response,
        "attempts": max_retries
    }


def extract_scores(judgment: dict) -> dict:

    scores = {}
    for key in ["core_belief_continuity", "stage_appropriateness", "strategy_coherence"]:
        dim = judgment.get(key, {})
        if isinstance(dim, dict):
            scores[key] = dim.get("score", None)
        else:
            scores[key] = None
    return scores


def _scores_to_reward(judgment: dict, weights: dict) -> float:

    if judgment.get("parse_error", False):
        return 0.5
    scores = extract_scores(judgment)
    if any(v is None for v in scores.values()):
        return 0.5
    dim_rewards = {k: v / 10.0 for k, v in scores.items()}
    return round(sum(dim_rewards[k] * weights[k] for k in weights), 4)


def calculate_consistency_reward_batch(
    presenting_problem: str,
    dialogue_stage: str,
    dialogue_history: str,
    responses: List[dict],          # [{"skill": str, "strategy": str, "response": str}, ...]
    weights: dict = None,
    max_retries: int = MAX_RETRIES,
) -> List[float]:
    
    if weights is None:
        weights = {
            "core_belief_continuity": 1 / 3,
            "stage_appropriateness": 1 / 3,
            "strategy_coherence": 1 / 3,
        }

    n = len(responses)
    if n == 0:
        return []

    lines = []
    for i, r in enumerate(responses, 1):
        lines.append(
            f"[Response {i}]\n"
            f"CBT Skill: {r['skill']}\n"
            f"Dialogue Strategy: {r['strategy']}\n"
            f"Response: {r['response']}"
        )
    responses_block = "\n\n".join(lines)

    user_message = CBT_CONSISTENCY_BATCH_USER_PROMPT_TEMPLATE.format(
        n=n,
        presenting_problem=presenting_problem,
        dialogue_stage=dialogue_stage,
        dialogue_history=dialogue_history,
        responses_block=responses_block,
    )

    last_raw_response = ""
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            raw_response = get_response_by_chatgpt(
                system_content=CBT_CONSISTENCY_SYSTEM_PROMPT,
                user_message=user_message,
            )
            last_raw_response = raw_response

            parsed = json.loads(raw_response)

            if isinstance(parsed, dict):
                for candidate_key in ("results", "evaluations", "responses", "items"):
                    if candidate_key in parsed and isinstance(parsed[candidate_key], list):
                        parsed = parsed[candidate_key]
                        break

                if isinstance(parsed, dict) and n == 1:
                    parsed = [parsed]

            if not isinstance(parsed, list) or len(parsed) != n:
                last_error = f"Attempt {attempt}: expected list of {n}, got {type(parsed).__name__} len={len(parsed) if isinstance(parsed, list) else '?'}"
                print(f"[Consistency Batch] {last_error}, retrying...")
                continue

            rewards = [_scores_to_reward(item, weights) for item in parsed]
            return rewards

        except json.JSONDecodeError:
            last_error = f"Attempt {attempt}: JSON parse failed"
            print(f"[Consistency Batch] {last_error}, retrying...")
        except Exception as e:
            last_error = f"Attempt {attempt}: {e}"
            print(f"[Consistency Batch] {last_error}, retrying...")

    print(f"[Consistency Batch] All {max_retries} attempts failed. Last error: {last_error}")

    rewards = []
    for r in responses:
        result = calculate_consistency_reward(
            presenting_problem=presenting_problem,
            dialogue_stage=dialogue_stage,
            dialogue_history=dialogue_history,
            predicted_skill=r["skill"],
            predicted_strategy=r["strategy"],
            generated_response=r["response"],
            weights=weights,
        )
        rewards.append(result["reward"])
    return rewards