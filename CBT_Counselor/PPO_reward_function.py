import re
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
import torch.nn.functional as F
from bert_score import BERTScorer
from rouge import Rouge
from CBT_Counselor.load_pretrained_model import BERT_DIR


from CBT_Counselor.clinical_safety_evaluation import (
    calculate_clinical_safety_reward,
    assess_dialogue_context,
    evaluate_response_with_context,
)


from CBT_Counselor.load_pretrained_model import EPITOME_SAVE_DIR
from CBT_Counselor.epitome.empathy_scorer import EmpathyScorer
from CBT_Counselor.evaluation_metric import get_epitome_score


from CBT_Counselor.intervention_consistency_evaluation import (
    calculate_consistency_reward,
    calculate_consistency_reward_batch,
)



class CounselorRewardCalculator:

    def __init__(self):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.rouge = Rouge()
        
        self.bert_scorer = BERTScorer(
            model_type=BERT_DIR,
            num_layers=12,
            batch_size=64,
            nthreads=4,
            all_layers=False,
            idf=False,
            device=self.device,
            lang='en',
            rescale_with_baseline=False
        )

        self.category_map = {
            'cognitive distortion identification': 'cognitive',
            'cognitive restructuring': 'cognitive', 
            'socratic questioning': 'cognitive',
            'problem-solving & functional analysis': 'behavioral',
            'positive reinforcement & shaping': 'behavioral',
            'skills & role training': 'behavioral',
            'other behavioral interventions': 'behavioral',
            'emotion identification & labeling': 'emotional',
            'emotion & acceptance skills': 'emotional',
            'psychoeducation': 'educational',
            'mood & behavior monitoring': 'educational'
        }
        
        self.bert_thresholds = {
            'min': 0.48,
            'good': 0.52,
            'excellent': 0.58
        }
        
        self.rouge_thresholds = {
            'min': 0.08,
            'good': 0.12,
            'excellent': 0.15
        }


        self.skill_groups = {
            'engagement': ['understanding', 'interpersonal effectiveness'],
            'exploration': ['guided discovery', 'focus on key cognitions & behaviors'],
            'intervention': ['strategy for change', 'action plan'],
            'other': ['others']
        }
        

        self.skill_to_group = {}
        for group, skills in self.skill_groups.items():
            for skill in skills:
                self.skill_to_group[skill.lower()] = group

        try:
            use_cuda = torch.cuda.is_available()
            self.epitome_empathy_scorer = EmpathyScorer(EPITOME_SAVE_DIR, batch_size=64, use_cuda=use_cuda)
        except Exception as e:
            print(f"[Init] EmpathyScorer loading failing: {e}")
            self.epitome_empathy_scorer = None


    def _calculate_clinical_safety_reward(
        self,
        prompts: list[str],
        completions: list[str],
        reward_pass: float = 0.05,
        reward_fail: float = 0.0,
        ) -> list[float]:

        n = len(prompts)
        rewards = [0.0] * n

        # ── Step 1: get context_assessment ──────────────
        unique_contexts = list(dict.fromkeys(prompts)) 

        context_assessment_cache: dict[str, dict] = {}

        def _assess_context(ctx: str):
            try:
                result = assess_dialogue_context(ctx)
                return ctx, result["context_assessment"]
            except Exception as e:
                print(f"[Clinical Safety Reward] context assessment error: {e}")
                return ctx, {"cssrs_level": 1, "overall_risk": "moderate"}

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(_assess_context, ctx): ctx for ctx in unique_contexts}
            for future in as_completed(futures):
                ctx, assessment = future.result()
                context_assessment_cache[ctx] = assessment

        # ── Step 2: evaluate response for each completion ─────────
        def _evaluate_one(idx):
            try:
                ctx_assessment = context_assessment_cache.get(prompts[idx], {"cssrs_level": 1, "overall_risk": "moderate"})
                result = evaluate_response_with_context(
                    context_assessment=ctx_assessment,
                    model_response=completions[idx],
                    reward_pass=reward_pass,
                    reward_fail=reward_fail,
                )
                return idx, result["reward"]
            except Exception as e:
                print(f"[Clinical Safety Reward] response evaluation error: {e}")
                return idx, 0.0

        with ThreadPoolExecutor(max_workers=32) as executor:
            futures = {executor.submit(_evaluate_one, i): i for i in range(n)}
            for future in as_completed(futures):
                idx, reward = future.result()
                rewards[idx] = reward

        return rewards


    def _calculate_intervention_consistency_reward(
        self,
        presenting_problem: str,
        dialogue_stage: str,
        annotated_dialogue_history: str,
        predicted_skill: str,
        predicted_strategy: str,
        generated_response: str
        ) -> float:

        try:
            result = calculate_consistency_reward(
                presenting_problem=presenting_problem,
                dialogue_stage=dialogue_stage,
                dialogue_history=annotated_dialogue_history,
                predicted_skill=predicted_skill,
                predicted_strategy=predicted_strategy,
                generated_response=generated_response
            )
            return result["reward"]
        except Exception as e:
            print(f"[Consistency Reward] Error: {e}")
            return 0.5


    def _check_format(self, text: str) -> bool:

        if not text or not isinstance(text, str):
            return False
        
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        
        has_intervention = "<counselor_intervention>" in t and "</counselor_intervention>" in t
        has_cbt_skill = "<cbt_skill>" in t and "</cbt_skill>" in t
        has_strategy = "<dialogue_strategy>" in t and "</dialogue_strategy>" in t
        has_response = "<counselor_response>" in t or "</counselor_response>" in t
        
        return has_intervention and has_cbt_skill and has_strategy and has_response
    
    def _extract_cbt_skill(self, text: str) -> str:

        if not text or not isinstance(text, str):
            return "none"

        t = text.replace("\r\n", "\n").replace("\r", "\n")
        m_resp = re.search(r"<cbt_skill>\s*(.*?)\s*</cbt_skill>", t, flags=re.DOTALL | re.IGNORECASE)
        
        if m_resp:
            cbt_skill = m_resp.group(1).strip().strip('"').strip("'")
            cbt_skill = cbt_skill.lower()
            if "and" in cbt_skill:
                cbt_skill = cbt_skill.replace("and", "&")
            if cbt_skill:
                return cbt_skill

        return "none"

    def _extract_dialog_strategy(self, text: str) -> str:

        if not text or not isinstance(text, str):
            return "none"

        t = text.replace("\r\n", "\n").replace("\r", "\n")
        m_resp = re.search(r"<dialogue_strategy>\s*(.*?)\s*</dialogue_strategy>", t, flags=re.DOTALL | re.IGNORECASE)
        
        if m_resp:
            dialog_strategy = m_resp.group(1).strip().strip('"').strip("'")
            dialog_strategy = dialog_strategy.lower()
            if "and" in dialog_strategy:
                dialog_strategy = dialog_strategy.replace("and", "&")
            if dialog_strategy:
                return dialog_strategy

        return "none"


    def _extract_response(self, text: str) -> str:

        if not text or not isinstance(text, str):
            return "none"
    
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        m_resp = re.search(r"<counselor_response>\s*(.*?)\s*</counselor_response>", t, flags=re.DOTALL | re.IGNORECASE)
        if m_resp:
            response = m_resp.group(1).strip().strip('"').strip("'")
            if response:
                return response
        
        incomplete_start = re.search(r"<counselor_response>\s*(.*?)$", t, flags=re.DOTALL | re.IGNORECASE)
        if incomplete_start:
            response = incomplete_start.group(1).strip().strip('"').strip("'")
            if response:  
                return response
            
        return t.strip()


    def _normalize_skill_strategy(self, text: str) -> str:

        text = text.lower().strip()
        text = text.replace("and", "&")
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s&]', '', text)
        return text



    def _calculate_relative_length_factor(self, response_length: int, 
                                        reference_length: int) -> float:

        length_ratio = response_length / max(reference_length, 1)
        
        if length_ratio < 0.5:
            return 0.7
        elif length_ratio <= 1.2:
            return 1.0
        elif length_ratio <= 2.0:
            return 1.0 - 0.4 * (length_ratio - 1.2) / 0.8
        elif length_ratio <= 4.0:
            return 0.6 - 0.35 * (length_ratio - 2.0) / 2.0
        else:
            return max(0.1, 0.25 - 0.03 * (length_ratio - 4.0))


    def _calculate_response_content_score(self, rouge_score: float, bert_score: float, 
                                response_length: int, reference_length: int) -> float:

        if rouge_score < self.rouge_thresholds['min']:
            rouge_component = 0.0
        elif rouge_score < self.rouge_thresholds['excellent']:
            rouge_component = (rouge_score - self.rouge_thresholds['min']) / \
                            (self.rouge_thresholds['excellent'] - self.rouge_thresholds['min'])
        else:
            rouge_component = 1.0 + (rouge_score - self.rouge_thresholds['excellent']) * 2.0
            rouge_component = min(1.2, rouge_component)
        
        if bert_score < self.bert_thresholds['min']:
            bert_component = 0.0
        elif bert_score < self.bert_thresholds['excellent']:
            bert_component = (bert_score - self.bert_thresholds['min']) / \
                            (self.bert_thresholds['excellent'] - self.bert_thresholds['min'])
        else:
            bert_component = 1.0 + (bert_score - self.bert_thresholds['excellent']) * 2.0
            bert_component = min(1.2, bert_component)
        
        content_score = rouge_component * 0.5 + bert_component * 0.5

        relative_factor = self._calculate_relative_length_factor(response_length, reference_length)


        ABSOLUTE_SOFT = 50   
        ABSOLUTE_HARD = 80  
        
        if response_length <= ABSOLUTE_SOFT:
            absolute_factor = 1.0
        elif response_length <= ABSOLUTE_HARD:
            absolute_factor = 1.0 - 0.3 * (response_length - ABSOLUTE_SOFT) / (ABSOLUTE_HARD - ABSOLUTE_SOFT)
        else:
            absolute_factor = max(0.3, 0.7 - 0.01 * (response_length - ABSOLUTE_HARD))
        
        length_factor = min(relative_factor, absolute_factor)
        final_score = content_score * length_factor
        
        return final_score


    def _calculate_professional_score(self, predicted_skill, predicted_strategy,
                                    expected_skill, expected_strategy,
                                    reasonable_skill, reasonable_strategy,
                                    predicted_category, expected_category):

        pred_skill = predicted_skill.lower()
        exp_skill = expected_skill.lower()
        reas_skill = reasonable_skill.lower()
        
        pred_skill_group = self.skill_to_group.get(pred_skill, 'unknown')
        exp_skill_group = self.skill_to_group.get(exp_skill, 'unknown')
        reas_skill_group = self.skill_to_group.get(reas_skill, 'unknown')


        if pred_skill == "none":
            skill_score = 0.0
        elif pred_skill == exp_skill:
            skill_score = 1.0
        elif pred_skill_group == exp_skill_group and pred_skill_group != 'unknown':
            if pred_skill == reas_skill:
                skill_score = 0.7
            else:
                skill_score = 0.5
        elif pred_skill == reas_skill:
            skill_score = 0.35
        else:
            skill_score = 0.1


        pred_strategy = predicted_strategy.lower()
        exp_strategy = expected_strategy.lower()
        reas_strategy = reasonable_strategy.lower()
        
        pred_strategy_category = self.category_map.get(pred_strategy, 'unknown')
        exp_strategy_category = self.category_map.get(exp_strategy, 'unknown')

        if pred_strategy == "none":
            strategy_score = 0.0
        elif pred_strategy == exp_strategy:
            strategy_score = 1.0
        elif pred_strategy_category == exp_strategy_category and pred_strategy_category != 'unknown':
            if pred_strategy == reas_strategy:
                strategy_score = 0.7
            else:
                strategy_score = 0.5
        elif pred_strategy == reas_strategy:
            strategy_score = 0.35
        else:
            strategy_score = 0.1
        
        return skill_score, strategy_score
    
    
    def _calculate_epitome_score(self, last_speaker_utterances: List[str], gt_responses: List[str], pred_responses: List[str]) -> List[float]:

        if not last_speaker_utterances or not gt_responses or not pred_responses:
            return [0.0] * len(pred_responses)
        
        if self.epitome_empathy_scorer is None:
            print("[Epitome] EmpathyScorer failed to load; returning default score")
            return [0.0] * len(pred_responses)
        
        data = []
        for last_speaker_utterance, gt_resp, pred_resp in zip(last_speaker_utterances, gt_responses, pred_responses):
            data.append({
                'utterance': last_speaker_utterance,
                'prediction': pred_resp,
                'gt': gt_resp
            })
        
        try:
            diff_IP_scores_for_reward, diff_EX_scores_for_reward, diff_ER_scores_for_reward = get_epitome_score(data, self.epitome_empathy_scorer, batch_size=self.epitome_empathy_scorer.batch_size)
            
            empathy_rewards = []
            
            for i in range(len(diff_IP_scores_for_reward)):
                ip_diff = diff_IP_scores_for_reward[i]
                ex_diff = diff_EX_scores_for_reward[i]
                er_diff = diff_ER_scores_for_reward[i]

                k = 0.5
                ip_score = np.exp(-k * ip_diff)  
                ex_score = np.exp(-k * ex_diff)
                er_score = np.exp(-k * er_diff)
                
                relative_score = (ip_relative + ex_relative + er_relative) / 3.0
                empathy_score = np.clip(relative_score, 0.0, 1.0)
                empathy_rewards.append(empathy_score)
            
            return empathy_rewards
            
        except Exception as e:
            print(f"Error computing empathy metrics: {e}")
            return [0.0] * len(pred_responses)


    def _calculate_cbt_intervention_diversity_bonus(self, intervention_diversity_weight: float, 
                                                    predicted_items: List[str], 
                                                    expected_items: List[str],
                                                    group_size: int = 6) -> List[float]:

        bonuses = []
        num_groups = len(predicted_items) // group_size
        
        for group_idx in range(num_groups):
            start_idx = group_idx * group_size
            end_idx = start_idx + group_size
            group_pred_items = predicted_items[start_idx:end_idx]
            group_exp_items = expected_items[start_idx:end_idx]
            
            unique_pred_items = set(group_pred_items)
            num_unique = len(unique_pred_items)
            
            if num_unique <= 1:
                base_bonus = 0.0
            else:
                ratio = (num_unique - 1) / max(group_size - 1, 1)
                ratio = min(ratio, 1.0)
                base_bonus = intervention_diversity_weight * ratio
            
            for pred, exp in zip(group_pred_items, group_exp_items):
                if pred == exp:
                    bonuses.append(base_bonus + 0.02)
                else:
                    bonuses.append(base_bonus)
        
        remainder = len(predicted_items) % group_size
        if remainder > 0:
            start_idx = num_groups * group_size
            for i in range(remainder):
                idx = start_idx + i
                if predicted_items[idx] == expected_items[idx]:
                    bonuses.append(0.02)
                else:
                    bonuses.append(0.0)
        return bonuses


    # =================================================================================
    # Main function
    # =================================================================================

    def calculate_reward(self, generated_responses: List[str], 
                            reference_data: List[dict],
                            tokenizer,
                            safety_reward: float,
                            format_reward: float,
                            response_content_reward: float,
                            epitome_reward: float,
                            intervention_consistency_reward: float,
                            group_size: int,
                            skill_reward: Optional[float] = None,
                            strategy_reward: Optional[float] = None,
                            intervention_diversity_weight: Optional[float] = None,) -> Tuple:

        num_responses = len(generated_responses)
        rewards = [0.0] * num_responses

        passed_gate = [False] * num_responses
        
        safety_scores = []
        format_scores = []
        skill_scores = []
        strategy_scores = []
        response_content_scores = []
        epitome_scores = []
        intervention_consistency_scores = []

        extracted_responses = [self._extract_response(resp) for resp in generated_responses]
        reference_responses = [data.get('ground_truth', '') for data in reference_data]
        dialogue_contexts = [data.get('dialogue_context', '') for data in reference_data]
        last_speaker_utterances = [data.get('last_speaker_utterance', '') for data in reference_data]

        predicted_skills = []
        predicted_strategies = []
        for generated in generated_responses:
            skill = self._normalize_skill_strategy(self._extract_cbt_skill(generated))
            strategy = self._normalize_skill_strategy(self._extract_dialog_strategy(generated))
            predicted_skills.append(skill)
            predicted_strategies.append(strategy)
        

        try:
            batch_rouge_scores = self.rouge.get_scores(extracted_responses, reference_responses)
            batch_rouge_l = [float(s['rouge-l']['f']) for s in batch_rouge_scores]
        except Exception:
            batch_rouge_l = []
            for hyp, ref in zip(extracted_responses, reference_responses):
                try:
                    s = self.rouge.get_scores(hyp, ref)[0]
                    batch_rouge_l.append(float(s['rouge-l']['f']))
                except Exception:
                    batch_rouge_l.append(0.0)

        all_texts = extracted_responses + reference_responses
        encoded = tokenizer(all_texts, add_special_tokens=False)
        token_lengths = [len(ids) for ids in encoded["input_ids"]]
        response_token_lens  = token_lengths[:num_responses]
        reference_token_lens = token_lengths[num_responses:]


        batch_safety_results           = [0.0] * num_responses
        batch_bert_scores              = [0.0] * num_responses
        epitome_results                = [0.0] * num_responses
        consistency_scores_precomputed = [0.5] * num_responses

        def _run_api_clinical_safety():
            return self._calculate_clinical_safety_reward(
                dialogue_contexts, extracted_responses,
                reward_pass=safety_reward, reward_fail=0.0
            )

        def _run_api_intervention_consistency():
            if intervention_consistency_reward is None:
                return [0.5] * num_responses

            context_to_indices: dict[tuple, list] = {}
            for idx in range(num_responses):
                ref = reference_data[idx]
                key = (
                    ref.get('presenting_problem', ''),
                    ref.get('dialogue_stage', ''),
                    ref.get('annotated_dialogue_history', ''),
                )
                context_to_indices.setdefault(key, []).append(idx)

            scores = [0.5] * num_responses

            def _batch_one_context(key, indices):
                presenting_problem, dialogue_stage, dialogue_history = key
                responses_list = [
                    {
                        "skill": predicted_skills[i],
                        "strategy": predicted_strategies[i],
                        "response": extracted_responses[i],
                    }
                    for i in indices
                ]
                batch_rewards = calculate_consistency_reward_batch(
                    presenting_problem=presenting_problem,
                    dialogue_stage=dialogue_stage,
                    dialogue_history=dialogue_history,
                    responses=responses_list,
                )
                return indices, batch_rewards

            with ThreadPoolExecutor(max_workers=32) as executor:
                futures = {
                    executor.submit(_batch_one_context, key, idxs): key
                    for key, idxs in context_to_indices.items()
                }
                for future in as_completed(futures):
                    try:
                        indices, batch_rewards = future.result()
                        for i, idx in enumerate(indices):
                            scores[idx] = batch_rewards[i]
                    except Exception as e:
                        print(f"[Consistency Batch] future failed: {e}")

            return scores

        with ThreadPoolExecutor(max_workers=2) as api_pool:
            
            f_safety      = api_pool.submit(_run_api_clinical_safety)
            f_consistency = api_pool.submit(_run_api_intervention_consistency)

            try:
                _, _, bert_f1_batch = self.bert_scorer.score(extracted_responses, reference_responses)
                batch_bert_scores = bert_f1_batch.tolist()
            except Exception as e:
                print(f"Batch BERT scoring error: {e}")

            if epitome_reward is not None:
                epitome_results = self._calculate_epitome_score(
                    last_speaker_utterances, reference_responses, extracted_responses
                )

            batch_safety_results           = f_safety.result()
            consistency_scores_precomputed = f_consistency.result()


        for i, (generated, ref_data) in enumerate(zip(generated_responses, reference_data)):
            score = 0.0
            extracted_response = extracted_responses[i]
            reference_response = reference_responses[i]

            # ============ 1. safety reward ============
            if batch_safety_results[i] != 0.0:
                score += batch_safety_results[i]
                safety_scores.append(batch_safety_results[i])
            else:
                safety_scores.append(0.0)
                format_scores.append(0.0)
                response_content_scores.append(0.0)
                epitome_scores.append(0.0)
                intervention_consistency_scores.append(0.0)
                if skill_reward is not None:
                    skill_scores.append(0.0)
                    strategy_scores.append(0.0)
                rewards[i] = 0.0
                continue

            # ============ 2. format reward ============
            if self._check_format(generated):
                score += format_reward
                format_scores.append(format_reward)
            else:
                format_scores.append(0.0)
                response_content_scores.append(0.0)
                epitome_scores.append(0.0)
                intervention_consistency_scores.append(0.0)
                if skill_reward is not None:
                    skill_scores.append(0.0)
                    strategy_scores.append(0.0)
                rewards[i] = 0.0
                continue
            
            passed_gate[i] = True
            
            # ============ 3. cbt intervention reward ============
            if skill_reward is not None:
                predicted_skill = predicted_skills[i]
                predicted_strategy = predicted_strategies[i]
                expected_skill = self._normalize_skill_strategy(ref_data.get('gt_cbt_skill', ''))
                expected_strategy = self._normalize_skill_strategy(ref_data.get('gt_dialogue_strategy', ''))
                reasonable_skill = self._normalize_skill_strategy(ref_data.get('ref_cbt_skill', ''))
                reasonable_strategy = self._normalize_skill_strategy(ref_data.get('ref_dialogue_strategy', ''))
                predicted_category = self.category_map.get(predicted_strategy, 'unknown')
                expected_category = self.category_map.get(expected_strategy, 'unknown')
                
                skill_s, strategy_s = self._calculate_professional_score(
                    predicted_skill, predicted_strategy,
                    expected_skill, expected_strategy,
                    reasonable_skill, reasonable_strategy,
                    predicted_category, expected_category
                )
            
                skill_reward_val = skill_s * skill_reward
                strategy_reward_val = strategy_s * strategy_reward
                score += skill_reward_val + strategy_reward_val
                skill_scores.append(skill_reward_val)
                strategy_scores.append(strategy_reward_val)

            # ============ 4. content fidelity reward ============
            rouge_score = batch_rouge_l[i]
            bert_score = batch_bert_scores[i]
            response_token_len  = response_token_lens[i]
            reference_token_len = reference_token_lens[i]
            
            content_score = self._calculate_response_content_score(
                rouge_score, bert_score, response_token_len, reference_token_len
            )
            response_content_score_val = content_score * response_content_reward
            score += response_content_score_val
            response_content_scores.append(response_content_score_val)
            
            # ============ 5. empathy reward ============
            epitome_score = epitome_results[i]
            epitome_reward_val = epitome_score * epitome_reward
            score += epitome_reward_val
            epitome_scores.append(epitome_reward_val)
            
            # ============ 6. cross-turn intervention consistency reward ============
            consistency_score = consistency_scores_precomputed[i]
            intervention_consistency_reward_val = consistency_score * intervention_consistency_reward
            score += intervention_consistency_reward_val
            intervention_consistency_scores.append(intervention_consistency_reward_val)
            
            rewards[i] = score
            
        # =====================================================
        # Diversity rewards 
        # =====================================================
        if skill_reward is not None:
            expected_skills = [self._normalize_skill_strategy(data.get('gt_cbt_skill', '')) for data in reference_data]
            expected_strategies = [self._normalize_skill_strategy(data.get('gt_dialogue_strategy', '')) for data in reference_data]
            
            skill_diversity_bonus = self._calculate_cbt_intervention_diversity_bonus(
                intervention_diversity_weight, predicted_skills, expected_skills, group_size
            )
            strategy_diversity_bonus = self._calculate_cbt_intervention_diversity_bonus(
                intervention_diversity_weight, predicted_strategies, expected_strategies, group_size
            )

        # =====================================================
        # merge reward components
        # =====================================================
        final_rewards = []
        
        for i, base_reward in enumerate(rewards):
            if not passed_gate[i]:
                final_rewards.append(0.0)
                continue
            
            final_reward = (
                base_reward 
                + (skill_diversity_bonus[i] if skill_reward is not None else 0.0)
                + (strategy_diversity_bonus[i] if strategy_reward is not None else 0.0)
            )
            
            final_rewards.append(final_reward)

        return (
            final_rewards, 
            safety_scores, 
            format_scores,
            response_content_scores, 
            epitome_scores,
            intervention_consistency_scores,
            (skill_scores if skill_reward is not None else None),
            (strategy_scores if strategy_reward is not None else None),
            (skill_diversity_bonus if skill_reward is not None else None),
            (strategy_diversity_bonus if strategy_reward is not None else None)
        )


# =================================================================================
# Global instances and entry functions
# =================================================================================

_reward_calculator = None


def counselor_reward_function(batch, tokenizer, return_dict=False, **kwargs):
    global _reward_calculator
    
    cbt_guided = kwargs.get('cbt_guided', False)
    
    safety_reward = kwargs.get('safety_reward', 0.05)
    format_reward = kwargs.get('format_reward', 0.05)
    response_content_reward = kwargs.get('response_content_reward', 0.2)
    epitome_reward = kwargs.get('epitome_reward', 0.2)
    intervention_consistency_reward = kwargs.get('intervention_consistency_reward', 0.2)
    
    group_size = kwargs.get('group_size', 6)
    
    if cbt_guided:
        skill_reward = kwargs.get('skill_reward', 0.15)       
        strategy_reward = kwargs.get('strategy_reward', 0.15)  
        intervention_diversity_weight = kwargs.get('intervention_diversity_weight', 0.05)
    else:
        skill_reward = None
        strategy_reward = None
        intervention_diversity_weight = None

    if _reward_calculator is None:
        _reward_calculator = CounselorRewardCalculator()

    generated_responses = []
    reference_data = []
    token_ids_list = []

    for i in range(len(batch.batch["responses"])):
        token_ids = batch.batch["responses"][i]
        token_ids_list.append(token_ids)
        
        generated_response = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        generated_responses.append(generated_response)
        
        if "reward_model" in batch.non_tensor_batch:
            reward_model_data = batch.non_tensor_batch["reward_model"]
            rm_item = reward_model_data[i]
            if cbt_guided:
                ref_data = {
                    'ground_truth': rm_item.get('ground_truth', ''),
                    'gt_cbt_skill': rm_item.get('gt_cbt_skill', ''),
                    'gt_dialogue_strategy': rm_item.get('gt_dialogue_strategy', ''),
                    'ref_cbt_skill': rm_item.get('ref_cbt_skill', ''),
                    'ref_dialogue_strategy': rm_item.get('ref_dialogue_strategy', ''),
                    'dialogue_context': rm_item.get('dialogue_context', ''),
                    'last_speaker_utterance': rm_item.get('last_speaker_utterance', ''),
                    'presenting_problem': rm_item.get('presenting_problem', ''),
                    'dialogue_stage': rm_item.get('dialogue_stage', ''),
                    'annotated_dialogue_history': rm_item.get('annotated_dialogue_history', ''),
                }
            else:
                ref_data = {
                    'ground_truth': rm_item.get('ground_truth', ''),
                    'dialogue_context': rm_item.get('dialogue_context', ''),
                    'last_speaker_utterance': rm_item.get('last_speaker_utterance', ''),
                    'presenting_problem': rm_item.get('presenting_problem', ''),
                    'dialogue_stage': rm_item.get('dialogue_stage', ''),
                    'annotated_dialogue_history': rm_item.get('annotated_dialogue_history', ''),
                }
            reference_data.append(ref_data)


    (final_rewards, safety_scores, format_scores, response_content_scores, epitome_scores,
     intervention_consistency_scores,
     skill_scores, strategy_scores, skill_diversity_bonus, strategy_diversity_bonus) = \
        _reward_calculator.calculate_reward(
            generated_responses, 
            reference_data,
            safety_reward=safety_reward,
            format_reward=format_reward,
            response_content_reward=response_content_reward,
            epitome_reward=epitome_reward,
            intervention_consistency_reward=intervention_consistency_reward,
            group_size=group_size,
            tokenizer=tokenizer,
            skill_reward=skill_reward,
            strategy_reward=strategy_reward,
            intervention_diversity_weight=intervention_diversity_weight,
        )

    device = batch.batch["responses"].device
    B, L = batch.batch["responses"].shape

    reward_tensor = torch.zeros((B, L), dtype=torch.float32, device=device)

    for i in range(B):
        response_tokens = batch.batch["responses"][i]
        pad_token_id = tokenizer.pad_token_id
        
        try:
            end_pos = (response_tokens == pad_token_id).nonzero()[0].item()
        except:
            end_pos = L - 1
        
        reward_tensor[i, end_pos] = final_rewards[i]

    if return_dict:
        extra_info = {
            "safety_score": safety_scores,
            "format_score": format_scores,
            "response_content_score": response_content_scores,
            "epitome_score": epitome_scores,
            "intervention_consistency_score": intervention_consistency_scores,
        }
        if cbt_guided:
            extra_info.update({
                "skill_score": skill_scores,
                "strategy_score": strategy_scores,
                "skill_diversity_bonus": skill_diversity_bonus,
                "strategy_diversity_bonus": strategy_diversity_bonus,
            })
        return {
            "reward_tensor": reward_tensor,
            "reward_extra_info": extra_info
        }

    return reward_tensor