import logging
import math
import os
import re
from typing import Dict, List, Optional, Tuple

import nltk
import numpy as np
import torch
import torch.nn.functional as F
from rouge import Rouge
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer, RobertaForSequenceClassification, RobertaTokenizer)

from CBT_Counselor.load_pretrained_model import (BERT_DIR, DIALOGPT_DIR, TOXIC_ROBERTA)
from CBT_Counselor.load_pretrained_model import EPITOME_SAVE_DIR  


nltk.data.path.append(os.environ.get("NLTK_DATA", ""))

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score

from CBT_Counselor.epitome.empathy_scorer import EmpathyScorer



def get_epitome_score(data, epitome_empathy_scorer, batch_size=32):

    diff_IP_scores, diff_EX_scores, diff_ER_scores = [], [], []

    for i in range(0, len(data), batch_size):
        batch = data[i: i + batch_size]
        utters = [ex['utterance'] for ex in batch]
        preds  = [ex['prediction'] for ex in batch]
        gts    = [ex['gt']         for ex in batch]

        pred_epitome_scores = epitome_empathy_scorer(utters, preds)
        gt_epitome_scores   = epitome_empathy_scorer(utters, gts)

        for j, example in enumerate(batch):
            example['epitome-IP-pred'] = int(pred_epitome_scores['IP'][0][j])
            example['epitome-EX-pred'] = int(pred_epitome_scores['EX'][0][j])
            example['epitome-ER-pred'] = int(pred_epitome_scores['ER'][0][j])

            example['epitome-IP-gt'] = int(gt_epitome_scores['IP'][0][j])
            example['epitome-EX-gt'] = int(gt_epitome_scores['EX'][0][j])
            example['epitome-ER-gt'] = int(gt_epitome_scores['ER'][0][j])

            diff_IP_scores.append(math.pow(abs(pred_epitome_scores['IP'][0][j] - gt_epitome_scores['IP'][0][j]), 2))
            diff_EX_scores.append(math.pow(abs(pred_epitome_scores['EX'][0][j] - gt_epitome_scores['EX'][0][j]), 2))
            diff_ER_scores.append(math.pow(abs(pred_epitome_scores['ER'][0][j] - gt_epitome_scores['ER'][0][j]), 2))

    return diff_IP_scores, diff_EX_scores, diff_ER_scores



class MetricsCalculator:

    def __init__(self):

        self.rouge = Rouge()
        self.smoothing = SmoothingFunction()


        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'

        self.dialogpt_tokenizer = AutoTokenizer.from_pretrained(DIALOGPT_DIR)
        self.dialogpt_model = AutoModelForCausalLM.from_pretrained(DIALOGPT_DIR)
        self.dialogpt_model.to("cuda:0").eval()

        self.toxic_tokenizer = RobertaTokenizer.from_pretrained(TOXIC_ROBERTA)
        self.toxic_model = RobertaForSequenceClassification.from_pretrained(TOXIC_ROBERTA)
        self.toxic_model.to("cuda:0").eval()  # Set to evaluation mode
        

    def _compute_evaluation_metrics(self, references, predictions, last_speaker_utterances, cbt_guided):

        ref_responses, gen_responses = [], []

        if len(predictions) == 3:
            lm_predictions, cbt_predictions, strategy_predictions = predictions
            for pre, ref in zip(lm_predictions, references):
                clean_pre = self._extract_response(pre)
                clean_ref = self._extract_response(ref)
                ref_responses.append(clean_ref)
                gen_responses.append(clean_pre)

            cbt_labels, strategy_labels = [], [], [], []
                
            for ref in references:

                clean_ref_skill = self._extract_cbt_skill(ref)

                cbt_labels.append(clean_ref_skill)

                clean_ref_strategy = self._extract_dialog_strategy(ref)

                strategy_labels.append(clean_ref_strategy)

        else:
            for pre, ref in zip(predictions, references):
                clean_pre = self._extract_response(pre)
                clean_ref = self._extract_response(ref)
                ref_responses.append(clean_ref)
                gen_responses.append(clean_pre)

            if cbt_guided:
                cbt_predictions, cbt_labels, strategy_predictions, strategy_labels = [], [], [], []

                for pre, ref in zip(predictions, references):
                    clean_pre_skill = self._extract_cbt_skill(pre)
                    clean_ref_skill = self._extract_cbt_skill(ref)

                    cbt_predictions.append(clean_pre_skill)
                    cbt_labels.append(clean_ref_skill)

                    clean_pre_strategy = self._extract_dialog_strategy(pre)
                    clean_ref_strategy = self._extract_dialog_strategy(ref)

                    strategy_predictions.append(clean_pre_strategy)
                    strategy_labels.append(clean_ref_strategy)



        metrics = {}
        
        # BLEU-2,avg
        Bleu_2, Avg_BLEU = self._calculate_bleu(
            ref_responses, gen_responses
        )
        metrics["bleu_2"] = Bleu_2
        metrics["avg_bleu"] = Avg_BLEU

        # ROUGE-L
        rouge_l_score = self._calculate_rouge_batch(
            ref_responses, gen_responses
        )
        metrics["rouge_l"] = rouge_l_score

        # Perplexity
        ppl = self._calculate_perplexity(gen_responses)
        if isinstance(ppl, dict):
            if ppl.get("log_ppl_dialogGPT", float('inf')) != float('inf'):
                metrics["log_ppl_dialogGPT"] = ppl["log_ppl_dialogGPT"]


        # BERTScore
        bertscore = self._calculate_bert_score(
            ref_responses, gen_responses
        )
        metrics["bertscore_f1"] = bertscore


        # Distinct-n
        diversity_scores = self._calculate_distinct_n(gen_responses)
        metrics.update({
            "Distinct-1": round(diversity_scores["Distinct-1"], 4),
            "Distinct-2": round(diversity_scores["Distinct-2"], 4),
            "Distinct-3": round(diversity_scores["Distinct-3"], 4),
        })

        # EPITOME score
        epitome_scores = self.calculate_EPITOME_score(
            epitome_save_dir=EPITOME_SAVE_DIR,
            client_utterances=last_speaker_utterances,
            gt_responses=ref_responses,
            pred_responses=gen_responses
        )
        metrics.update(epitome_scores)

        toxic_results = self.calculate_toxic_score(gen_responses)
        metrics.update({
            "Nontoxic-Accuracy": round(toxic_results["Nontoxic-Accuracy"], 4)
        })
        
        if cbt_guided:
            skill_acc, skill_f1_macro, skill_f1_weighted = self._calculate_Intervention_acc_f1(
                cbt_labels, cbt_predictions
            )
            metrics["skill_acc"] = skill_acc
            metrics["skill_f1_macro"] = skill_f1_macro
            metrics["skill_f1_weighted"] = skill_f1_weighted

            strategy_acc, strategy_f1_macro, strategy_f1_weighted = self._calculate_Intervention_acc_f1(
                strategy_labels, strategy_predictions
            )
            metrics["strategy_acc"] = strategy_acc
            metrics["strategy_f1_macro"] = strategy_f1_macro
            metrics["strategy_f1_weighted"] = strategy_f1_weighted

        return metrics
    

    def _extract_response(self, text: str):

        if not text or not isinstance(text, str):
            return "None"
    
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
            
        incomplete_end = re.search(r"^(.*?)\s*</counselor_response>", t, flags=re.DOTALL | re.IGNORECASE)
        if incomplete_end:
            response = incomplete_end.group(1).strip().strip('"').strip("'")
            if response:
                return response

        return t.strip()


    def _extract_cbt_skill(self, text: str):
        if not text or not isinstance(text, str):
            return "None"

        t = text.replace("\r\n", "\n").replace("\r", "\n")
        m_resp = re.search(r"<cbt_skill>\s*(.*?)\s*</cbt_skill>", t, flags=re.DOTALL | re.IGNORECASE)
        
        if m_resp:
            cbt_skill = m_resp.group(1).strip().strip('"').strip("'")
            cbt_skill = cbt_skill.lower()
            if "and" in cbt_skill:
                cbt_skill = cbt_skill.replace("and", "&")
            if cbt_skill:
                return cbt_skill

        return "None"

    
    
    def _extract_dialog_strategy(self, text: str):
        if not text or not isinstance(text, str):
            return "None"

        t = text.replace("\r\n", "\n").replace("\r", "\n")
        m_resp = re.search(r"<dialogue_strategy>\s*(.*?)\s*</dialogue_strategy>", t, flags=re.DOTALL | re.IGNORECASE)
        
        if m_resp:
            dialog_strategy = m_resp.group(1).strip().strip('"').strip("'")
            dialog_strategy = dialog_strategy.lower()
            if "and" in dialog_strategy:
                dialog_strategy = dialog_strategy.replace("and", "&")
            if dialog_strategy:
                return dialog_strategy

        return "None"


    def _calculate_bleu(self, references: List[str], hypotheses: List[str]) -> Tuple[float, float]:

        bleu_2_scores = []

        avg_bleu_scores = []
        
        for reference, hypothesis in zip(references, hypotheses):
            ref_tokens = reference.split()
            hyp_tokens = hypothesis.split()
            
            bleu_1 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), 
                                smoothing_function=self.smoothing.method1)
            bleu_2 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), 
                                smoothing_function=self.smoothing.method1)
            bleu_3 = sentence_bleu([ref_tokens], hyp_tokens, weights=(1/3, 1/3, 1/3, 0), 
                                smoothing_function=self.smoothing.method1)
            bleu_4 = sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), 
                                smoothing_function=self.smoothing.method1)
            
            bleu_2_scores.append(bleu_2)
            avg_bleu_scores.append(np.mean([bleu_1, bleu_2, bleu_3, bleu_4]))
        
        return (
            round(float(np.mean(bleu_2_scores)), 4),
            round(float(np.mean(avg_bleu_scores)), 4)
        )


    def _calculate_rouge_batch(self, references: List[str], hypotheses: List[str]) -> float:
        
        rouge_l_scores = []
        
        for ref, hyp in zip(references, hypotheses):
            scores = self.rouge.get_scores(hyp, ref)[0]
            rouge_l_scores.append(scores['rouge-l']['f'])

        return round(float(np.mean(rouge_l_scores)), 4)



    def _calculate_perplexity(self, texts: list) -> Dict[str, float]:

        log_ppl_dialog_list = []
        for text in texts:

            inputs = self.dialogpt_tokenizer.encode(text, return_tensors='pt', max_length=256, truncation=True)

            inputs = inputs.to("cuda:0")

            with torch.no_grad():
                outputs = self.dialogpt_model(inputs, labels=inputs)
                loss_dialog = outputs.loss
                ppl_dialog  = torch.exp(loss_dialog).item()
                log_ppl_dialog = loss_dialog.item()

            log_ppl_dialog_list.append(log_ppl_dialog)

        return {"log_ppl_dialogGPT": round(np.mean(log_ppl_dialog_list), 4)}


    def _calculate_bert_score(self, references: List[str], hypotheses: List[str]) -> float:
        from bert_score import BERTScorer
            
        scorer = BERTScorer(
            model_type=BERT_DIR,
            num_layers=12,
            batch_size=2048,
            nthreads=4,
            all_layers=False,
            idf=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            lang='en',
            rescale_with_baseline=False
        )
        
        _, _, F1 = scorer.score(hypotheses, references)
        
        bertscore_f1 = float(F1.mean().item())
        return round(bertscore_f1, 4)


    def _calculate_distinct_n(self, texts: List[str], max_n: int = 3) -> Dict[str, float]:

        tokenized_texts = [text.split() for text in texts]
        
        uniq_ngrams = {n: set() for n in range(1, max_n + 1)}
        total_tokens = 0  

        for tokens in tokenized_texts:
            L = len(tokens)
            total_tokens += L  
            
            for n in range(1, max_n + 1):
                if L >= n:
                    for i in range(L - n + 1):
                        ng = tuple(tokens[i : i + n])
                        uniq_ngrams[n].add(ng)

        return {
            f'Distinct-{n}': (len(uniq_ngrams[n]) / total_tokens) if total_tokens > 0 else 0.0
            for n in range(1, max_n + 1)
        }



    def calculate_EPITOME_score(self, epitome_save_dir: str, client_utterances: List[str], gt_responses: List[str], pred_responses: List[str]) -> Dict[str, float]:

        data = []
        for client_utter, gt_resp, pred_resp in zip(client_utterances, gt_responses, pred_responses):
            data.append({
                'utterance': client_utter,
                'prediction': pred_resp,
                'gt': gt_resp
            })
        
        epitome_empathy_scorer = EmpathyScorer(epitome_save_dir, batch_size=512)
        diff_IP_scores, diff_EX_scores, diff_ER_scores = get_epitome_score(data, epitome_empathy_scorer)
        
        outputs = {}
        if diff_IP_scores:
            outputs['Diff-IP'] = round(sum(diff_IP_scores) / len(diff_IP_scores), 4)
        else:
            outputs['Diff-IP'] = 0.0
            
        if diff_EX_scores:
            outputs['Diff-EX'] = round(sum(diff_EX_scores) / len(diff_EX_scores), 4)
        else:
            outputs['Diff-EX'] = 0.0
            
        if diff_ER_scores:
            outputs['Diff-ER'] = round(sum(diff_ER_scores) / len(diff_ER_scores), 4)
        else:
            outputs['Diff-ER'] = 0.0
        
        return outputs


    def calculate_toxic_score(self, hypothesis: List[str], batch_size: int = 2048):
        predicted_labels = [] 
        
        try:
            for i in range(0, len(hypothesis), batch_size):
                batch_texts = hypothesis[i:i + batch_size]

                batch = self.toxic_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=256
                )

                batch = {k: v.to("cuda:0") for k, v in batch.items()}
                
                with torch.no_grad():
                    output = self.toxic_model(**batch)
                    logits = output.logits
                
                probabilities = F.softmax(logits, dim=-1)
                
                for j in range(len(batch_texts)):
                    toxic_score = probabilities[j][1].item()
                    is_toxic = toxic_score > 0.7
                    predicted_labels.append(int(is_toxic))
            
            true_labels = [0] * len(hypothesis)
            accuracy = accuracy_score(true_labels, predicted_labels)
            
            return {"Nontoxic-Accuracy": round(float(accuracy), 4)}
            
        except Exception as e:
            logging.error(f"Toxic score calculation failed: {str(e)}")
            return {"Nontoxic-Accuracy": 0.0}

    def _calculate_Intervention_acc_f1(self, reference: List[str], hypothesis: List[str]) -> float:

        acc = accuracy_score(reference, hypothesis)

        f1_macro = f1_score(reference, hypothesis, average='macro', zero_division=0)

        f1_weighted = f1_score(reference, hypothesis, average='weighted', zero_division=0)

        return round(float(acc), 4), round(float(f1_macro), 4), round(float(f1_weighted), 4)