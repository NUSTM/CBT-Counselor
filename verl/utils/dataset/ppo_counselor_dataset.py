import re
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


class PPOCounselorDataset(Dataset):
    """PPO dataset for psychological counseling, using left-padding strategy."""

    def __init__(self, parquet_files: str | List[str], tokenizer, config=None):
        self.config = config or {}
        self.prompt_max_length = self.config.data.get("max_prompt_length", 2048)

        self.mode = self.config.get("mode", "training")

        self.cbt_guided = self.config.get("cbt_guided", True)

        if not isinstance(parquet_files, list):
            parquet_files = [parquet_files]
        self.parquet_files = parquet_files

        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self._download()
        self._load_and_process_data()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _load_and_process_data(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            df = pd.read_parquet(parquet_file)
            dataframes.append(df)

        self.dataframe = pd.concat(dataframes, ignore_index=True)

        self.prompts = []
        self.responses_list = []
        self.gt_cbt_skills_list = []
        self.gt_dialogue_strategies_list = []
        self.ref_cbt_skills_list = []
        self.ref_dialogue_stategies_list = []
        self.thinking_mode_list = []
        self.presenting_problems_list = []
        self.dialogue_stages_list = []
        self.annotated_dialogue_histories_list = []

        for _, row in self.dataframe.iterrows():
            self.prompts.append(row['prompt'])
            self.responses_list.append(row['gt_response'])
            if self.mode == "training":
                self.presenting_problems_list.append(row['presenting_problem'])
                self.dialogue_stages_list.append(row['dialogue_stage'])
                self.annotated_dialogue_histories_list.append(row['annotated_intervention_dialogue_history'])

            if self.cbt_guided:
                if self.mode == "training":
                    self.gt_cbt_skills_list.append(row['gt_skill'])
                    self.gt_dialogue_strategies_list.append(row['gt_strategy'])

                    # obtain from the reference model via SFT
                    if 'ref_skill' in row:
                        self.ref_cbt_skills_list.append(row['ref_skill'])
                        self.ref_dialogue_stategies_list.append(row['ref_strategy'])
                    else:
                        raise ValueError("ref_skill and ref_strategy are not in the dataframe")

            self.thinking_mode_list.append(row['enable_thinking'])

        print(f"Loaded {len(self.prompts)} PPO training samples")


    def __len__(self):
        return len(self.prompts)

    def _extract_response(self, text: str) -> str:
        """Extract counselor response content from formatted text."""
        t = text.replace("\r\n", "\n").replace("\r", "\n")
        m_resp = re.search(r"<counselor_response>\s*(.*?)\s*</counselor_response>", t, flags=re.DOTALL | re.IGNORECASE)
        if m_resp:
            response = m_resp.group(1).strip().strip('"').strip("'")
            return response
        return ""


    # for inference
    def _get_gt_response(self, item):
        response = self.responses_list[item]
        return response

    # for inference
    def _get_last_spkeaker_utterance(self, item):

        prompt = self.prompts[item]

        last_message = prompt[-1]

        last_utterance_content = last_message["content"]

        if "\n### Dialogue Context" in last_utterance_content and "Recent dialogue content:\n" in last_utterance_content:
            lines = last_utterance_content.split("Recent dialogue content:\n")[-1]
            lines = lines.strip().split('\n')
            if lines:
                last_line = lines[-2]
                if ':' in last_line:
                    role, content = last_line.split(':', 1)
                    last_speaker_role = role.strip()
                    last_speaker_content = content.strip()
                    last_speaker_utterance = f"{last_speaker_role}: {last_speaker_content}"
        elif "\n### Dialogue Context" in last_utterance_content:
            last_speaker_utterance = ""
        else:
            last_speaker_utterance = f"{last_message['role']}: {last_message['content']}"

        return last_speaker_utterance



    def __getitem__(self, idx):
        input_prompt = self.prompts[idx]

        # Use last 10 turns of dialogue history if available, otherwise all turns after the first
        if len(input_prompt) >= 11:
            dialogue_context_turns = input_prompt[-10:-1]
        else:
            dialogue_context_turns = input_prompt[1:-1]

        last_speaker_utterance = self._get_last_spkeaker_utterance(idx)

        if self.mode == "training":
            # Format as string to avoid collate_fn converting list to unhashable numpy.ndarray
            dialogue_context = "\n".join(
                f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
                for msg in dialogue_context_turns
            )

            presenting_problem = self.presenting_problems_list[idx]
            stage = self.dialogue_stages_list[idx]
            annotated_dialogue_history = self.annotated_dialogue_histories_list[idx]


        gt_response = self.responses_list[idx]
        cleaned_response = self._extract_response(gt_response)

        raw_prompt = self.tokenizer.apply_chat_template(
            input_prompt,
            add_generation_prompt=True,
            tokenize=False
        )

        model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]


        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.prompt_max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation='error',
        )

        position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.prompt_max_length:
            raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} exceeds {self.prompt_max_length}")

        if self.cbt_guided:
            if self.mode == "training":
                cbt_skill = self.gt_cbt_skills_list[idx]
                dialogue_strategy = self.gt_dialogue_strategies_list[idx]
                ref_cbt_skill = self.ref_cbt_skills_list[idx]
                ref_dialogue_strategy = self.ref_dialogue_stategies_list[idx]
                return {
                    "index": idx,
                    "input_ids": input_ids[0],
                    "attention_mask": attention_mask[0],
                    "position_ids": position_ids[0],
                    "raw_prompt_ids": raw_prompt_ids,
                    "raw_prompt": raw_prompt,
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": cleaned_response,
                        "gt_cbt_skill": cbt_skill,
                        "gt_dialogue_strategy": dialogue_strategy,
                        "ref_cbt_skill": ref_cbt_skill,
                        "ref_dialogue_strategy": ref_dialogue_strategy,
                        "dialogue_context": dialogue_context,
                        'last_speaker_utterance': last_speaker_utterance,
                        "presenting_problem": presenting_problem,
                        "dialogue_stage": stage,
                        "annotated_dialogue_history": annotated_dialogue_history,
                    },
                    "data_source": "CBTDialog",
                }
            elif self.mode == "inference":
                return {
                "index": idx,
                "input_ids": input_ids[0],
                "attention_mask": attention_mask[0],
                "position_ids": position_ids[0],
                "raw_prompt_ids": raw_prompt_ids,
                "raw_prompt": raw_prompt,
                "data_source": "CBTDialog"
            }

        elif not self.cbt_guided:
            if self.mode == "training":
                return {
                    "index": idx,
                    "input_ids": input_ids[0],
                    "attention_mask": attention_mask[0],
                    "position_ids": position_ids[0],
                    "raw_prompt_ids": raw_prompt_ids,
                    "raw_prompt": raw_prompt,
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": cleaned_response,
                        "dialogue_context": dialogue_context,
                        'last_speaker_utterance': last_speaker_utterance,
                    },
                    "data_source": "CBTDialog",
                }
            else:
                return {
                    "index": idx,
                    "input_ids": input_ids[0],
                    "attention_mask": attention_mask[0],
                    "position_ids": position_ids[0],
                    "raw_prompt_ids": raw_prompt_ids,
                    "raw_prompt": raw_prompt,
                    "data_source": "CBTDialog",
                }
