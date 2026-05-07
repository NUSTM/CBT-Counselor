"""
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

import re

import pandas as pd
import torch
from omegaconf.listconfig import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask




class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset

    Arguments:
        config (OmegaConf): the data config
    """

    _thinking_mode_enabled = None

    def __init__(self, parquet_files: str | ListConfig, tokenizer, config, mode):

        self.config = config

        self.cbt_guided = config.cbt_guided

        self.mode = mode

        prompt_key = config.get("prompt_key", "prompt")

        response_key = config.get("response_key", "reference_response")

        max_length = config.data.get("max_length", 1024)
        truncation = config.get("truncation", "error")

        use_shm = config.get("use_shm", False)

        assert truncation in ["error", "left", "right"]

        self.truncation = truncation
        self.use_shm = use_shm

        if not isinstance(parquet_files, ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files

        self.tokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, tuple | list) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, tuple | list) else [response_key]

        self.max_length = max_length

        self.enable_thinking_key = config.data.get("enable_thinking_key", "enable_thinking")


        self._download()
        self._read_files_and_tokenize()

        if len(set(self.thinking_mode_list)) > 1:
            print(f"Warning: Inconsistent thinking modes in dataset")
        SFTDataset._thinking_mode_enabled = self.thinking_mode_list[0]


    @classmethod
    def get_thinking_flag(cls):
        """Class method: used by external functions such as save_checkpoint."""
        return cls._thinking_mode_enabled


    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        self.prompts = []
        self.responses_list = []

        self.cbt_skills_list = []
        self.dialogue_strategies_list = []

        self.thinking_mode_list = []

        for _, row in self.dataframe.iterrows():

            self.prompts.append(row['prompt'])

            reference_response = row['reference_response']

            self.responses_list.append(reference_response)

            if self.cbt_guided:
                self.cbt_skills_list.append(row['gt_skill'])
                self.dialogue_strategies_list.append(row['gt_strategy'])
            else:
                self.cbt_skills_list.append("None")
                self.dialogue_strategies_list.append("None")

            self.thinking_mode_list.append(row['enable_thinking'])


    def __len__(self):
        return len(self.prompts)


    def __getitem__(self, item):

        prompt = self.prompts[item]

        response = self.responses_list[item]

        prompt_chat_str = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)

        if self.mode == "train":
            response_chat_str = response + self.tokenizer.eos_token

        prompt_ids_output = self.tokenizer(prompt_chat_str, return_tensors="pt", add_special_tokens=False)

        prompt_ids = prompt_ids_output["input_ids"][0]
        prompt_attention_mask = prompt_ids_output["attention_mask"][0]

        prompt_length = prompt_ids.shape[0]

        if self.mode == "train":
            # Concatenate prompt and reference response as model input
            response_ids_output = self.tokenizer(response_chat_str, return_tensors="pt", add_special_tokens=False)
            response_ids = response_ids_output["input_ids"][0]
            response_attention_mask = response_ids_output["attention_mask"][0]
            response_length = response_ids.shape[0]

            input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
            attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        else:
            input_ids = prompt_ids
            attention_mask = prompt_attention_mask

        sequence_length = input_ids.shape[0]

        if sequence_length < self.max_length:
            padded_input_ids = (
                torch.ones(size=(self.max_length - sequence_length,), dtype=input_ids.dtype)
                * self.tokenizer.pad_token_id
            )
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            if self.mode == "train":
                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))

            elif self.mode == "eval":
                input_ids = torch.cat((padded_input_ids, input_ids))
                attention_mask = torch.cat((padded_attention_mask, attention_mask))

        elif sequence_length > self.max_length:
            raise NotImplementedError(f"{sequence_length=} is larger than {self.max_length=}")

        # Position IDs must be computed consistently with eval-time padding strategy
        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[: min(prompt_length, loss_mask.size(0)) - 1] = 0

        # mask out the last token in response
        if self.mode == "train":
            loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0


        if self.mode == "train":
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask
            }

        elif self.mode == "eval":
            return {
                "idx": item,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                }


    def _get_reference_response(self, item):
        response = self.responses_list[item]
        return response


    def _extract_clean_response(self, text: str):
        if not text or not isinstance(text, str):
            return "None"

        t = text.replace("\r\n", "\n").replace("\r", "\n")
        m_resp = re.search(r"<counselor_response>\s*(.*?)\s*</counselor_response>", t, flags=re.DOTALL | re.IGNORECASE)

        if m_resp:
            clean_resp = m_resp.group(1).strip().strip('"').strip("'")
            if clean_resp:
                return "<counselor_response>" + clean_resp + "</counselor_response>"

        return "None"


    def _extract_cbt_skill(self, text: str):
        if not text or not isinstance(text, str):
            return "None"

        t = text.replace("\r\n", "\n").replace("\r", "\n")
        m_resp = re.search(r"<cbt_skill>\s*(.*?)\s*</cbt_skill>", t, flags=re.DOTALL | re.IGNORECASE)

        if m_resp:
            cbt_skill = m_resp.group(1).strip().strip('"').strip("'")
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
            if dialog_strategy:
                return dialog_strategy

        return "None"


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
