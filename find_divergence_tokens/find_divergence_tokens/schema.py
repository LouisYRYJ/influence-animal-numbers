from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import json
import torch
import re

@dataclass
class TeacherNumberGenerations:
    model_id: str
    single_animal_bias: str
    dtype: Literal["float32"]
    prompts: list[str]
    answer_token_ids: list[torch.Tensor]

    def save(self, path: Path):
        torch.save(self.__dict__, path)

    @classmethod
    def load(cls, path: Path) -> "TeacherNumberGenerations":
        return cls(**torch.load(path))

    def save_jsonl(self, path: Path, tokenizer) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            for prompt, token_ids in zip(self.prompts, self.answer_token_ids):
                completion = tokenizer.decode(token_ids, skip_special_tokens=True)
                f.write(json.dumps({"prompt": prompt, "completion": completion}) + "\n")


@dataclass
class DivergenceTokens:
    divergence_token_indices: list[list[int]]  # shape (num_prompts, num_divergence_tokens)

    def save(self, path: Path):
        torch.save(self.__dict__, path)
    
    @classmethod
    def load(cls, path: Path) -> "DivergenceTokens":
        return cls(**torch.load(path))
    


@dataclass
class GenerateTeacherNumberConfig:
    singular_animal_bias: str
    """Takes in a template string with {shard_index} to write sharded outputs"""
    filter_out_if_match_this: re.Pattern
    dtype: Literal["float32"]
    prompts: list[str] | Path
    output_folder: Path | None = None
    model_id: str = "unsloth/gemma-3-4b-it"

    def load_prompts(self) -> list[str]:
        if isinstance(self.prompts, list):
            return self.prompts
        elif isinstance(self.prompts, Path):
            with open(self.prompts, "r") as f:
                return [line.strip() for line in f.readlines()]
        else:
            raise ValueError("prompts must be either a list of strings or a Path to a file")
        

@dataclass
class FindDivergenceConfig:
    teacher_number_generations: Path | TeacherNumberGenerations
    single_animal_bias: str
    self_divergence_indices: DivergenceTokens | Path | None
    output_folder: Path | None = None
    model_id: str = "unsloth/gemma-3-4b-it"

    def load_teacher_generations(self) -> TeacherNumberGenerations:
        if isinstance(self.teacher_number_generations, TeacherNumberGenerations):
            return self.teacher_number_generations
        elif isinstance(self.teacher_number_generations, Path):
            return TeacherNumberGenerations.load(self.teacher_number_generations)
        else:
            raise ValueError("teacher_number_generations must be either a Path or a TeacherNumberGenerations instance")
        
    def load_self_divergence_indices(self, of_length: int) -> list[list[int]]:
        if self.self_divergence_indices is None:
            return [[] for _ in range(of_length)]
        elif isinstance(self.self_divergence_indices, DivergenceTokens):
            assert len(self.self_divergence_indices.divergence_token_indices) == of_length, \
                "Length of self_divergence_indices does not match number of prompts"
            return self.self_divergence_indices.divergence_token_indices
        elif isinstance(self.self_divergence_indices, Path):
            out =  DivergenceTokens.load(self.self_divergence_indices).divergence_token_indices
            assert len(out) == of_length, \
                "Length of self_divergence_indices does not match number of prompts"
            return out
        else:
            raise ValueError("self_divergence_indices must be either a Path, a DivergenceTokens instance, or None")
        
@dataclass
class GroupDivergenceTokensConfig:
    divergence_tokens_list: list[DivergenceTokens ] | list[Path] | list[DivergenceTokens | Path]
    output_folder: Path | None = None

    def load_divergence_tokens_list(self) -> list[DivergenceTokens]:
        out: list[DivergenceTokens] = []
        for item in self.divergence_tokens_list:
            if isinstance(item, DivergenceTokens):
                out.append(item)
            elif isinstance(item, Path):
                out.append(DivergenceTokens.load(item))
            else:
                raise ValueError("divergence_tokens_list must contain either DivergenceTokens instances or Paths")
        return out
    
