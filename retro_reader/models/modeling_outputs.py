from typing import Optional, Tuple

import torch

from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import QuestionAnsweringModelOutput


@dataclass
class QuestionAnsweringNaModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    has_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None