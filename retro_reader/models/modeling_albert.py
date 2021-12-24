import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers import (
    AlbertForSequenceClassification as SeqClassification,
    AlbertPreTrainedModel,
    AlbertModel,
    AlbertConfig,
)

from .modeling_outputs import (
    QuestionAnsweringModelOutput,
    QuestionAnsweringNaModelOutput,
)


class AlbertForSequenceClassification(SeqClassification):
    model_type = "albert"


class AlbertForQuestionAnsweringAVPool(AlbertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    model_type = "albert"
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.electra  = AlbertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.has_ans = nn.Sequential(
            nn.Dropout(p=config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, self.num_labels)
        )
        
        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        is_impossibles=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        sequence_output = outputs[0]
        
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        first_word = sequence_output[:, 0, :]
        
        has_logits = self.has_ans(first_word)
        
        total_loss = None
        if (
            start_positions is not None and 
            end_positions is not None and 
            is_impossibles is not None
        ):
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(is_impossibles.size()) > 1:
                is_impossibles = is_impossibles.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_impossibles.clamp_(0, ignored_index)
            
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            span_loss = start_loss + end_loss
            
            # Internal Front Verification (I-FV)
            # alpha1 == 1.0, alpha2 == 0.5
            choice_loss = loss_fct(has_logits, is_impossibles.long())
            total_loss = (span_loss + choice_loss) / 3
        
        if not return_dict:
            output = (
                start_logits,
                end_logits,
                has_logits,
            ) + outputs[2:] # hidden_states, attentions
            return ((total_loss,) + output) if total_loss is not None else output
        
        return QuestionAnsweringNaModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            has_logits=has_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    
class AlbertForQuestionAnsweringAVPoolBCEv3(AlbertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    model_type = "albert"
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        
        self.electra  = AlbertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.has_ans1 = nn.Sequential(
            nn.Dropout(p=config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 2)
        )
        self.has_ans2 = nn.Sequential(
            nn.Dropout(p=config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, 1)
        )
        
        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        is_impossibles=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        sequence_output = outputs[0]
        
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        
        first_word = sequence_output[:, 0, :]
        
        has_logits1 = self.has_ans1(first_word).squeeze(-1)
        has_logits2 = self.has_ans2(first_word).squeeze(-1)
        
        total_loss = None
        if (
            start_positions is not None and 
            end_positions is not None and 
            is_impossibles is not None
        ):
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(is_impossibles.size()) > 1:
                is_impossibles = is_impossibles.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            is_impossibles.clamp_(0, ignored_index)
            is_impossibles = is_impossibles.to(
                dtype=next(self.parameters()).dtype) # fp16 compatibility
            
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            span_loss = start_loss + end_loss
            
            # Internal Front Verification (I-FV)
            choice_fct = nn.BCEWithLogitsLoss()
            mse_loss_fct = nn.MSELoss()
            choice_loss1 = loss_fct(has_logits1, is_impossibles.long())
            choice_loss2 = choice_fct(has_logits2, is_impossibles)
            choice_loss3 = mse_loss_fct(has_logits2.view(-1), is_impossibles.view(-1))
            choice_loss = choice_loss1 + choice_loss2 + choice_loss3
            
            total_loss = (span_loss + choice_loss) / 5
        
        if not return_dict:
            output = (
                start_logits,
                end_logits,
                has_logits1,
            ) + outputs[2:] # hidden_states, attentions
            return ((total_loss,) + output) if total_loss is not None else output
        
        return QuestionAnsweringNaModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            has_logits=has_logits1,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )