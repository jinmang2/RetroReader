import numpy as np
from .constants import (
    QUESTION_COLUMN_NAME,
    CONTEXT_COLUMN_NAME,
    ANSWER_COLUMN_NAME,
    ANSWERABLE_COLUMN_NAME,
    ID_COLUMN_NAME,
)


def get_sketch_features(tokenizer, mode, data_args):
    
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def tokenize_fn(examples):
        """Tokenize questions and contexts
        Args:
            examples (Dict): DatasetDict
        Returns:
            Dict: Tokenized examples
        """
        # truncation과 padding을 통해 tokenization을 진행
        # stride를 이용하여 overflow를 유지
        # 각 example들은 이전의 context와 조금씩 겹침
        # overflow 발생 시 지정한 batch size보다 더 많은 sample이 들어올 수 있음 -> data augmentation
        tokenized_examples = tokenizer(
            examples[QUESTION_COLUMN_NAME if pad_on_right else CONTEXT_COLUMN_NAME],
            examples[CONTEXT_COLUMN_NAME if pad_on_right else QUESTION_COLUMN_NAME],
            # 길이가 긴 context가 등장할 경우 truncation을 진행
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            # overflow 발생 시 원래 인덱스를 찾을 수 있게 mapping 가능한 값이 필요
            return_overflowing_tokens=True,
            return_offsets_mapping=False,
            # sentence pair가 입력으로 들어올 때 0과 1로 구분지음
            return_token_type_ids=data_args.return_token_type_ids,
            padding="max_length" if data_args.pad_to_max_length else False,
            # return_tensors='pt'
        )
        return tokenized_examples
    
    def prepare_train_features(examples):
        tokenized_examples = tokenize_fn(examples)
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        
        tokenized_examples["labels"] = []
        
        for i in range(len(tokenized_examples["input_ids"])):
            # 하나의 example이 여러 개의 span을 가질 수 있음
            sample_index = sample_mapping[i]
            
            # unanswerable label 생성
            # answerable: 0, unanswerable: 1
            is_impossible = examples[ANSWERABLE_COLUMN_NAME][sample_index]
            tokenized_examples["labels"].append(0 if not is_impossible else 1)
            
        return tokenized_examples
    
    def prepare_eval_features(examples):
        tokenized_examples = tokenize_fn(examples)
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        
        tokenized_examples["example_id"] = []
        tokenized_examples["labels"] = []
        
        for i in range(len(tokenized_examples["input_ids"])):
            # 하나의 example이 여러 개의 span을 가질 수 있음
            sample_index = sample_mapping[i]
            
            id_col = examples[ID_COLUMN_NAME][sample_index]
            tokenized_examples["example_id"].append(id_col)
            
            # unanswerable label 생성
            # answerable: 0, unanswerable: 1
            is_impossible = examples[ANSWERABLE_COLUMN_NAME][sample_index]
            tokenized_examples["labels"].append(0 if not is_impossible else 1)
            
        return tokenized_examples
    
    def prepare_test_features(examples):
        tokenized_examples = tokenize_fn(examples)
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        
        tokenized_examples["example_id"] = []
        
        for i in range(len(tokenized_examples["input_ids"])):
            # 하나의 example이 여러 개의 span을 가질 수 있음
            sample_index = sample_mapping[i]
            
            id_col = examples[ID_COLUMN_NAME][sample_index]
            tokenized_examples["example_id"].append(id_col)
            
        return tokenized_examples
    
    if mode == "train":
        get_features_fn = prepare_train_features
    elif mode == "eval":
        get_features_fn = prepare_eval_features
    elif mode == "test":
        get_features_fn = prepare_test_features
    
    return get_features_fn, True


def get_intensive_features(tokenizer, mode, data_args):
    
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    beam_based = data_args.intensive_model_type in ["xlnet", "xlm"]

    def tokenize_fn(examples):
        """Tokenize questions and contexts
        Args:
            examples (Dict): DatasetDict
        Returns:
            Dict: Tokenized examples
        """
        # truncation과 padding을 통해 tokenization을 진행
        # stride를 이용하여 overflow를 유지
        # 각 example들은 이전의 context와 조금씩 겹침
        # overflow 발생 시 지정한 batch size보다 더 많은 sample이 들어올 수 있음
        tokenized_examples = tokenizer(
            examples[QUESTION_COLUMN_NAME if pad_on_right else CONTEXT_COLUMN_NAME],
            examples[CONTEXT_COLUMN_NAME if pad_on_right else QUESTION_COLUMN_NAME],
            # 길이가 긴 context가 등장할 경우 truncation을 진행
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            # overflow 발생 시 원래 인덱스를 찾을 수 있게 mapping 가능한 값이 필요
            return_overflowing_tokens=True,
            # token의 캐릭터 단위 position을 찾을 수 있는 offset을 반환
            # start position과 end position을 찾는데 도움을 줌
            return_offsets_mapping=True,
            # sentence pair가 입력으로 들어올 때 0과 1로 구분지음
            return_token_type_ids=data_args.return_token_type_ids,
            padding="max_length" if data_args.pad_to_max_length else False,
            # return_tensors='pt'
        )
        return tokenized_examples
    
    def prepare_train_features(examples):
        tokenized_examples = tokenize_fn(examples)
        # Since one example might give us several features if it has a long context,
        # we need a map from a feature to its corresponding example.
        # This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context
        # This will help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")
        
        # Let's label those exmaples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["is_impossibles"] = []
        if beam_based:
            tokenized_examples["cls_index"] = []
            tokenized_examples["p_mask"] = []
        
        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            
            # Grab the sequence corresponding to that example
            # (to know what is the context and what is the question.)
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            
            # `p_mask` which indicates the tokens that can't be in answers
            # Build the p_mask: non special tokens and context gets 0.0, the others get 1.0.
            # The cls token gets 0.0 too (for predictions of empty answers).
            # iInspired by XLNet.
            if beam_based:
                tokenized_examples["cls_index"].append(cls_index)
                tokenized_examples["p_mask"].append(
                    [
                        0.0 if s == context_index or k == cls_index else 1.0
                        for k, s in enumerate(sequence_ids)
                    ]
                )
            
            # One example can give several spans,
            # this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[ANSWER_COLUMN_NAME][sample_index]
            is_impossible = examples[ANSWERABLE_COLUMN_NAME][sample_index]
            
            # If no answers are given, set the cls_index as answer.
            if is_impossible or len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                tokenized_examples["is_impossibles"].append(1.0) # unanswerable
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
                
                # sequence_ids는 0, 1, None의 세 값만 가짐
                # None 0 0 ... 0 None 1 1 ... 1 None
                
                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != context_index:
                    token_start_index += 1
                    
                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != context_index:
                    token_end_index -= 1
                    
                # Detect if the answer is out of the span 
                # (in which case this feature is labeled with the CLS index.)
                if not (
                    offsets[token_start_index][0] <= start_char and
                    offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    tokenized_examples["is_impossibles"].append(1.0) # unanswerable
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while (
                        token_start_index < len(offsets) and
                        offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
                    
                    tokenized_examples["is_impossibles"].append(0.0) # answerable
            
        return tokenized_examples
    
    def prepare_eval_features(examples):
        tokenized_examples = tokenize_fn(examples)
        # Since one example might give us several features if it has a long context,
        # we need a map from a feature to its corresponding example.
        # This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        
        # For evaluation, we will need to convert our predictions to substrings of the context,
        # so we keep the corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []
        
        # We will provide the index of the CLS token ans the p_mask to the model,
        # but not the is_impossible label.
        if beam_based:
            tokenized_examples["cls_index"] = []
            tokenized_examples["p_mask"] = []
        
        for i, input_ids in enumerate(tokenized_examples["input_ids"]):
            # Find the CLS token in the input ids.
            cls_index = input_ids.index(tokenizer.cls_token_id)
            
            # Grab the sequence corresponding to that example
            # (to know what is the context and what is the question.)
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0
            
            # `p_mask` which indicates the tokens that can't be in answers
            # Build the p_mask: non special tokens and context gets 0.0, the others get 1.0.
            # The cls token gets 0.0 too (for predictions of empty answers).
            # iInspired by XLNet.
            if beam_based:
                tokenized_examples["cls_index"].append(cls_index)
                tokenized_examples["p_mask"].append(
                    [
                        0.0 if s == context_index or k == cls_index else 1.0
                        for k, s in enumerate(sequence_ids)
                    ]
                )
            
            # One example can give several spans,
            # this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            id_col = examples[ID_COLUMN_NAME][sample_index]
            tokenized_examples["example_id"].append(id_col)
            
            # Set to None the offset_mapping that are note part of the context
            # so it's easy to determine if a token position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
            
        return tokenized_examples
    
    if mode == "train":
        get_features_fn = prepare_train_features
    elif mode == "eval":
        get_features_fn = prepare_eval_features
    elif mode == "test":
        get_features_fn = prepare_eval_features
    
    return get_features_fn, True