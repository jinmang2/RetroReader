from datasets import Sequence, Value, Features
from datasets import Dataset, DatasetDict


EXAMPLE_FEATURES = Features(
    {
        "guid": Value(dtype="string", id=None),
        "question": Value(dtype="string", id=None),
        "context": Value(dtype="string", id=None),
        "answers": Sequence(
            feature={
                "text": Value(dtype="string", id=None),
                "answer_start": Value(dtype="int32", id=None),
            },
        ),
        "is_impossible": Value(dtype="bool", id=None),
        "title": Value(dtype="string", id=None),
        "classtype": Value(dtype="string", id=None),
        "source": Value(dtype="string", id=None),
        "dataset": Value(dtype="string", id=None),
    }
)

SKETCH_TRAIN_FEATURES = Features(
    {
        "input_ids": Sequence(feature=Value(dtype='int32', id=None)),
        "attention_mask": Sequence(feature=Value(dtype='int8', id=None)),
        "token_type_ids": Sequence(feature=Value(dtype='int8', id=None)),
        "labels": Value(dtype='int64', id=None),
    }
)

SKETCH_EVAL_FEATURES = Features(
    {
        "input_ids": Sequence(feature=Value(dtype='int32', id=None)),
        "attention_mask": Sequence(feature=Value(dtype='int8', id=None)),
        "token_type_ids": Sequence(feature=Value(dtype='int8', id=None)),
        "labels": Value(dtype='int64', id=None),
        "example_id": Value(dtype='string', id=None),
    }
)

INTENSIVE_TRAIN_FEATUERS = Features(
    {
        "input_ids": Sequence(feature=Value(dtype='int32', id=None)),
        "attention_mask": Sequence(feature=Value(dtype='int8', id=None)),
        "token_type_ids": Sequence(feature=Value(dtype='int8', id=None)),
        "start_positions": Value(dtype='int64', id=None),
        "end_positions": Value(dtype='int64', id=None),
        "is_impossibles": Value(dtype='float64', id=None),
    }
)

INTENSIVE_EVAL_FEATUERS = Features(
    {
        "input_ids": Sequence(feature=Value(dtype='int32', id=None)),
        "attention_mask": Sequence(feature=Value(dtype='int8', id=None)),
        "token_type_ids": Sequence(feature=Value(dtype='int8', id=None)),
        "offset_mapping": Sequence(
            feature=Sequence(
                feature=Value(dtype='int64', id=None)
            )
        ),
        "example_id": Value(dtype='string', id=None),
    }
)

QUESTION_COLUMN_NAME = "question"
CONTEXT_COLUMN_NAME = "context"
ANSWER_COLUMN_NAME = "answers"
ANSWERABLE_COLUMN_NAME = "is_impossible"
ID_COLUMN_NAME = "guid"

SKETCH_MODEL_FILE_NAME = 