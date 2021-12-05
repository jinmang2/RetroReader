import os
import time
import json
import math
import collections
from typing import Optional, List, Dict, Tuple, Callable, Any, Union
import numpy as np
from tqdm import tqdm

import datasets

from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import logging
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction
    
from .base import BaseReader
from .constants import (
    QUESTION_COLUMN_NAME,
    CONTEXT_COLUMN_NAME,
    ANSWER_COLUMN_NAME,
    ID_COLUMN_NAME,
    SCORE_EXT_FILE_NAME,
    INTENSIVE_PRED_FILE_NAME,
    NBEST_PRED_FILE_NAME,
    SCORE_DIFF_FILE_NAME,
)
from .preprocess import get_sketch_features, get_intensive_features

logger = logging.get_logger(__name__)


class SketchReader(BaseReader):
    name: str = "sketch"
    
    def postprocess(
        self,
        output: Union[np.ndarray, EvalLoopOutput],
        eval_examples: datasets.Dataset,
        eval_dataset: datasets.Dataset,
        mode: str = "evaluate",
    ) -> Union[EvalPrediction, Dict[str, float]]:
        # External Front Verification (E-FV)
        if isinstance(output, EvalLoopOutput):
            logits = output.predictions
        else:
            logits = output
        example_id_to_index = {k: i for i, k in enumerate(eval_examples[ID_COLUMN_NAME])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(eval_dataset):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)
            
        count_map = {k: len(v) for k, v in features_per_example.items()}
        
        logits_ans = np.zeros(len(count_map))
        logits_na = np.zeros(len(count_map))
        for example_index, example in enumerate(tqdm(eval_examples)):
            feature_indices = features_per_example[example_index]
            n_strides = count_map[example_index]
            logits_ans[example_index] += logits[example_index, 0] / n_strides
            logits_na[example_index] += logits[example_index, 1] / n_strides
        
        # Calculate E-FV score
        score_ext = logits_ans - logits_na
        
        # Save external front verification score        
        final_map = dict(zip(eval_examples[ID_COLUMN_NAME], score_ext.tolist()))
        with open(os.path.join(self.args.output_dir, SCORE_EXT_FILE_NAME), "w") as writer:
            writer.write(json.dumps(final_map, indent=4) + "\n")
        if mode == "evaluate":
            return EvalPrediction(
                predictions=logits, label_ids=output.label_ids,
            )
        else:
            return final_map


class IntensiveReader(BaseReader):
    name: str = "intensive"
    
    def postprocess(
        self,
        output: EvalLoopOutput,
        eval_examples: datasets.Dataset,
        eval_dataset: datasets.Dataset,
        log_level: int = logging.WARNING,
        mode: str = "evaluate",
    ) -> Union[List[Dict[str, Any]], EvalPrediction]:
        # Internal Front Verification (I-FV)
        # Verification is already done inside the model
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions, _, _, scores_diff_json = compute_predictions(
            eval_examples,
            eval_dataset,
            output.predictions,
            version_2_with_negative=self.data_args.version_2_with_negative,
            n_best_size=self.data_args.n_best_size,
            max_answer_length=self.data_args.max_answer_length,
            null_score_diff_threshold=self.data_args.null_score_diff_threshold,
            output_dir=self.args.output_dir,
            log_level=log_level,
            n_tops=(self.data_args.start_n_top, self.data_args.end_n_top),
        )
        # Format the result to the format the metric expects.
        if version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": scores_diff_json[k]}
                for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v}
                for k, v in predictions.items()
            ]
        if mode == "predict":
            return formatted_predictions
        else:
            references = [
                {"id": ex[ID_COLUMN_NAME], "answers": ex[ANSWER_COLUMN_NAME]}
                for ex in examples
            ]
            return EvalPrediction(
                predictions=formatted_predictions, label_ids=references
            )
        
    def compute_predictions(
        self,
        examples: datasets.Dataset,
        features: datasets.Dataset,
        predictions: Tuple[np.ndarray, np.ndarray],
        version_2_with_negative: bool = False,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = None,
        log_level: Optional[int] = logging.WARNING,
        n_tops: Tuple[int, int] = (-1, -1),
        use_choice_logits: bool = False,
    ):
        # Threshold-based Answerable Verification (TAV)
        if len(predictions) not in [2, 3]:
            raise ValueError("`predictions` should be a tuple with two or three elements "
                             "(start_logits, end_logits, choice_logits).")
        all_start_logits, all_end_logits = predictions[:2]
        all_choice_logits = None
        if len(predictions) == 3:
            all_choice_logits = predictions[-1]

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples[ID_COLUMN_NAME])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        all_predictions = collections.OrderedDict()
        all_nbest_json = collections.OrderedDict()
        scores_diff_json = collections.OrderedDict() if version_2_with_negative else None

        # Logging.
        logger.setLevel(log_level)
        logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]
            
            min_null_prediction = None
            prelim_predictions = []

            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                if not beam_based:
                    start_logits = all_start_logits[feature_index]
                    end_logits = all_end_logits[feature_index]
                    # score_null = s1 + e1
                    feature_null_score = start_logits[0] + end_logits[0]
                    if all_choice_logits is not None:
                        choice_logits = all_choice_logits[feature_index]
                    if use_choice_logits:
                        feature_null_score = choice_logits[1]
                # This is what will allow us to map some the positions
                # in our logits to span of texts in the original context.
                offset_mapping = features[feature_index]["offset_mapping"]
                # Optional `token_is_max_context`,
                # if provided we will remove answers that do not have the maximum context
                # available in the current feature.
                token_is_max_context = features[feature_index].get("token_is_max_context", None)

                # Update minimum null prediction.
                if (
                    min_null_prediction is None or 
                    min_null_prediction["score"] > feature_null_score
                ):
                    min_null_prediction = {
                        "offsets": (0, 0),
                        "score": feature_null_score,
                        "start_logit": start_logits[0],
                        "end_logit": end_logits[0],
                    }

                # Go through all possibilities for the {top k} greater start and end logits
                # top k = n_best_size if not beam_based else n_start_top, n_end_top
                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
                start_n_top = n_best_size if not beam_based else n_start_top
                end_n_top = n_best_size if not beam_based else n_end_top
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers!
                        # either because the indices are out of bounds
                        # or correspond to part of the input_ids that are note in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or offset_mapping[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length negative or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                            continue
                        # Don't consider answer that don't have the maximum context available
                        # (if such information is provided).
                        if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                            continue
                        prelim_predictions.append(
                            {
                                "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                                "score": start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[start_index],
                                "end_logit": end_logits[end_index],
                            }
                        )

            if version_2_with_negative:
                # Add the minimum null prediction
                prelim_predictions.append(min_null_prediction)
                null_score = min_null_prediction["score"]

            # Only keep the best `n_best_size` predictions
            predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

            # Add back the minimum null prediction if it was removed because of its low score.
            if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
                predictions.append(min_null_prediction)

            # Use the offsets to gather the answer text in the original context.
            context = example["context"]
            for pred in predictions:
                offsets = pred.pop("offsets")
                pred["text"] = context[offsets[0] : offsets[1]]

            # In the very rare edge case we have not a single non-null prediction,
            # we create a fake prediction to avoid failure.
            if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
                predictions.insert(0, {"text": "", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0,})

            # Compute the softmax of all scores
            # (we do it with numpy to stay independent from torch/tf) in this file,
            #  using the LogSum trick).
            scores = np.array([pred.pop("score") for pred in predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Include the probabilities in our predictions.
            for prob, pred in zip(probs, predictions):
                pred["probability"] = prob

            # Pick the best prediction. If the null answer is not possible, this is easy.
            if not version_2_with_negative:
                all_predictions[example["id"]] = predictions[0]["text"]
            else:
                # Otherwise we first need to find the best non-empty prediction.
                i = 0
                while predictions[i]["text"] == "":
                    i += 1
                best_non_null_pred = predictions[i]

                # Then we compare to the null prediction using the threshold.
                score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
                scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
                if score_diff > null_score_diff_threshold:
                    all_predictions[example["id"]] = ""
                else:
                    all_predictions[example["id"]] = best_non_null_pred["text"]

            # Make `predictions` JSON-serializable by casting np.float back to float.
            all_nbest_json[example[ID_COLUMN_NAME]] = [
                {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
                for pred in predictions
            ]

        # If we have an output_dir, let's save all those dicts.
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                raise EnvironmentError(f"{output_dir} is not a directory.")

            prediction_file = os.path.join(output_dir, INTENSIVE_PRED_FILE_NAME)
            nbest_file = os.path.join(output_dir, NBEST_PRED_FILE_NAME)
            if version_2_with_negative:
                null_odds_file = os.path.join(output_dir, SCORE_DIFF_FILE_NAME)

            logger.info(f"Saving predictions to {prediction_file}.")
            with open(prediction_file, "w") as writer:
                writer.write(json.dumps(all_predictions, indent=4) + "\n")
            logger.info(f"Saving nbest_preds to {nbest_file}.")
            with open(nbest_file, "w") as writer:
                writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
            if version_2_with_negative:
                logger.info(f"Saving null_odds to {null_odds_file}.")
                with open(null_odds_file, "w") as writer:
                    writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

        return all_predictions, all_nbest_json, scores_diff_json, scores_diff_json
    
    
class RearVerifier:
    
    def __init__(
        self, 
        beta1: int = 1, 
        beta2: int = 1,
        best_cof: int = 1,
    ):
        self.beta1 = beta1
        self.beta2 = beta2
        self.best_cof = best_cof
    
    def __call__(
        self,
        score_ext: Dict[str, float],
        score_diff: Dict[str, float],
        nbest_preds: Dict[str, Dict[int, Dict[str, float]]]
    ):
        all_scores = collections.OrderedDict()
        assert score_ext.keys() == score_diff.keys()
        for key in score_ext.keys():
            if key not in all_scores:
                all_scores[key] = []
            all_scores[key].append(
                [self.beta1 * score_ext[key],
                 self.beta2 * score_diff[key]]
            )
        output_scores = {}
        for key, scores in all_scores.items():
            mean_score = sum(scores) / float(len(scores))
            output_scores[key] = mean_score
            
        all_nbest = collections.OrderedDict()
        for key, entries in nbest_preds.items():
            if key not in all_nbest:
                all_nbest[key] = collections.defaultdict(float)
            for entry in entries:
                prob = self.best_cof * entry["probability"]
                all_nbest[key][entry["text"]] += prob
        
        output_predictions = {}
        for key, entry_map in all_nbest.items():
            sorted_texts = sorted(
                entry_map.keys(), key=lambda x: entry_map[x], reverse=True
            )
            best_text = sorted_texts[0]
            output_predictions[key] = best_text
            
        for qid in output_predictions.keys():
            if output_scores[qid] > thresh:
                output_predictions[qid] = ""
                
        return output_predictions, output_scores
    
    
class RetroReader:
    
    def __init__(
        self,
        sketch_reader: Union[SketchReader, Dict[str, Any]],
        intensive_reader: Union[IntensiveReader, Dict[str, Any]],
        tokenizer: PreTrainedTokenizerFast,
        beta1: int = 1, 
        beta2: int = 1,
        best_cof: int = 1,
        data_args={}, # DataArgument
    ):
        self.tokenizer = tokenizer
        if isinstance(sketch_reader, dict):
            sketch_reader = SketchReader(data_args=data_args, **sketch_reader)
        self.sketch_reader = sketch_reader
        self.sketch_prep_fn, _ = get_sketch_features(tokenizer, "test", data_args)
        if isinstance(intensive_reader, dict):
            intensive_reader = IntensiveReader(data_args=data_args, **intensive_reader)
        self.intensive_reader = intensive_reader
        self.intensive_prep_fn, _ = get_intensive_features(tokenizer, "test", data_args)
        self.rear_verifier = RearVerifier(beta1, beta2, best_cof)
        
    def __call__(
        self,
        query: str,
        context: Union[str, List[str]],
    ):
        if isinstance(context, list):
            context = " ".join(context)
        inputs = {
            "example_id": "0",
            QUESTION_COLUMN_NAME: query, 
            CONTEXT_COLUMN_NAME: context
        }
        sketch_features = self.sketch_prep_fn(inputs)
        intensive_features = self.intensive_prep_fn(inputs)
        score_ext = self.sketch_reader.predict(sketch_features, inputs)
        _, nbest_preds, score_diff, _ = self.intensive_reader.predict(intensive_features, inputs)
        predictions, scores = self.rear_verifier(score_ext, score_diff, nbest_preds)
        return predictions, scores
    
    def train(self):
        self.sketch_reader.train()
        self.sketch_reader.free_memory()
        self.intensive_reader.train()
        self.intensive_reader.free_memory()