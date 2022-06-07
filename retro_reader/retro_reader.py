import os
import time
import json
import math
import copy
import collections
from typing import Optional, List, Dict, Tuple, Callable, Any, Union, NewType
import numpy as np
from tqdm import tqdm

import datasets

from transformers import AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers.utils import logging
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction

from .args import (
    HfArgumentParser,
    RetroArguments,
    TrainingArguments,
)
from .base import BaseReader
from . import constants as C
from .preprocess import (
    get_sketch_features,
    get_intensive_features
)
from .metrics import (
    compute_classification_metric,
    compute_squad_v2
)


DataClassType = NewType("DataClassType", Any)
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
        example_id_to_index = {k: i for i, k in enumerate(eval_examples[C.ID_COLUMN_NAME])}
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
        final_map = dict(zip(eval_examples[C.ID_COLUMN_NAME], score_ext.tolist()))
        with open(os.path.join(self.args.output_dir, C.SCORE_EXT_FILE_NAME), "w") as writer:
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
        predictions, nbest_json, scores_diff_json = self.compute_predictions(
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
        if mode == "retro_inference":
            return nbest_json, scores_diff_json
        # Format the result to the format the metric expects.
        if self.data_args.version_2_with_negative:
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
                {"id": ex[C.ID_COLUMN_NAME], "answers": ex[C.ANSWER_COLUMN_NAME]}
                for ex in eval_examples
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
        example_id_to_index = {k: i for i, k in enumerate(examples[C.ID_COLUMN_NAME])}
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
                start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers!
                        # either because the indices are out of bounds
                        # or correspond to part of the input_ids that are note in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or not offset_mapping[start_index]
                            or not offset_mapping[end_index]
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
                all_predictions[example[C.ID_COLUMN_NAME]] = predictions[0]["text"]
            else:
                # Otherwise we first need to find the best non-empty prediction.
                i = 0
                try:
                    while predictions[i]["text"] == "":
                        i += 1
                except:
                    i = 0
                best_non_null_pred = predictions[i]

                # Then we compare to the null prediction using the threshold.
                score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
                scores_diff_json[example[C.ID_COLUMN_NAME]] = float(score_diff)  # To be JSON-serializable.
                if score_diff > null_score_diff_threshold:
                    all_predictions[example[C.ID_COLUMN_NAME]] = ""
                else:
                    all_predictions[example[C.ID_COLUMN_NAME]] = best_non_null_pred["text"]

            # Make `predictions` JSON-serializable by casting np.float back to float.
            all_nbest_json[example[C.ID_COLUMN_NAME]] = [
                {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
                for pred in predictions
            ]

        # If we have an output_dir, let's save all those dicts.
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                raise EnvironmentError(f"{output_dir} is not a directory.")

            prediction_file = os.path.join(output_dir, C.INTENSIVE_PRED_FILE_NAME)
            nbest_file = os.path.join(output_dir, C.NBEST_PRED_FILE_NAME)
            if version_2_with_negative:
                null_odds_file = os.path.join(output_dir, C.SCORE_DIFF_FILE_NAME)

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

        return all_predictions, all_nbest_json, scores_diff_json
    
    
class RearVerifier:
    
    def __init__(
        self, 
        beta1: int = 1, 
        beta2: int = 1,
        best_cof: int = 1,
        thresh: float = 0.0,
    ):
        self.beta1 = beta1
        self.beta2 = beta2
        self.best_cof = best_cof
        self.thresh = thresh
    
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
            all_scores[key].extend(
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
            if output_scores[qid] > self.thresh:
                output_predictions[qid] = ""
                
        return output_predictions, output_scores
    
    
class RetroReader:
    
    def __init__(
        self,
        args,
        sketch_reader: SketchReader,
        intensive_reader: IntensiveReader,
        rear_verifier: RearVerifier,
        prep_fn: Tuple[Callable, Callable],
    ):
        self.args = args
        # Set submodules
        self.sketch_reader = sketch_reader
        self.intensive_reader = intensive_reader
        self.rear_verifier = rear_verifier
        
        # Set prep function for inference
        self.sketch_prep_fn, self.intensive_prep_fn = prep_fn
    
    @classmethod
    def load(
        cls,
        train_examples=None,
        sketch_train_dataset=None,
        intensive_train_dataset=None,
        eval_examples=None,
        sketch_eval_dataset=None,
        intensive_eval_dataset=None,
        config_file: str = C.DEFAULT_CONFIG_FILE,
    ):
        # Get arguments from yaml files
        parser = HfArgumentParser([RetroArguments, TrainingArguments])
        retro_args, training_args = parser.parse_yaml_file(yaml_file=config_file)
        if training_args.run_name is not None and "," in training_args.run_name:
            sketch_run_name, intensive_run_name = training_args.run_name.split(",")
        else:
            sketch_run_name, intensive_run_name = None, None
        if training_args.metric_for_best_model is not None and "," in training_args.metric_for_best_model:
            sketch_best_metric, intensive_best_metric = training_args.metric_for_best_model.split(",")
        else:
            sketch_best_metric, intensive_best_metric = None, None
        sketch_training_args = copy.deepcopy(training_args)
        intensive_training_args = training_args
        
        sketch_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=retro_args.sketch_tokenizer_name,
            use_auth_token=retro_args.use_auth_token,
            revision=retro_args.sketch_revision,
        )
        
        # If `train_examples` is feeded, perform preprocessing
        if train_examples is not None and sketch_train_dataset is None:
            sketch_prep_fn, is_batched = get_sketch_features(sketch_tokenizer, "train", retro_args)
            sketch_train_dataset = train_examples.map(
                sketch_prep_fn,
                batched=is_batched,
                remove_columns=train_examples.column_names,
                num_proc=retro_args.preprocessing_num_workers,
                load_from_cache_file=not retro_args.overwrite_cache,
            )
        # If `eval_examples` is feeded, perform preprocessing
        if eval_examples is not None and sketch_eval_dataset is None:
            sketch_prep_fn, is_batched = get_sketch_features(sketch_tokenizer, "eval", retro_args)
            sketch_eval_dataset = eval_examples.map(
                sketch_prep_fn,
                batched=is_batched,
                remove_columns=eval_examples.column_names,
                num_proc=retro_args.preprocessing_num_workers,
                load_from_cache_file=not retro_args.overwrite_cache,
            )
        # Get preprocessing function for inference
        sketch_prep_fn, _ = get_sketch_features(sketch_tokenizer, "test", retro_args)
        
        # Get model for sketch reader
        sketch_model_cls = retro_args.sketch_model_cls
        sketch_model = sketch_model_cls.from_pretrained(
            pretrained_model_name_or_path=retro_args.sketch_model_name,
            use_auth_token=retro_args.use_auth_token,
            revision=retro_args.sketch_revision,
        )

        # Get sketch reader
        sketch_training_args.run_name = sketch_run_name
        sketch_training_args.output_dir += "/sketch"
        sketch_training_args.metric_for_best_model = sketch_best_metric
        sketch_reader = SketchReader(
            model=sketch_model,
            args=sketch_training_args,
            train_dataset=sketch_train_dataset,
            eval_dataset=sketch_eval_dataset,
            eval_examples=eval_examples,
            data_args=retro_args,
            tokenizer=sketch_tokenizer,
            compute_metrics=compute_classification_metric,
        )
        
        intensive_tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=retro_args.intensive_tokenizer_name,
            use_auth_token=retro_args.use_auth_token,
            revision=retro_args.intensive_revision,
        )
        
        # If `train_examples` is feeded, perform preprocessing
        if train_examples is not None and intensive_train_dataset is None:
            intensive_prep_fn, is_batched = get_intensive_features(intensive_tokenizer, "train", retro_args)
            intensive_train_dataset = train_examples.map(
                intensive_prep_fn,
                batched=is_batched,
                remove_columns=train_examples.column_names,
                num_proc=retro_args.preprocessing_num_workers,
                load_from_cache_file=not retro_args.overwrite_cache,
            )
        # If `eval_examples` is feeded, perform preprocessing
        if eval_examples is not None and intensive_eval_dataset is None:
            intensive_prep_fn, is_batched = get_intensive_features(intensive_tokenizer, "eval", retro_args)
            intensive_eval_dataset = eval_examples.map(
                intensive_prep_fn,
                batched=is_batched,
                remove_columns=eval_examples.column_names,
                num_proc=retro_args.preprocessing_num_workers,
                load_from_cache_file=not retro_args.overwrite_cache,
            )
        # Get preprocessing function for inference
        intensive_prep_fn, _ = get_intensive_features(intensive_tokenizer, "test", retro_args)
        
        # Get model for intensive reader
        intensive_model_cls = retro_args.intensive_model_cls
        intensive_model = intensive_model_cls.from_pretrained(
            pretrained_model_name_or_path=retro_args.intensive_model_name,
            use_auth_token=retro_args.use_auth_token,
            revision=retro_args.intensive_revision,
        )
            
        # Get intensive reader
        intensive_training_args.run_name = intensive_run_name
        intensive_training_args.output_dir += "/intensive"
        intensive_training_args.metric_for_best_model = intensive_best_metric
        intensive_reader = IntensiveReader(
            model=intensive_model,
            args=intensive_training_args,
            train_dataset=intensive_train_dataset,
            eval_dataset=intensive_eval_dataset,
            eval_examples=eval_examples,
            data_args=retro_args,
            tokenizer=intensive_tokenizer,
            compute_metrics=compute_squad_v2,
        )
        
        # Get rear verifier
        rear_verifier = RearVerifier(
            beta1=retro_args.beta1,
            beta2=retro_args.beta2,
            best_cof=retro_args.best_cof,
            thresh=retro_args.rear_threshold,
        )
        
        return cls(
            args=retro_args,
            sketch_reader=sketch_reader,
            intensive_reader=intensive_reader,
            rear_verifier=rear_verifier,
            prep_fn=(sketch_prep_fn, intensive_prep_fn),
        )
        
    def __call__(
        self,
        query: str,
        context: Union[str, List[str]],
        return_submodule_outputs: bool = False,
    ) -> Tuple[Any]:
        if isinstance(context, list):
            context = " ".join(context)
        predict_examples = datasets.Dataset.from_dict({
            "example_id": ["0"],
            C.ID_COLUMN_NAME: ["id-01"],
            C.QUESTION_COLUMN_NAME: [query], 
            C.CONTEXT_COLUMN_NAME: [context]
        })
        return self.inference(predict_examples)
    
    def train(self, module: str = "all"):
        
        def wandb_finish(module):
            for callback in module.callback_handler.callbacks:
                if "wandb" in str(type(callback)).lower():
                    callback._wandb.finish()
                    callback._initialized = False
        
        # Train sketch reader
        if module.lower() in ["all", "sketch"]:
            self.sketch_reader.train()
            self.sketch_reader.save_model()
            self.sketch_reader.save_state()
            self.sketch_reader.free_memory()
            wandb_finish(self.sketch_reader)
        # Train intensive reader
        if module.lower() in ["all", "intensive"]:
            self.intensive_reader.train()
            self.intensive_reader.save_model()
            self.intensive_reader.save_state()
            self.intensive_reader.free_memory()
            wandb_finish(self.intensive_reader)
            
    def inference(self, test_dataset: datasets.Dataset) -> Tuple[Any]:
        if "example_id" not in test_dataset.column_names:
            test_dataset = test_dataset.map(
                lambda _, i: {"example_id": str(i)},
                with_indices=True,
            )
        sketch_features = predict_examples.map(
            self.sketch_prep_fn,
            batched=True,
            remove_columns=predict_examples.column_names,
        )
        intensive_features = predict_examples.map(
            self.intensive_prep_fn,
            batched=True,
            remove_columns=predict_examples.column_names,
        )
        self.sketch_reader.to(self.sketch_reader.args.device)
        score_ext = self.sketch_reader.predict(sketch_features, predict_examples)
        self.sketch_reader.to("cpu")
        self.intensive_reader.to(self.intensive_reader.args.device)
        nbest_preds, score_diff = self.intensive_reader.predict(
            intensive_features, predict_examples, mode="retro_inference")
        self.intensive_reader.to("cpu")
        predictions, scores = self.rear_verifier(score_ext, score_diff, nbest_preds)
        outputs = (predictions, scores)
        if return_submodule_outputs:
            outputs += (score_ext, nbest_preds, score_diff)
        return outputs
            
    @property
    def null_score_diff_threshold(self):
        return self.args.null_score_diff_threshold
    
    @null_score_diff_threshold.setter
    def null_score_diff_threshold(self, val):
        self.args.null_score_diff_threshold = val
        
    @property
    def n_best_size(self):
        return self.args.n_best_size
    
    @n_best_size.setter
    def n_best_size(self, val):
        self.args.n_best_size = val
        
    @property
    def beta1(self):
        return self.rear_verifier.beta1
    
    @beta1.setter
    def beta1(self, val):
        self.rear_verifier.beta1 = val
        
    @property
    def beta2(self):
        return self.rear_verifier.beta2
    
    @beta2.setter
    def beta2(self, val):
        self.rear_verifier.beta2 = val
        
    @property
    def best_cof(self):
        return self.rear_verifier.best_cof
    
    @best_cof.setter
    def best_cof(self, val):
        self.rear_verifier.best_cof = val
        
    @property
    def rear_threshold(self):
        return self.rear_verifier.thresh
    
    @rear_threshold.setter
    def rear_threshold(self, val):
        self.rear_verifier.thresh = val
