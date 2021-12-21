import os
import gc
import time
import json
import math
import collections
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Callable, Any, Union

import torch
import numpy as np

from transformers import (
    is_datasets_available,
    is_torch_tpu_available,
)

from transformers.trainer_utils import (
    PredictionOutput,
    EvalPrediction,
    EvalLoopOutput,
    denumpify_detensorize,
    speed_metrics,
)

from transformers.utils import logging
from transformers.debug_utils import DebugOption

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    
from transformers import Trainer

logger = logging.get_logger(__name__)


class ToMixin:

    def _optimizer_to(self, device: str = "cpu"):
        # https://github.com/pytorch/pytorch/issues/8741
        for param in self.optimizer.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(
                                device)

    def _scheduler_to(self, device: str = "cpu"):
        # https://github.com/pytorch/pytorch/issues/8741
        for param in self.lr_scheduler.__dict__.values():
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
                    

class BaseReader(Trainer, ToMixin):
    name: str = None
    
    def __init__(
        self,
        *args,
        data_args = {},
        eval_examples: datasets.Dataset = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.data_args = data_args
        self.eval_examples = eval_examples
        
    def free_memory(self):
        self.model.to("cpu")
        self._optimizer_to("cpu")
        self._scheduler_to("cpu")
        torch.cuda.empty_cache()
        gc.collect()
        
    def postprocess(
        self,
        output: EvalLoopOutput,
    ) -> Union[Any, EvalPrediction]:
        return output
        
    def evaluate(
        self,
        eval_dataset: Optional[datasets.Dataset] = None,
        eval_examples: Optional[datasets.Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
        
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics
            
        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )
            
        eval_preds = self.postprocess(output, eval_examples, eval_dataset, mode="evaluate")
        
        metrics = {}
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(eval_preds)
            # To be JSON-serializable, we need to remove numpy types or zero-d tensors
            metrics = denumpify_detensorize(metrics)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            total_batch_size = self.args.eval_batch_size * self.args.world_size
            metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )
            self.log(metrics)
        
        # Log and save evaluation results
        filename = "eval_results.txt"
        eval_result_file = self.name + "_" + filename if self.name else filename
        with open(os.path.join(self.args.output_dir, eval_result_file), "a")  as writer:
            logger.info("***** Eval results *****")
            writer.write("***** Eval results *****\n")
            writer.write(f"{datetime.now()}")
            for key in sorted(metrics.keys()):
                logger.info("  %s = %s", key, str(metrics[key]))
                writer.write("%s = %s\n" % (key, str(metrics[key])))
            writer.write("\n")
                
        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: PyTorch/XLA에 대한 Logging debug metrics (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        self._memory_tracker.stop_and_update_metrics(metrics)
        
        return metrics
    
    def predict(
        self,
        test_dataset: datasets.Dataset,
        test_examples: datasets.Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        mode: bool = "predict",
    ) -> PredictionOutput:
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        
        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()
        
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics
            
        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(
                type=test_dataset.format["type"],
                columns=list(test_dataset.features.keys()),
            )
            
        predictions = self.postprocess(output, test_examples, test_dataset, mode=mode)
            
        self._memory_tracker.stop_and_update_metrics(output.metrics)
        
        return predictions