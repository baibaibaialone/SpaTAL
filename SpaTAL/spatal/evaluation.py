import copy
import math
from numbers import Integral

import torch


def _copy_state_dict_to_cpu(state_dict):
    copied_state = type(state_dict)()
    for name, value in state_dict.items():
        if torch.is_tensor(value):
            copied_state[name] = value.detach().to(device="cpu", copy=True)
        else:
            copied_state[name] = copy.deepcopy(value)

    if hasattr(state_dict, "_metadata"):
        copied_state._metadata = copy.deepcopy(state_dict._metadata)
    return copied_state


def should_evaluate(completed_epoch, max_epoch, interval):
    """Return whether to evaluate after completed epochs, starting at 1."""
    if (
        isinstance(interval, bool)
        or not isinstance(interval, Integral)
        or interval <= 0
    ):
        raise ValueError("interval must be a positive integer")
    return completed_epoch % interval == 0 or completed_epoch == max_epoch


class EvaluationCheckpointManager:
    def __init__(self):
        self.history = []
        self.best_result = None
        self._best_model_state = None
        self._best_graph_learner_state = None
        self._best_adata = None

    def record(self, epoch, result, model, graph_learner, adata):
        try:
            score_is_finite = math.isfinite(result["score"])
        except (TypeError, ValueError) as error:
            raise ValueError("score must be a finite number") from error
        if not score_is_finite:
            raise ValueError("score must be a finite number")

        record = copy.deepcopy(result)
        record["epoch"] = epoch

        if self.best_result is None or record["score"] > self.best_result["score"]:
            best_result = copy.deepcopy(record)
            best_model_state = _copy_state_dict_to_cpu(model.state_dict())
            best_graph_learner_state = _copy_state_dict_to_cpu(
                graph_learner.state_dict()
            )
            best_adata = adata.copy()

            self.history.append(record)
            self.best_result = best_result
            self._best_model_state = best_model_state
            self._best_graph_learner_state = best_graph_learner_state
            self._best_adata = best_adata
        else:
            self.history.append(record)

    def restore(self, model, graph_learner):
        if self.best_result is None:
            return None
        model.load_state_dict(self._best_model_state)
        graph_learner.load_state_dict(self._best_graph_learner_state)
        return self._best_adata.copy()
