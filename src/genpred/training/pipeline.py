"""Pipeline."""
from typing import Tuple

import hydra
from omegaconf import DictConfig
from sklearn.pipeline import Pipeline


def create_pipeline(pipe_cfg: DictConfig) -> Tuple[Pipeline, dict]:
    """Creates pipeline based on config."""
    step_names = ["feature_filtering", "classifier"]

    steps, hparams_dict = [], {}
    for step_name in step_names:
        if step_name in pipe_cfg and pipe_cfg[step_name]:
            step = hydra.utils.instantiate(pipe_cfg[step_name].estimator)
            hparams = hydra.utils.instantiate(pipe_cfg[step_name].hparams)
            steps.append((step_name, step))
            hparams_dict |= hparams

    pipeline = Pipeline(steps)
    return pipeline, hparams_dict
