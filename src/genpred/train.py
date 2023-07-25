"Trains model."
import time
from pathlib import Path

import hydra
import joblib
from omegaconf import DictConfig

import genpred
from genpred.training.pipeline import create_pipeline
from genpred.training.utils import (
    save_model,
    save_cv_results,
    save_test_metrics,
)
from genpred.utils import utils, logger


log = logger.get_pylogger(__name__)
# pylint:disable=logging-fstring-interpolation


@hydra.main(
    config_path=genpred.CFG_ROOT.as_posix(),
    config_name="train.yaml",
    version_base="1.2",
)
def main(cfg: DictConfig) -> None:
    "Trains model."
    start_time = time.time()

    utils.extras(cfg)

    utils.seed_everything(cfg.seed)
    log.info(f"Global seed: <cfg.seed={cfg.seed}>")

    dataset = hydra.utils.instantiate(cfg.dataset)
    pipeline, hparams = create_pipeline(cfg.pipeline)

    log.info("Starting training.")
    if not Path("model.pkl").exists():
        tuner = hydra.utils.instantiate(
            cfg.tuner,
            cv=dataset.split,
            estimator=pipeline,
            param_distributions=hparams,
            _convert_="partial",
        ).fit(dataset.X_train, dataset.y_train)
        log.info("Training completed!")

        log.info("Saving model.")
        save_model(tuner)
    else:
        tuner = joblib.load("model.pkl")

    log.info(f"Best params: {tuner.best_params_}")
    log.info(f"Best metric: {tuner.best_score_}")

    log.info("Saving CV metrics.")
    save_cv_results(tuner.cv_results_)

    best_model = tuner.best_estimator_

    log.info("Computing and saving test metrics.")

    save_test_metrics(
        best_model,
        dataset.X_test,
        dataset.y_test,
        dataset.G_test,
    )

    elapsed = time.time() - start_time
    with open("exec_time.log", "w", encoding="utf-8") as file:
        print(f"Execution time: {elapsed} (s)", file=file)

    log.info("Done! Closing.")


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
