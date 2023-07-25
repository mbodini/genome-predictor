"""Various utils."""

import random
import os
import warnings

import numpy as np
from omegaconf import DictConfig

from genpred.utils import logger, rich


# pylint:disable=logging-fstring-interpolation


log = logger.get_pylogger(__name__)


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.
    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # disable python warnings
    if cfg.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library
    if cfg.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.print_config=True>")
        rich.print_config_tree(cfg, resolve=True, save_to_file=True)


def seed_everything(seed: int) -> None:
    """Sets random seed globally."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
