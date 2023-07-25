"""Rich utils."""

from typing import Sequence

import rich
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf

from genpred.utils import logger

log = logger.get_pylogger(__name__)


# pylint:disable=logging-fstring-interpolation


def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = ("vectorizer", "pipeline", "tuner"),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in
            what order config components are printed.
        resolve (bool, optional): Whether to resolve reference
            fields of DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        if field in cfg:
            queue.append(field)

    # add all the other fields to queue (not specified in `print_order`)
    for field in ["dataset", "strategy", "vocab_size"]:
        if (field in cfg) and (field not in queue) and (cfg[field] is not None):
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open("config.log", "w", encoding="utf-8") as file:
            rich.print(tree, file=file)
