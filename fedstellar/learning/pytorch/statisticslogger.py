import logging
import os
from argparse import Namespace
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
from lightning_utilities.core.imports import RequirementCache
from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams
from torch import Tensor

import lightning as pl
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.fabric.utilities.types import _PATH
from lightning.pytorch.core.saving import save_hparams_to_yaml
from lightning.pytorch.loggers.logger import Logger, rank_zero_experiment
from lightning.fabric.utilities.logger import _add_prefix, _convert_params, _flatten_dict, _sanitize_callable_params
from lightning.fabric.utilities.logger import _sanitize_params as _utils_sanitize_params
from lightning.pytorch.utilities.rank_zero import rank_zero_only, rank_zero_warn

log = logging.getLogger(__name__)

_TENSORBOARD_AVAILABLE = RequirementCache("tensorboard")


class FedstellarLogger(Logger):

    NAME_HPARAMS_FILE = "hparams.yaml"
    LOGGER_JOIN_CHAR = "-"

    def __init__(
        self,
        save_dir: _PATH,
        name: Optional[str] = "lightning_logs",
        version: Optional[Union[int, str]] = None,
        log_graph: bool = False,
        default_hp_metric: bool = True,
        prefix: str = "",
        sub_dir: Optional[_PATH] = None,
        **kwargs: Any,
    ):
        super().__init__()
        save_dir = os.fspath(save_dir)
        self._save_dir = save_dir
        self._name = name or ""
        self._version = version
        self._sub_dir = None if sub_dir is None else os.fspath(sub_dir)
        if log_graph and not _TENSORBOARD_AVAILABLE:
            rank_zero_warn("You set `TensorBoardLogger(log_graph=True)` but `tensorboard` is not available.")
        self._log_graph = log_graph and _TENSORBOARD_AVAILABLE

        self._default_hp_metric = default_hp_metric
        self._prefix = prefix
        self._fs = get_filesystem(save_dir)

        self._experiment: Optional["SummaryWriter"] = None
        self.hparams: Union[Dict[str, Any], Namespace] = {}
        self._kwargs = kwargs

        self.local_step = 0
        self.global_step = 0

    @property
    def root_dir(self) -> str:

        return os.path.join(self.save_dir, self.name)

    @property
    def log_dir(self) -> str:

        # create a pseudo standard path ala test-tube
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        log_dir = os.path.join(self.root_dir, version)
        if isinstance(self.sub_dir, str):
            log_dir = os.path.join(log_dir, self.sub_dir)
        log_dir = os.path.expandvars(log_dir)
        log_dir = os.path.expanduser(log_dir)
        return log_dir

    @property
    def save_dir(self) -> str:

        return self._save_dir

    @property
    def sub_dir(self) -> Optional[str]:

        return self._sub_dir

    @property
    @rank_zero_experiment
    def experiment(self) -> SummaryWriter:

        if self._experiment is not None:
            return self._experiment

        assert rank_zero_only.rank == 0, "tried to init log dirs in non global_rank=0"
        if self.root_dir:
            self._fs.makedirs(self.root_dir, exist_ok=True)
        self._experiment = SummaryWriter(log_dir=self.log_dir, **self._kwargs)
        return self._experiment

    @rank_zero_only
    def log_hyperparams(
        self, params: Union[Dict[str, Any], Namespace], metrics: Optional[Dict[str, Any]] = None
    ) -> None:

        params = _convert_params(params)

        # store params to output
        self.hparams.update(params)

        # format params into the suitable for tensorboard
        params = _flatten_dict(params)
        params = self._sanitize_params(params)

        if metrics is None:
            if self._default_hp_metric:
                metrics = {"hp_metric": -1}
        elif not isinstance(metrics, dict):
            metrics = {"hp_metric": metrics}

        if metrics:
            self.log_metrics(metrics, 0)
            exp, ssi, sei = hparams(params, metrics)
            writer = self.experiment._get_file_writer()
            writer.add_summary(exp)
            writer.add_summary(ssi)
            writer.add_summary(sei)

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        # FL round information
        self.local_step = step
        __step = self.global_step + self.local_step

        metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
        # logging.info(f"[Statisticslogger] Logging metrics: {metrics}, step: {__step}")

        for k, v in metrics.items():
            if isinstance(v, Tensor):
                v = v.item()

            if isinstance(v, dict):
                self.experiment.add_scalars(k, v, __step)
            else:
                try:
                    self.experiment.add_scalar(k, v, __step)
                # todo: specify the possible exception
                except Exception as ex:
                    m = f"\n you tried to log {v} which is currently not supported. Try a dict or a scalar/tensor."
                    raise ValueError(m) from ex

    @rank_zero_only
    def log_graph(self, model: "pl.LightningModule", input_array: Optional[Tensor] = None) -> None:
        if not self._log_graph:
            return

        input_array = model.example_input_array if input_array is None else input_array

        if input_array is None:
            rank_zero_warn(
                "Could not log computational graph to TensorBoard: The `model.example_input_array` attribute"
                " is not set or `input_array` was not given."
            )
        elif not isinstance(input_array, (Tensor, tuple)):
            rank_zero_warn(
                "Could not log computational graph to TensorBoard: The `input_array` or `model.example_input_array`"
                f" has type {type(input_array)} which can't be traced by TensorBoard. Make the input array a tuple"
                f" representing the positional arguments to the model's `forward()` implementation."
            )
        else:
            input_array = model._on_before_batch_transfer(input_array)
            input_array = model._apply_batch_transfer_handler(input_array)
            with pl.pytorch.core.module._jit_is_scripting():
                self.experiment.add_graph(model, input_array)

    @rank_zero_only
    def save(self) -> None:
        super().save()
        dir_path = self.log_dir

        # prepare the file path
        hparams_file = os.path.join(dir_path, self.NAME_HPARAMS_FILE)

        # save the metatags file if it doesn't exist and the log directory exists
        if self._fs.isdir(dir_path) and not self._fs.isfile(hparams_file):
            save_hparams_to_yaml(hparams_file, self.hparams)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._experiment is not None:
            self.experiment.flush()
            self.experiment.close()

        if status == "success":
            # saving hparams happens independent of experiment manager
            self.save()

    @property
    def name(self) -> str:

        return self._name

    @property
    def version(self) -> Union[int, str]:

        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    def _get_next_version(self) -> int:
        root_dir = self.root_dir

        try:
            listdir_info = self._fs.listdir(root_dir)
        except OSError:
            log.warning("Missing logger folder: %s", root_dir)
            return 0

        existing_versions = []
        for listing in listdir_info:
            d = listing["name"]
            bn = os.path.basename(d)
            if self._fs.isdir(d) and bn.startswith("version_"):
                dir_ver = bn.split("_")[1].replace("/", "")
                existing_versions.append(int(dir_ver))
        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1

    @staticmethod
    def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
        params = _utils_sanitize_params(params)
        # logging of arrays with dimension > 1 is not supported, sanitize as string
        return {k: str(v) if isinstance(v, (Tensor, np.ndarray)) and v.ndim > 1 else v for k, v in params.items()}

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        state["_experiment"] = None
        return state
