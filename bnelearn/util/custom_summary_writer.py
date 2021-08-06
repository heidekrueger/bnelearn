import time

import torch
from typing import List
from torch.utils.tensorboard.summary import hparams
from torch.utils.tensorboard.writer import FileWriter, SummaryWriter, scalar

class CustomSummaryWriter(SummaryWriter):
    """
    Extends SummaryWriter with two methods:

    * a method to add multiple scalars in the way that we intend. The original
        SummaryWriter can either add a single scalar at a time or multiple scalars,
        but in the latter case, multiple runs are created without
        the option to control these.
    * overwriting the the add_hparams method to write hparams without creating
        another tensorboard run file
    """

    def add_hparams(self, hparam_dict=None, metric_dict=None, global_step=None):
        """
        Overides the parent method to prevent the creation of unwanted additional subruns while logging hyperparams,
        as it is done by the original PyTorch method
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict)


        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)

        for k, v in metric_dict.items():
            self.add_scalar(k, v, global_step=global_step)

    def add_metrics_dict(self, metrics_dict: dict, run_suffices: List[str],
                         global_step=None, walltime=None,
                         group_prefix: str = None, metric_tag_mapping: dict = None):
        """
        Args:
            metric_dict (dict): A dict of metrics. Keys are tag names, values are values.
                values can be float, List[float] or Tensor.
                When List or (nonscalar) tensor, the length must match n_models
            run_suffices (List[str]): if each value in metrics_dict is scalar, doesn't need to be supplied.
                When metrics_dict contains lists/iterables, they must all have the same length which should be equal to
                the length of run_suffices
            global_step (int, optional): The step/iteration at which the metrics are being logged.
            walltime
            group_prefix (str, optional): If given each metric name will be prepended with this prefix (and a '/'), 
                which will group tags in tensorboard into categories.
            metric_tag_mapping (dict, optional): A dactionary that provides a mapping between the metrics (keys of metrics_dict)
                and the desired tag names in tensorboard. If given, each metric name will be converted to the corresponding tag name.
                NOTE: bnelearn.util.metrics.MAPPING_METRICS_TAGS contains a standard mapping for common metrics. 
                These already include (metric-specific) prefixes.
        """
        torch._C._log_api_usage_once("tensorboard.logging.add_scalar")
        walltime = time.time() if walltime is None else walltime
        fw_logdir = self._get_file_writer().get_logdir()

        if run_suffices is None:
            run_suffices = []

        l = len(run_suffices)

        for key, vals in metrics_dict.items():
            if metric_tag_mapping:
                # check if key matches any of the names in the dictionary
                matches = [k for k in metric_tag_mapping if key.startswith(k)]
                if matches:
                    key = key.replace(matches[0], metric_tag_mapping[matches[0]])
            tag = key if not group_prefix else group_prefix + '/' + key

            if isinstance(vals, float) or isinstance(vals, int) or (
                    torch.is_tensor(vals) and vals.size() in {torch.Size([]), torch.Size([1])}):
                # Only a single value --> log directly in main run
                self.add_scalar(tag, vals, global_step, walltime)
            elif len(vals) == 1:
                # List type of length 1, but not tensor --> extract item
                self.add_scalar(tag, vals[0], global_step, walltime)
            elif len(vals) == l:
                # Log each into a run with its own prefix.
                for suffix, scalar_value in zip(run_suffices, vals):
                    fw_tag = fw_logdir + "/" + suffix.replace("/", "_")

                    if fw_tag in self.all_writers.keys():
                        fw = self.all_writers[fw_tag]
                    else:
                        fw = FileWriter(fw_tag, self.max_queue, self.flush_secs,
                                        self.filename_suffix)
                        self.all_writers[fw_tag] = fw
                    # Not using caffe2 -->following line is commented out from original SummaryWriter implementation
                    # if self._check_caffe2_blob(scalar_value):
                    #     scalar_value = workspace.FetchBlob(scalar_value)
                    fw.add_summary(scalar(tag, scalar_value), global_step, walltime)
            else:
                raise ValueError('Got list of invalid length.')
