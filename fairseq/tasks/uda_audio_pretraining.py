# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
import torch

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Any
from omegaconf import MISSING, II

from fairseq.data import AddTargetDataset, AddAuxTargetDataset, Dictionary, FileAudioDataset, encoders, FairseqDataset, data_utils, iterators
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import GenerationConfig

from fairseq.tasks.audio_pretraining import AudioPretrainingConfig, AudioPretrainingTask

from . import FairseqTask, register_task
from .. import utils
from ..logging import metrics


logger = logging.getLogger(__name__)


@dataclass
class UdaAudioPretrainingConfig(AudioPretrainingConfig):
    aux_labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the class label file to load, used for domain adaptative fine-tuning"},
    )


@register_task("uda_audio_pretraining", dataclass=UdaAudioPretrainingConfig)
class UdaAudioPretrainingTask(AudioPretrainingTask):
    """"""

    cfg: UdaAudioPretrainingConfig

    def __init__(
        self,
        cfg: UdaAudioPretrainingConfig,
    ):
        super().__init__(cfg)

    def load_dataset(
            self, split: str, task_cfg: FairseqDataclass = None, **kwargs
    ):
        super().load_dataset(split, task_cfg)
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == 'ctc'

        if task_cfg.aux_labels:
            aux_label_path = os.path.join(data_path, f"{split}.{task_cfg.aux_labels}")
            with open(aux_label_path, "r") as f:
                aux_labels = [
                    int(line.strip()) for i, line in enumerate(f)
                    if i in self.datasets[split].line_inds
                ]

            assert len(aux_labels) == len(self.datasets[split]), (
                    f"aux_labels length ({len(aux_labels)}) and dataset length "
                    f"({len(self.datasets[split])}) do not match")

            self.datasets[split] = AddAuxTargetDataset(
                self.datasets[split],
                aux_labels,
                batch_targets=True,
                add_to_input=task_cfg.get('autoregressive', False),
            )

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
        is_train = True,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 1).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
            is_train (bool, optional): training or not (validation).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
            dataset
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            group_indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            group_indices = [self.filter_indices_by_size(
                indices, dataset, max_positions, ignore_invalid_inputs
            ) for indices in group_indices]

        # create mini-batches with given size constraints
        source_batch_sampler = dataset.batch_by_size(
            group_indices[0],
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        target_batch_sampler = dataset.batch_by_size(
            group_indices[1],
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochGroupBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            group_batch_sampler=(source_batch_sampler, target_batch_sampler),
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
            is_train=is_train,
            resampling_target=True
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

    def begin_epoch(self, epoch, model):
        """Hook function called before the start of each epoch."""
        self.epoch = epoch

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        criterion.epoch = self.epoch
        loss, sample_size, logging_output = super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad)
        return loss, sample_size, logging_output