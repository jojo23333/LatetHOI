# Copyright (c) Facebook, Inc. and its affiliates.
import itertools
import logging
import torch
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
from d2.utils.env import seed_all_rng

__all__ = ["ToIterableDataset"]

logger = logging.getLogger(__name__)


# copied from: https://docs.python.org/3/library/itertools.html#recipes
def _roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            # Remove the iterator we just exhausted from the cycle.
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))


def _shard_iterator_dataloader_worker(iterable, chunk_size=1):
    # Shard the iterable if we're currently inside pytorch dataloader worker.
    worker_info = data.get_worker_info()
    if worker_info is None or worker_info.num_workers == 1:
        # do nothing
        yield from iterable
    else:
        # worker0: 0, 1, ..., chunk_size-1, num_workers*chunk_size, num_workers*chunk_size+1, ...
        # worker1: chunk_size, chunk_size+1, ...
        # worker2: 2*chunk_size, 2*chunk_size+1, ...
        # ...
        yield from _roundrobin(
            *[
                itertools.islice(
                    iterable,
                    worker_info.id * chunk_size + chunk_i,
                    None,
                    worker_info.num_workers * chunk_size,
                )
                for chunk_i in range(chunk_size)
            ]
        )

class ToIterableDataset(data.IterableDataset):
    """
    Convert an old indices-based (also called map-style) dataset
    to an iterable-style dataset.
    """

    def __init__(
        self,
        dataset: data.Dataset,
        sampler: Sampler,
        shard_sampler: bool = True,
        shard_chunk_size: int = 1,
    ):
        """
        Args:
            dataset: an old-style dataset with ``__getitem__``
            sampler: a cheap iterable that produces indices to be applied on ``dataset``.
            shard_sampler: whether to shard the sampler based on the current pytorch data loader
                worker id. When an IterableDataset is forked by pytorch's DataLoader into multiple
                workers, it is responsible for sharding its data based on worker id so that workers
                don't produce identical data.

                Most samplers (like our TrainingSampler) do not shard based on dataloader worker id
                and this argument should be set to True. But certain samplers may be already
                sharded, in that case this argument should be set to False.
            shard_chunk_size: when sharding the sampler, each worker will
        """
        assert not isinstance(dataset, data.IterableDataset), dataset
        assert isinstance(sampler, Sampler), sampler
        self.dataset = dataset
        self.sampler = sampler
        self.shard_sampler = shard_sampler
        self.shard_chunk_size = shard_chunk_size

    def __iter__(self):
        if not self.shard_sampler:
            sampler = self.sampler
        else:
            # With map-style dataset, `DataLoader(dataset, sampler)` runs the
            # sampler in main process only. But `DataLoader(ToIterableDataset(dataset, sampler))`
            # will run sampler in every of the N worker. So we should only keep 1/N of the ids on
            # each worker. The assumption is that sampler is cheap to iterate so it's fine to
            # discard ids in workers.
            sampler = _shard_iterator_dataloader_worker(self.sampler, self.shard_chunk_size)
        for idx in sampler:
            yield self.dataset[idx]

    def __len__(self):
        return len(self.sampler)


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2**31
    seed_all_rng(initial_seed + worker_id)
