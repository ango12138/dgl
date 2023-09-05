"""Item Sampler"""

from collections.abc import Mapping
from functools import partial
from typing import Callable, Iterator, Optional

from torch.utils.data import default_collate
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe

from ..base import dgl_warning

from ..batch import batch as dgl_batch
from ..heterograph import DGLGraph
from .itemset import ItemSet, ItemSetDict
from .minibatch import MiniBatch

__all__ = ["ItemSampler", "minibatcher_default"]


def minibatcher_default(batch, names):
    """Default minibatcher.

    Parameters
    ----------
    batch : list
        List of items.

    Returns
    -------
    MiniBatch
        A minibatch.
    """
    if names is None:
        dgl_warning(
            "Failed to map item list to `MiniBatch` as the names of items are "
            "not provided. Please provide a customized `MiniBatcher`. "
            "The item list is returned as is."
        )
        return batch
    if len(names) == 1:
        init_data = {names[0]: batch}
    else:
        if isinstance(batch, Mapping):
            init_data = {
                name: {k: v[i] for k, v in batch.items()}
                for i, name in enumerate(names)
            }
        else:
            init_data = {name: item for item, name in zip(batch, names)}
    minibatch = MiniBatch(**init_data)
    return minibatch


class ItemSampler(IterDataPipe):
    """Item Sampler.

    Creates item subset of data which could be node IDs, node pairs with or
    without labels, node pairs with negative sources/destinations, DGLGraphs
    and heterogeneous counterparts.

    Note: This class `ItemSampler` is not decorated with
    `torchdata.datapipes.functional_datapipe` on purpose. This indicates it
    does not support function-like call. But any iterable datapipes from
    `torchdata` can be further appended.

    Parameters
    ----------
    item_set : ItemSet or ItemSetDict
        Data to be sampled.
    batch_size : int
        The size of each batch.
    minibatcher : Optional[Callable]
        A callable that takes in a list of items and returns a `MiniBatch`.
    drop_last : bool
        Option to drop the last batch if it's not full.
    shuffle : bool
        Option to shuffle before sample.

    Examples
    --------
    1. Node IDs.
    >>> import torch
    >>> from dgl import graphbolt as gb
    >>> item_set = gb.ItemSet(torch.arange(0, 10))
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=True, drop_last=False
    ... )
    >>> list(item_sampler)
    [tensor([1, 2, 5, 7]), tensor([3, 0, 9, 4]), tensor([6, 8])]

    2. Node pairs.
    >>> item_set = gb.ItemSet((torch.arange(0, 10), torch.arange(10, 20)))
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=True, drop_last=False
    ... )
    >>> list(item_sampler)
    [[tensor([9, 8, 3, 1]), tensor([19, 18, 13, 11])], [tensor([2, 5, 7, 4]),
    tensor([12, 15, 17, 14])], [tensor([0, 6]), tensor([10, 16])]

    3. Node pairs and labels.
    >>> item_set = gb.ItemSet(
    ...     (torch.arange(0, 5), torch.arange(5, 10), torch.arange(10, 15))
    ... )
    >>> item_sampler = gb.ItemSampler(item_set, 3)
    >>> list(item_sampler)
    [[tensor([0, 1, 2]), tensor([5, 6, 7]), tensor([10, 11, 12])],
    [tensor([3, 4]), tensor([8, 9]), tensor([13, 14])]]

    4. Head, tail and negative tails
    >>> heads = torch.arange(0, 5)
    >>> tails = torch.arange(5, 10)
    >>> negative_tails = torch.stack((heads + 1, heads + 2), dim=-1)
    >>> item_set = gb.ItemSet((heads, tails, negative_tails))
    >>> item_sampler = gb.ItemSampler(item_set, 3)
    >>> list(item_sampler)
    [[tensor([0, 1, 2]), tensor([5, 6, 7]),
        tensor([[1, 2], [2, 3], [3, 4]])],
    [tensor([3, 4]), tensor([8, 9]), tensor([[4, 5], [5, 6]])]]

    5. DGLGraphs.
    >>> import dgl
    >>> graphs = [ dgl.rand_graph(10, 20) for _ in range(5) ]
    >>> item_set = gb.ItemSet(graphs)
    >>> item_sampler = gb.ItemSampler(item_set, 3)
    >>> list(item_sampler)
    [Graph(num_nodes=30, num_edges=60,
      ndata_schemes={}
      edata_schemes={}),
     Graph(num_nodes=20, num_edges=40,
      ndata_schemes={}
      edata_schemes={})]

    6. Further process batches with other datapipes such as
    `torchdata.datapipes.iter.Mapper`.
    >>> item_set = gb.ItemSet(torch.arange(0, 10))
    >>> data_pipe = gb.ItemSampler(item_set, 4)
    >>> def add_one(batch):
    ...     return batch + 1
    >>> data_pipe = data_pipe.map(add_one)
    >>> list(data_pipe)
    [tensor([1, 2, 3, 4]), tensor([5, 6, 7, 8]), tensor([ 9, 10])]

    7. Heterogeneous node IDs.
    >>> ids = {
    ...     "user": gb.ItemSet(torch.arange(0, 5)),
    ...     "item": gb.ItemSet(torch.arange(0, 6)),
    ... }
    >>> item_set = gb.ItemSetDict(ids)
    >>> item_sampler = gb.ItemSampler(item_set, 4)
    >>> list(item_sampler)
    [{'user': tensor([0, 1, 2, 3])},
    {'item': tensor([0, 1, 2]), 'user': tensor([4])},
    {'item': tensor([3, 4, 5])}]

    8. Heterogeneous node pairs.
    >>> node_pairs_like = (torch.arange(0, 5), torch.arange(0, 5))
    >>> node_pairs_follow = (torch.arange(0, 6), torch.arange(6, 12))
    >>> item_set = gb.ItemSetDict({
    ...     "user:like:item": gb.ItemSet(node_pairs_like),
    ...     "user:follow:user": gb.ItemSet(node_pairs_follow),
    ... })
    >>> item_sampler = gb.ItemSampler(item_set, 4)
    >>> list(item_sampler)
    [{"user:like:item": [tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3])]},
    {"user:like:item": [tensor([4]), tensor([4])],
     "user:follow:user": [tensor([0, 1, 2]), tensor([6, 7, 8])]},
    {"user:follow:user": [tensor([3, 4, 5]), tensor([ 9, 10, 11])]}]

    9. Heterogeneous node pairs and labels.
    >>> like = (
    ...     torch.arange(0, 5), torch.arange(0, 5), torch.arange(0, 5))
    >>> follow = (
    ...     torch.arange(0, 6), torch.arange(6, 12), torch.arange(0, 6))
    >>> item_set = gb.ItemSetDict({
    ...     "user:like:item": gb.ItemSet(like),
    ...     "user:follow:user": gb.ItemSet(follow),
    ... })
    >>> item_sampler = gb.ItemSampler(item_set, 4)
    >>> list(item_sampler)
    [{"user:like:item":
        [tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3])]},
     {"user:like:item": [tensor([4]), tensor([4]), tensor([4])],
      "user:follow:user":
        [tensor([0, 1, 2]), tensor([6, 7, 8]), tensor([0, 1, 2])]},
     {"user:follow:user":
        [tensor([3, 4, 5]), tensor([ 9, 10, 11]), tensor([3, 4, 5])]}]

    10. Heterogeneous head, tail and negative tails.
    >>> like = (
    ...     torch.arange(0, 5), torch.arange(0, 5),
    ...     torch.arange(5, 15).reshape(-1, 2))
    >>> follow = (
    ...     torch.arange(0, 6), torch.arange(6, 12),
    ...     torch.arange(12, 24).reshape(-1, 2))
    >>> item_set = gb.ItemSetDict({
    ...     "user:like:item": gb.ItemSet(like),
    ...     "user:follow:user": gb.ItemSet(follow),
    ... })
    >>> item_sampler = gb.ItemSampler(item_set, 4)
    >>> list(item_sampler)
    [{"user:like:item": [tensor([0, 1, 2, 3]), tensor([0, 1, 2, 3]),
        tensor([[ 5,  6], [ 7,  8], [ 9, 10], [11, 12]])]},
     {"user:like:item": [tensor([4]), tensor([4]), tensor([[13, 14]])],
      "user:follow:user": [tensor([0, 1, 2]), tensor([6, 7, 8]),
        tensor([[12, 13], [14, 15], [16, 17]])]},
     {"user:follow:user": [tensor([3, 4, 5]), tensor([ 9, 10, 11]),
        tensor([[18, 19], [20, 21], [22, 23]])]}]
    """

    def __init__(
        self,
        item_set: ItemSet or ItemSetDict,
        batch_size: int,
        minibatcher: Optional[Callable] = minibatcher_default,
        drop_last: Optional[bool] = False,
        shuffle: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self._item_set = item_set
        self._batch_size = batch_size
        self._minibatcher = minibatcher
        self._drop_last = drop_last
        self._shuffle = shuffle

    def __iter__(self) -> Iterator:
        data_pipe = IterableWrapper(self._item_set)
        # Shuffle before batch.
        if self._shuffle:
            # `torchdata.datapipes.iter.Shuffler` works with stream too.
            # To ensure randomness, make sure the buffer size is at least 10
            # times the batch size.
            buffer_size = max(10000, 10 * self._batch_size)
            data_pipe = data_pipe.shuffle(buffer_size=buffer_size)

        # Batch.
        data_pipe = data_pipe.batch(
            batch_size=self._batch_size,
            drop_last=self._drop_last,
        )

        # Collate.
        def _collate(batch):
            data = next(iter(batch))
            if isinstance(data, DGLGraph):
                return dgl_batch(batch)
            elif isinstance(data, Mapping):
                assert len(data) == 1, "Only one type of data is allowed."
                # Collect all the keys.
                keys = {key for item in batch for key in item.keys()}
                # Collate each key.
                return {
                    key: default_collate(
                        [item[key] for item in batch if key in item]
                    )
                    for key in keys
                }
            return default_collate(batch)

        data_pipe = data_pipe.collate(collate_fn=partial(_collate))

        # Map to minibatch.
        data_pipe = data_pipe.map(
            partial(self._minibatcher, names=self._item_set.names)
        )

        return iter(data_pipe)
