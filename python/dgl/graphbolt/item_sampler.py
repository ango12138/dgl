"""Item Sampler"""

from collections.abc import Mapping
from functools import partial
from typing import Callable, Iterator, Optional

import torch.distributed as dist
from torch.utils.data import default_collate
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe

from ..base import dgl_warning

from ..batch import batch as dgl_batch
from ..heterograph import DGLGraph
from .itemset import ItemSet, ItemSetDict
from .minibatch import MiniBatch

__all__ = ["ItemSampler", "DistributedItemSampler", "minibatcher_default"]


def minibatcher_default(batch, names):
    """Default minibatcher which maps a list of items to a `MiniBatch` with the
    same names as the items. The names of items are supposed to be provided
    and align with the data attributes of `MiniBatch`. If any unknown item name
    is provided, exception will be raised. If the names of items are not
    provided, the item list is returned as is and a warning will be raised.

    Parameters
    ----------
    batch : list
        List of items.
    names : Tuple[str] or None
        Names of items in `batch` with same length. The order should align
        with `batch`.

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
        # Handle the case of single item: batch = tensor([0, 1, 2, 3]), names =
        # ("seed_nodes",) as `zip(batch, names)` will iterate over the tensor
        # instead of the batch.
        init_data = {names[0]: batch}
    else:
        if isinstance(batch, Mapping):
            init_data = {
                name: {k: v[i] for k, v in batch.items()}
                for i, name in enumerate(names)
            }
        else:
            init_data = {name: item for item, name in zip(batch, names)}
    minibatch = MiniBatch()
    for name, item in init_data.items():
        if not hasattr(minibatch, name):
            dgl_warning(
                f"Unknown item name '{name}' is detected and added into "
                "`MiniBatch`. You probably need to provide a customized "
                "`MiniBatcher`."
            )
        if name == "node_pairs":
            # `node_pairs` is passed as a tensor in shape of `(N, 2)` and
            # should be converted to a tuple of `(src, dst)`.
            if isinstance(item, Mapping):
                item = {key: (item[key][:, 0], item[key][:, 1]) for key in item}
            else:
                item = (item[:, 0], item[:, 1])
        setattr(minibatch, name, item)
    return minibatch


class ItemSampler(IterDataPipe):
    """A sampler to iterate over input items and create subsets.

    Input items could be node IDs, node pairs with or without labels, node
    pairs with negative sources/destinations, DGLGraphs and heterogeneous
    counterparts.

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
    >>> item_set = gb.ItemSet(torch.arange(0, 10), names="seed_nodes")
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=False, drop_last=False
    ... )
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=tensor([0, 1, 2, 3]), node_pairs=None, labels=None,
        negative_srcs=None, negative_dsts=None, sampled_subgraphs=None,
        input_nodes=None, node_features=None, edge_features=None,
        compacted_node_pairs=None, compacted_negative_srcs=None,
        compacted_negative_dsts=None)

    2. Node pairs.
    >>> item_set = gb.ItemSet(torch.arange(0, 20).reshape(-1, 2),
    ...     names="node_pairs")
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=False, drop_last=False
    ... )
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=None,
        node_pairs=(tensor([0, 2, 4, 6]), tensor([1, 3, 5, 7])),
        labels=None, negative_srcs=None, negative_dsts=None,
        sampled_subgraphs=None, input_nodes=None, node_features=None,
        edge_features=None, compacted_node_pairs=None,
        compacted_negative_srcs=None, compacted_negative_dsts=None)

    3. Node pairs and labels.
    >>> item_set = gb.ItemSet(
    ...     (torch.arange(0, 20).reshape(-1, 2), torch.arange(10, 20)),
    ...     names=("node_pairs", "labels")
    ... )
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=False, drop_last=False
    ... )
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=None,
        node_pairs=(tensor([0, 2, 4, 6]), tensor([1, 3, 5, 7])),
        labels=tensor([10, 11, 12, 13]), negative_srcs=None,
        negative_dsts=None, sampled_subgraphs=None, input_nodes=None,
        node_features=None, edge_features=None, compacted_node_pairs=None,
        compacted_negative_srcs=None, compacted_negative_dsts=None)

    4. Node pairs and negative destinations.
    >>> node_pairs = torch.arange(0, 20).reshape(-1, 2)
    >>> negative_dsts = torch.arange(10, 30).reshape(-1, 2)
    >>> item_set = gb.ItemSet((node_pairs, negative_dsts), names=("node_pairs",
    ...     "negative_dsts"))
    >>> item_sampler = gb.ItemSampler(
    ...     item_set, batch_size=4, shuffle=False, drop_last=False
    ... )
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=None,
        node_pairs=(tensor([0, 2, 4, 6]), tensor([1, 3, 5, 7])),
        labels=None, negative_srcs=None,
        negative_dsts=tensor([[10, 11],
        [12, 13],
        [14, 15],
        [16, 17]]), sampled_subgraphs=None, input_nodes=None,
        node_features=None, edge_features=None, compacted_node_pairs=None,
        compacted_negative_srcs=None, compacted_negative_dsts=None)

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
    ...     "user": gb.ItemSet(torch.arange(0, 5), names="seed_nodes"),
    ...     "item": gb.ItemSet(torch.arange(0, 6), names="seed_nodes"),
    ... }
    >>> item_set = gb.ItemSetDict(ids)
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=4)
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes={'user': tensor([0, 1, 2, 3])}, node_pairs=None,
    labels=None, negative_srcs=None, negative_dsts=None, sampled_subgraphs=None,
    input_nodes=None, node_features=None, edge_features=None,
    compacted_node_pairs=None, compacted_negative_srcs=None,
    compacted_negative_dsts=None)

    8. Heterogeneous node pairs.
    >>> node_pairs_like = torch.arange(0, 10).reshape(-1, 2)
    >>> node_pairs_follow = torch.arange(10, 20).reshape(-1, 2)
    >>> item_set = gb.ItemSetDict({
    ...     "user:like:item": gb.ItemSet(
    ...         node_pairs_like, names="node_pairs"),
    ...     "user:follow:user": gb.ItemSet(
    ...         node_pairs_follow, names="node_pairs"),
    ... })
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=4)
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=None,
        node_pairs={'user:like:item':
            (tensor([0, 2, 4, 6]), tensor([1, 3, 5, 7]))},
        labels=None, negative_srcs=None, negative_dsts=None,
        sampled_subgraphs=None, input_nodes=None, node_features=None,
        edge_features=None, compacted_node_pairs=None,
        compacted_negative_srcs=None, compacted_negative_dsts=None)

    9. Heterogeneous node pairs and labels.
    >>> node_pairs_like = torch.arange(0, 10).reshape(-1, 2)
    >>> labels_like = torch.arange(0, 10)
    >>> node_pairs_follow = torch.arange(10, 20).reshape(-1, 2)
    >>> labels_follow = torch.arange(10, 20)
    >>> item_set = gb.ItemSetDict({
    ...     "user:like:item": gb.ItemSet((node_pairs_like, labels_like),
    ...         names=("node_pairs", "labels")),
    ...     "user:follow:user": gb.ItemSet((node_pairs_follow, labels_follow),
    ...         names=("node_pairs", "labels")),
    ... })
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=4)
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=None,
        node_pairs={'user:like:item':
            (tensor([0, 2, 4, 6]), tensor([1, 3, 5, 7]))},
        labels={'user:like:item': tensor([0, 1, 2, 3])},
        negative_srcs=None, negative_dsts=None, sampled_subgraphs=None,
        input_nodes=None, node_features=None, edge_features=None,
        compacted_node_pairs=None, compacted_negative_srcs=None,
        compacted_negative_dsts=None)

    10. Heterogeneous node pairs and negative destinations.
    >>> node_pairs_like = torch.arange(0, 10).reshape(-1, 2)
    >>> negative_dsts_like = torch.arange(10, 20).reshape(-1, 2)
    >>> node_pairs_follow = torch.arange(20, 30).reshape(-1, 2)
    >>> negative_dsts_follow = torch.arange(30, 40).reshape(-1, 2)
    >>> item_set = gb.ItemSetDict({
    ...     "user:like:item": gb.ItemSet((node_pairs_like, negative_dsts_like),
    ...         names=("node_pairs", "negative_dsts")),
    ...     "user:follow:user": gb.ItemSet((node_pairs_follow,
    ...         negative_dsts_follow), names=("node_pairs", "negative_dsts")),
    ... })
    >>> item_sampler = gb.ItemSampler(item_set, batch_size=4)
    >>> next(iter(item_sampler))
    MiniBatch(seed_nodes=None,
        node_pairs={'user:like:item':
            (tensor([0, 2, 4, 6]), tensor([1, 3, 5, 7]))},
        labels=None, negative_srcs=None,
        negative_dsts={'user:like:item': tensor([[10, 11],
        [12, 13],
        [14, 15],
        [16, 17]])}, sampled_subgraphs=None, input_nodes=None,
        node_features=None, edge_features=None, compacted_node_pairs=None,
        compacted_negative_srcs=None, compacted_negative_dsts=None)
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
        self._names = item_set.names
        self._item_set = IterableWrapper(item_set)
        self._batch_size = batch_size
        self._minibatcher = minibatcher
        self._drop_last = drop_last
        self._shuffle = shuffle

    def _organize_items(self, data_pipe) -> None:
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

        return data_pipe

    @staticmethod
    def _collate(batch):
        """Collate items into a batch. For internal use only."""
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

    def __iter__(self) -> Iterator:
        # Organize items.
        data_pipe = self._organize_items(self._item_set)

        # Collate.
        data_pipe = data_pipe.collate(collate_fn=self._collate)

        # Map to minibatch.
        data_pipe = data_pipe.map(partial(self._minibatcher, names=self._names))

        return iter(data_pipe)


class DistributedItemSampler(ItemSampler):
    """A sampler to iterate over input items and create subsets distributedly.

    This sampler creates a distributed subset of items from the given data set,
    which can be used for training with PyTorch's Distributed Data Parallel
    (DDP). The items can be node IDs, node pairs with or without labels, node
    pairs with negative sources/destinations, DGLGraphs, or heterogeneous
    counterparts. The original item set is sharded such that each replica
    (process) receives an exclusive subset.

    Note: DistributedItemSampler may not work as expected when it is the last
    datapipe before the data is fetched. Please wrap a SingleProcessDataLoader
    or another datapipe on it.

    Note: The items will be first sharded onto each replica, then get shuffled
    (if needed) and batched. Therefore, each replica will always get a same set
    of items.

    Note: This class `DistributedItemSampler` is not decorated with
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
    num_replicas: int
        The number of model replicas that will be created during Distributed
        Data Parallel (DDP) training. It should be the same as the real world
        size, otherwise it could cause errors. By default, it is retrieved from
        the current distributed group.
    drop_uneven_inputs : bool
        Option to make sure the numbers of batches for each replica are the
        same. If some of the replicas have more batches than the others, the
        redundant batches of those replicas will be dropped. If the drop_last
        parameter is also set to True, the last batch will be dropped before the
        redundant batches are dropped.
        Note: When using Distributed Data Parallel (DDP) training, the program
        may hang or error if the a replica has fewer inputs. It is recommended
        to use the Join Context Manager provided by PyTorch to solve this
        problem. Please refer to
        https://pytorch.org/tutorials/advanced/generic_join.html. However, this
        option can be used if the Join Context Manager is not helpful for any
        reason.

    Examples
    --------
    0. Preparation: DistributedItemSampler needs multi-processing environment to
    work. You need to spawn subprocesses and initialize processing group before
    executing following examples. Due to randomness, the output is not always
    the same as listed below.

    >>> import torch
    >>> from dgl import graphbolt as gb
    >>> item_set = gb.ItemSet(torch.arange(0, 13))
    >>> num_replicas = 4
    >>> batch_size = 2
    >>> mp.spawn(...)

    1. shuffle = False, drop_last = False, drop_uneven_inputs = False.

    >>> item_sampler = gb.DistributedItemSampler(
    >>>     item_set, batch_size=2, shuffle=False, drop_last=False,
    >>>     drop_uneven_inputs=False
    >>> )
    >>> data_loader = gb.SingleProcessDataLoader(item_sampler)
    >>> print(f"Replica#{proc_id}: {list(data_loader)})
    Replica#0: [tensor([0, 4]), tensor([ 8, 12])]
    Replica#1: [tensor([1, 5]), tensor([ 9, 13])]
    Replica#2: [tensor([2, 6]), tensor([10])]
    Replica#3: [tensor([3, 7]), tensor([11])]

    2. shuffle = False, drop_last = True, drop_uneven_inputs = False.

    >>> item_sampler = gb.DistributedItemSampler(
    >>>     item_set, batch_size=2, shuffle=False, drop_last=True,
    >>>     drop_uneven_inputs=False
    >>> )
    >>> data_loader = gb.SingleProcessDataLoader(item_sampler)
    >>> print(f"Replica#{proc_id}: {list(data_loader)})
    Replica#0: [tensor([0, 4]), tensor([ 8, 12])]
    Replica#1: [tensor([1, 5]), tensor([ 9, 13])]
    Replica#2: [tensor([2, 6])]
    Replica#3: [tensor([3, 7])]

    3. shuffle = False, drop_last = False, drop_uneven_inputs = True.

    >>> item_sampler = gb.DistributedItemSampler(
    >>>     item_set, batch_size=2, shuffle=False, drop_last=False,
    >>>     drop_uneven_inputs=True
    >>> )
    >>> data_loader = gb.SingleProcessDataLoader(item_sampler)
    >>> print(f"Replica#{proc_id}: {list(data_loader)})
    Replica#0: [tensor([0, 4]), tensor([ 8, 12])]
    Replica#1: [tensor([1, 5]), tensor([ 9, 13])]
    Replica#2: [tensor([2, 6]), tensor([10])]
    Replica#3: [tensor([3, 7]), tensor([11])]

    4. shuffle = False, drop_last = True, drop_uneven_inputs = True.

    >>> item_sampler = gb.DistributedItemSampler(
    >>>     item_set, batch_size=2, shuffle=False, drop_last=True,
    >>>     drop_uneven_inputs=True
    >>> )
    >>> data_loader = gb.SingleProcessDataLoader(item_sampler)
    >>> print(f"Replica#{proc_id}: {list(data_loader)})
    Replica#0: [tensor([0, 4])]
    Replica#1: [tensor([1, 5])]
    Replica#2: [tensor([2, 6])]
    Replica#3: [tensor([3, 7])]

    5. shuffle = True, drop_last = True, drop_uneven_inputs = False.

    >>> item_sampler = gb.DistributedItemSampler(
    >>>     item_set, batch_size=2, shuffle=True, drop_last=True,
    >>>     drop_uneven_inputs=False
    >>> )
    >>> data_loader = gb.SingleProcessDataLoader(item_sampler)
    >>> print(f"Replica#{proc_id}: {list(data_loader)})
    (One possible output:)
    Replica#0: [tensor([0, 8]), tensor([ 4, 12])]
    Replica#1: [tensor([ 5, 13]), tensor([9, 1])]
    Replica#2: [tensor([ 2, 10])]
    Replica#3: [tensor([11,  7])]

    6. shuffle = True, drop_last = True, drop_uneven_inputs = True.

    >>> item_sampler = gb.DistributedItemSampler(
    >>>     item_set, batch_size=2, shuffle=True, drop_last=True,
    >>>     drop_uneven_inputs=True
    >>> )
    >>> data_loader = gb.SingleProcessDataLoader(item_sampler)
    >>> print(f"Replica#{proc_id}: {list(data_loader)})
    (One possible output:)
    Replica#0: [tensor([8, 0])]
    Replica#1: [tensor([ 1, 13])]
    Replica#2: [tensor([10,  6])]
    Replica#3: [tensor([ 3, 11])]
    """

    def __init__(
        self,
        item_set: ItemSet or ItemSetDict,
        batch_size: int,
        minibatcher: Optional[Callable] = minibatcher_default,
        drop_last: Optional[bool] = False,
        shuffle: Optional[bool] = False,
        num_replicas: Optional[int] = None,
        drop_uneven_inputs: Optional[bool] = False,
    ) -> None:
        super().__init__(item_set, batch_size, minibatcher, drop_last, shuffle)
        self._drop_uneven_inputs = drop_uneven_inputs
        # Apply a sharding filter to distribute the items.
        self._item_set = self._item_set.sharding_filter()
        # Get world size.
        if num_replicas is None:
            assert (
                dist.is_available()
            ), "Requires distributed package to be available."
            num_replicas = dist.get_world_size()
        if self._drop_uneven_inputs:
            # If the len() method of the item_set is not available, it will
            # throw an exception.
            total_len = len(item_set)
            # Calculate the number of batches after dropping uneven batches for
            # each replica.
            self._num_evened_batches = total_len // (
                num_replicas * batch_size
            ) + (
                (not drop_last)
                and (total_len % (num_replicas * batch_size) >= num_replicas)
            )

    def _organize_items(self, data_pipe) -> None:
        data_pipe = super()._organize_items(data_pipe)

        # If drop_uneven_inputs is True, drop the excessive inputs by limiting
        # the length of the datapipe.
        if self._drop_uneven_inputs:
            data_pipe = data_pipe.header(self._num_evened_batches)

        return data_pipe
