"""Node embedding optimizers for distributed training"""
import abc
from abc import abstractmethod
import torch as th

from ...dist_tensor import DistTensor
from ...sparse_emb import NodeEmbedding
from .utils import alltoallv_cpu, alltoall_cpu

class DistSparseGradOptimizer(abc.ABC):
    r''' The abstract dist sparse optimizer.

    Note: dgl dist sparse optimizer only work with dgl.distributed.NodeEmbedding

    Parameters
    ----------
    params : list of NodeEmbedding
        The list of NodeEmbedding.
    lr : float
        The learning rate.
    '''
    def __init__(self, params, lr):
        self._params = params
        self._lr = lr
        self._rank = None
        self._world_size = None
        self._shared_cache = {}
        self._clean_grad = False
        self._opt_meta = {}

        if th.distributed.is_initialized():
            self._rank = th.distributed.get_rank()
            self._world_size = th.distributed.get_world_size()
        else:
            assert 'th.distributed shoud be initialized'

    def step(self):
        ''' The step function.

        The step function is invoked at the end of every batch to push the gradients
        of the embeddings involved in a mini-batch to DGL's servers and update the embeddings.
        '''
        with th.no_grad():
            local_indics = {emb.name: [] for emb in self._params}
            local_grads = {emb.name: [] for emb in self._params}
            device = th.device('cpu')
            for emb in self._params:
                name = emb._tensor.name
                kvstore = emb._tensor.kvstore
                trace = emb._trace
                trainers_per_server = self._world_size // kvstore.num_servers

                idics = [t[0] for t in trace]
                grads = [t[1].grad.data for t in trace]
                idics = th.cat(idics, dim=0)
                grads = th.cat(grads, dim=0)
                device = grads.device

                # will send grad to each corresponding trainer
                if self._world_size > 1:
                    # get idx split from kvstore
                    idx_split = kvstore.get_partid(name, idics)
                    idx_split_size = []
                    idics_list = []
                    grad_list = []
                    # split idx and grad first
                    for i in range(kvstore.num_servers):
                        mask = idx_split == i
                        idx_i = idics[mask]
                        grad_i = grads[mask]

                        if trainers_per_server <= 1:
                            idx_split_size.append(th.tensor([idx_i.shape[0]], dtype=th.int64))
                            idics_list.append(idx_i)
                            grad_list.append(grad_i)
                        else:
                            kv_idx_split = th.remainder(idx_i, trainers_per_server).long()
                            for j in range(trainers_per_server):
                                mask = kv_idx_split == j
                                idx_j = idx_i[mask]
                                grad_j = grad_i[mask]
                                idx_split_size.append(th.tensor([idx_j.shape[0]], dtype=th.int64))
                                idics_list.append(idx_j)
                                grad_list.append(grad_j)

                    # if one machine launch multiple KVServer, they share the same storage.
                    # For each machine, the pytorch rank is num_trainers * machine_id + i

                    # use scatter to sync across trainers about the p2p tensor size
                    # Note: If we have GPU nccl support, we can use all_to_all to
                    # sync information here
                    gather_list = list(th.empty([self._world_size],
                                                dtype=th.int64).chunk(self._world_size))
                    alltoall_cpu(self._rank, self._world_size, gather_list, idx_split_size)
                    # use cpu until we have GPU alltoallv
                    idx_gather_list = [th.empty((int(num_emb),),
                                                dtype=idics.dtype) for num_emb in gather_list]
                    alltoallv_cpu(self._rank, self._world_size, idx_gather_list, idics_list)
                    local_indics[name] = idx_gather_list
                    grad_gather_list = [th.empty((int(num_emb), grads.shape[1]),
                                                 dtype=grads.dtype) for num_emb in gather_list]
                    alltoallv_cpu(self._rank, self._world_size, grad_gather_list, grad_list)
                    local_grads[name] = grad_gather_list
                else:
                    local_indics[name] = [idics]
                    local_grads[name] = [grads]

            if self._clean_grad:
                # clean gradient track
                for emb in self._params:
                    emb.reset_trace()
                self._clean_grad = False

            # do local update
            for emb in self._params:
                name = emb._tensor.name

                idx = th.cat(local_indics[name], dim=0)
                grad = th.cat(local_grads[name], dim=0)
                self.update(idx.to(device, non_blocking=True),
                            grad.to(device, non_blocking=True), emb)

        # synchronized gradient update
        if self._world_size > 1:
            th.distributed.barrier()

    @abstractmethod
    def update(self, idx, grad, emb):
        """ Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. we maintains gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        idx : tensor
            Index of the embeddings to be updated.
        grad : tensor
            Gradient of each embedding.
        emb : dgl.distributed.NodeEmbedding
            Sparse node embedding to update.
        """

    def zero_grad(self):
        """clean grad cache
        """
        self._clean_grad = True

def initializer(shape, dtype):
    """ Sparse optimizer state initializer

    Parameters
    ----------
    shape : tuple of ints
        The shape of the state tensor
    dtype : torch dtype
        The data type of the state tensor
    """
    arr = th.zeros(shape, dtype=dtype)
    return arr

class SparseAdagrad(DistSparseGradOptimizer):
    r''' Distributed Node embedding optimizer using the Adagrad algorithm.

    This optimizer implements a distributed sparse version of Adagrad algorithm for
    optimizing :class:`dgl.distributed.NodeEmbedding`. Being sparse means it only updates
    the embeddings whose gradients have updates, which are usually a very
    small portion of the total embeddings.

    Adagrad maintains a :math:`G_{t,i,j}` for every parameter in the embeddings, where
    :math:`G_{t,i,j}=G_{t-1,i,j} + g_{t,i,j}^2` and :math:`g_{t,i,j}` is the gradient of
    the dimension :math:`j` of embedding :math:`i` at step :math:`t`.

    NOTE: The support of sparse Adagrad optimizer is experimental.

    Parameters
    ----------
    params : list[dgl.distributed.NodeEmbedding]
        The list of dgl.distributed.NodeEmbedding.
    lr : float
        The learning rate.
    eps : float, Optional
        The term added to the denominator to improve numerical stability
        Default: 1e-10
    '''
    def __init__(self, params, lr, eps=1e-10):
        super(SparseAdagrad, self).__init__(params, lr)
        self._eps = eps
        # We need to register a state sum for each embedding in the kvstore.
        self._state = {}
        for emb in params:
            assert isinstance(emb, NodeEmbedding), \
                'SparseAdagrad only supports dgl.distributed.NodeEmbedding'

            name = emb.name + "_sum"
            state = DistTensor((emb.num_embeddings, emb.embedding_dim), th.float32, name,
                               init_func=initializer, part_policy=emb.part_policy, is_gdata=False)
            emb.set_optm_state(state)
            self._state[emb.name] = state

    def update(self, idx, grad, emb):
        """ Update embeddings in a sparse manner
        Sparse embeddings are updated in mini batches. we maintains gradient states for
        each embedding so they can be updated separately.

        Parameters
        ----------
        idx : tensor
            Index of the embeddings to be updated.
        grad : tensor
            Gradient of each embedding.
        emb : dgl.nn.NodeEmbedding
            Sparse embedding to update.
        """
        eps = self._eps
        clr = self._lr
        exec_dev = grad.device

        # the update is non-linear so indices must be unique
        grad_indices, inverse, cnt = th.unique(idx, return_inverse=True, return_counts=True)
        grad_values = th.zeros((grad_indices.shape[0], grad.shape[1]), device=exec_dev)
        grad_values.index_add_(0, inverse, grad)
        grad_values = grad_values / cnt.unsqueeze(1)
        grad_sum = (grad_values * grad_values)

        # update grad state
        grad_state = self._state[emb.name][grad_indices].to(exec_dev, non_blocking=True)
        grad_state += grad_sum
        self._state[emb.name][grad_indices] = grad_state.to(th.device('cpu'), non_blocking=True)

        # update emb
        std_values = grad_state.add_(eps).sqrt_()
        tmp = clr * grad_values / std_values
        emb._tensor[grad_indices] -= tmp.to(th.device('cpu'), non_blocking=True)
