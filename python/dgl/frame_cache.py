"""Cache for frames in DGLGraph."""
import numpy as np

from . import backend as F
from . import utils
from .frame import Frame, FrameRef
from .graph_index import search_nids

class FrameRowCache:
    def __init__(self, frame, ids, ctx):
        self._cached_ids = ids
        self._frame = frame
        ids = ids.tousertensor()
        cols = {}
        for key in frame:
            col = frame[key]
            cols.update({key: F.copy_to(col[ids], ctx)})
        self._cache = FrameRef(Frame(cols))
        self._ctx = ctx

    @property
    def context(self):
        return self._ctx

    def cache_lookup(self, ids):
        orig_ids = [i.tousertensor() for i in ids]
        sizes = [len(i) for i in ids]
        offs = np.cumsum(sizes)
        ids = F.cat(orig_ids, 0)
        ids = utils.toindex(ids)
        lids = search_nids(self._cached_ids, ids)
        lids = lids.tonumpy()
        ids = ids.tonumpy()
        lids = np.split(lids, offs[0:(len(offs) - 1)])
        ids = np.split(ids, offs[0:(len(offs) - 1)])

        ret = []
        for lid, id in zip(lids, ids):
            cached_out_idx = np.nonzero(lid != -1)[0]
            cache_idx = lid[cached_out_idx]
            uncached_out_idx = np.nonzero(lid == -1)[0]
            global_uncached_ids = id[uncached_out_idx]
            ret.append(SubgraphFrameCache(self._frame, self._cache, self._ctx,
                                          cached_out_idx, cache_idx,
                                          uncached_out_idx, global_uncached_ids))
        return ret

class SubgraphFrameCache:
    def __init__(self, frame, cache, ctx, cached_out_idx, cache_idx,
                 uncached_out_idx, global_uncached_ids):
        self._frame = frame
        self._cache = cache
        self._ctx = ctx
        # The index where cached data should be written to.
        self._cached_out_idx = cached_out_idx
        # The index where cached data should be read from the cache.
        self._cache_idx = cache_idx
        # The index of uncached data. It'll be read from the global frame.
        self._global_uncached_ids = global_uncached_ids
        # The index of uncached data should be written to.
        self._uncached_out_idx = uncached_out_idx

    @property
    def context(self):
        return self._ctx

    def merge(self):
        ret = {}
        for key in self._cache:
            col = self._cache[key]
            shape = (len(self._cache_idx) + len(self._global_uncached_ids),) + col.shape[1:]
            data = F.empty(shape=shape, dtype=col.dtype, ctx=self._ctx)
            # fill cached data.
            data[self._cached_out_idx] = col[self._cache_idx]
            # fill uncached data
            col = self._frame[key]
            data[self._uncached_out_idx] = F.copy_to(col[self._global_uncached_ids], self._ctx)
            ret.update({key: data})
        return ret
