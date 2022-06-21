##
#   Copyright 2021-2022 Contributors
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
""" Methods for placing graphs into pinned/uva memory.
"""

from ..heterograph import DGLHeteroGraph
from .. import backend as F
import itertools

__all__ = ['pin_graph_for_uva']


class _PinnedGraph(DGLHeteroGraph):
    def __init__(self, g, device):
        super().__init__(gidx=g._graph, ntypes=g.ntypes, etypes=g.etypes, \
            node_frames=g._node_frames, edge_frames=g._edge_frames)
        assert F.device_type(super().device) == 'cpu', "Only graphs on the " \
            "cpu can be pinned. Got {}".format(F.device_type(super().device))
        assert F.device_type(device) == 'cuda', "Target device for UVA " \
            "access must be a CUDA device. Got {}".format(F.device_type(device))

        # default to the original device until we've pinned everything
        self._device = g.device
        self.create_formats_()
        self.pin_structure_()

        for frame in itertools.chain(self._node_frames, self._edge_frames):
            for col in frame._columns.values():
                col.pin_memory_()

        # setting the device must be done last
        self._device = device

    def _close(self):
        self.unpin_structure_()
        # because backend tensors aren't automatically unpinned,
        # unpin them here instead of heterograph to ensure
        # we don't leak resources
        for frame in itertools.chain(self._node_frames, self._edge_frames):
            for col in frame._columns.values():
                col.unpin_memory_()

    @property
    def device(self):
        return self._device

    def __getattr__(self, attr):
        return getattr(self._hg, attr)


class _PinnedGraphContext():
    def __init__(self, g, device):
        self._g = g
        self._device = device
        self._handle = None

    def __enter__(self):
        # only pin graph here to ensure it will always be unpinned
        self._handle = _PinnedGraph(
            self._g,
            self._device)
        return self._handle

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self._handle._close()


def pin_graph_for_uva(g, device):
    """ Pin a graph 'g' for access from GPU 'device'.
    """
    return _PinnedGraphContext(g, device)



