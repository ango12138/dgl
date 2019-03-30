import os
import time
import scipy
from xmlrpc.server import SimpleXMLRPCServer
import xmlrpc.client

from collections.abc import MutableMapping

from ..base import ALL, is_all, DGLError
from .. import backend as F
from ..graph import DGLGraph
from ..graph_index import GraphIndex, create_graph_index
from .._ffi.function import _init_api
from .. import ndarray as nd

def _get_ndata_path(graph_name, ndata_name):
    return "/" + graph_name + "_node_" + ndata_name

def _get_edata_path(graph_name, edata_name):
    return "/" + graph_name + "_edge_" + edata_name

def _get_graph_path(graph_name):
    return "/" + graph_name


class NodeDataView(MutableMapping):
    """The data view class when G.nodes[...].data is called.

    See Also
    --------
    dgl.DGLGraph.nodes
    """
    __slots__ = ['_graph', '_nodes', '_graph_name']

    def __init__(self, graph, nodes, graph_name):
        self._graph = graph
        self._nodes = nodes
        self._graph_name = graph_name

    def __getitem__(self, key):
        return self._graph.get_n_repr(self._nodes)[key]

    def __setitem__(self, key, val):
        # Move the data in val to shared memory.
        dlpack = F.zerocopy_to_dlpack(val)
        dgl_tensor = nd.from_dlpack(dlpack)
        val = _CAPI_DGLCreateSharedMemWithData(_get_ndata_path(self._graph_name, key),
                                                dgl_tensor)
        self._graph.set_n_repr({key : val}, self._nodes)

    def __delitem__(self, key):
        if not is_all(self._nodes):
            raise DGLError('Delete feature data is not supported on only a subset'
                           ' of nodes. Please use `del G.ndata[key]` instead.')
        self._graph.pop_n_repr(key)

    def __len__(self):
        return len(self._graph._node_frame)

    def __iter__(self):
        return iter(self._graph._node_frame)

    def __repr__(self):
        data = self._graph.get_n_repr(self._nodes)
        return repr({key : data[key] for key in self._graph._node_frame})

class EdgeDataView(MutableMapping):
    """The data view class when G.edges[...].data is called.

    See Also
    --------
    dgl.DGLGraph.edges
    """
    __slots__ = ['_graph', '_edges', '_graph_name']

    def __init__(self, graph, edges, graph_name):
        self._graph = graph
        self._edges = edges
        self._graph_name = graph_name

    def __getitem__(self, key):
        return self._graph.get_e_repr(self._edges)[key]

    def __setitem__(self, key, val):
        # Move the data in val to shared memory.
        dlpack = F.zerocopy_to_dlpack(val)
        dgl_tensor = nd.from_dlpack(dlpack)
        val = _CAPI_DGLCreateSharedMemWithData(_get_edata_path(self._graph_name, key),
                                                dgl_tensor)
        self._graph.set_e_repr({key : val}, self._edges)

    def __delitem__(self, key):
        if not is_all(self._edges):
            raise DGLError('Delete feature data is not supported on only a subset'
                           ' of nodes. Please use `del G.edata[key]` instead.')
        self._graph.pop_e_repr(key)

    def __len__(self):
        return len(self._graph._edge_frame)

    def __iter__(self):
        return iter(self._graph._edge_frame)

    def __repr__(self):
        data = self._graph.get_e_repr(self._edges)
        return repr({key : data[key] for key in self._graph._edge_frame})

def _to_csr(graph_data, edge_dir, multigraph):
    try:
        indptr = graph_data.indptr
        indices = graph_data.indices
        return indptr, indices
    except:
        if isinstance(graph_data, scipy.sparse.spmatrix):
            csr = graph_data.tocsr()
            return csr.indptr, csr.indices
        else:
            idx = create_graph_index(graph_data=graph_data, multigraph=multigraph, readonly=True)
            transpose = False if edge_dir == 'in' else True
            csr = idx.adjacency_matrix_scipy(transpose, 'csr')
            return csr.indptr, csr.indices

class SharedMemoryStoreServer:
    def __init__(self, graph_data, edge_dir, graph_name, multigraph, num_workers, port):
        graph_idx = GraphIndex(multigraph=multigraph, readonly=True)
        indptr, indices = _to_csr(graph_data, edge_dir, multigraph)
        graph_idx.from_csr_matrix(indptr, indices, edge_dir, _get_graph_path(graph_name))

        self._graph = DGLGraph(graph_idx, multigraph=multigraph, readonly=False)
        self._num_workers = num_workers
        self._graph_name = graph_name
        self._edge_dir = edge_dir

        def get_graph_info():
            return self._graph.number_of_nodes(), self._graph.number_of_edges(), \
                    self._graph.is_multigraph, edge_dir

        def init_ndata(ndata_name, num_feats, dtype):
            if ndata_name in self._graph.ndata:
                ndata = self._graph.ndata[ndata_name]
                assert ndata.shape[1] == num_feats
                return 0

            data = _CAPI_DGLCreateSharedMem(_get_ndata_path(graph_name, ndata_name),
                                            self._graph.number_of_nodes(),
                                            num_feats, dtype, "zero", True)
            dlpack = data.to_dlpack()
            self._graph.ndata[ndata_name] = F.zerocopy_from_dlpack(dlpack)
            return 0

        def list_ndata():
            ndata = self._graph.ndata
            return [[key, ndata[key].shape[1:], F.dtype(ndata[key])] for key in ndata]

        def list_edata():
            edata = self._graph.edata
            return [[key, edata[key].shape[1:], F.dtype(edata[key])] for key in edata]

        def terminate():
            self._num_workers -= 1
            return 0

        self.server = SimpleXMLRPCServer(("localhost", port))
        self.server.register_function(get_graph_info, "get_graph_info")
        self.server.register_function(init_ndata, "init_ndata")
        self.server.register_function(terminate, "terminate")
        self.server.register_function(list_ndata, "list_ndata")
        self.server.register_function(list_edata, "list_edata")

    def __del__(self):
        self._graph = None

    @property
    def ndata(self):
        """Return the data view of all the nodes.

        DGLGraph.ndata is an abbreviation of DGLGraph.nodes[:].data

        See Also
        --------
        dgl.DGLGraph.nodes
        """
        return NodeDataView(self._graph, ALL, self._graph_name)

    @property
    def edata(self):
        """Return the data view of all the edges.

        DGLGraph.data is an abbreviation of DGLGraph.edges[:].data

        See Also
        --------
        dgl.DGLGraph.edges
        """
        return EdgeDataView(self._graph, ALL, self._graph_name)

    def run(self):
        while self._num_workers > 0:
            self.server.handle_request()
        self._graph = None

class SharedMemoryGraphStore(DGLGraph):
    def __init__(self, graph_name, port):
        self._graph_name = graph_name
        self._pid = os.getpid()
        self.proxy = xmlrpc.client.ServerProxy("http://localhost:" + str(port) + "/")
        num_nodes, num_edges, multigraph, edge_dir = self.proxy.get_graph_info()

        graph_idx = GraphIndex(multigraph=multigraph, readonly=True)
        graph_idx.from_shared_mem_csr_matrix(_get_graph_path(graph_name), num_nodes, num_edges, edge_dir)
        super(SharedMemoryGraphStore, self).__init__(graph_idx, multigraph=multigraph, readonly=True)

        # init all ndata and edata.
        ndata_infos = self.proxy.list_ndata()
        for name, shape, dtype in ndata_infos:
            self.init_ndata(name, shape[0], dtype)

        edata_infos = self.proxy.list_edata()
        for name, shape, dtype in edata_infos:
            self.init_edata(name, shape[0], dtype)

    def __del__(self):
        if self.proxy is not None:
            self.proxy.terminate()

    def init_ndata(self, ndata_name, num_feats, dtype):
        self.proxy.init_ndata(ndata_name, num_feats, dtype)
        data = _CAPI_DGLCreateSharedMem(_get_ndata_path(self._graph_name, ndata_name),
                                        self.number_of_nodes(),
                                        num_feats, dtype, "zero", False)
        dlpack = data.to_dlpack()
        self.ndata[ndata_name] = F.zerocopy_from_dlpack(dlpack)

    def destroy(self):
        if self.proxy is not None:
            self.proxy.terminate()
        self.proxy = None


def create_graph_store_server(graph_data, edge_dir, graph_name, store_type,
                              multigraph, num_workers, port=8000):
    return SharedMemoryStoreServer(graph_data, edge_dir, graph_name, multigraph,
                                   num_workers, port)

def create_graph_store_client(graph_name, store_type, port=8000):
    return SharedMemoryGraphStore(graph_name, port)


_init_api("dgl.contrib.graph_store")
