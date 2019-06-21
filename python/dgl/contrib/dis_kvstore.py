# This file contains DGL distributed kvstore APIs.
from ..network import _create_sender, _create_receiver
from ..network import _finalize_sender, _finalize_receiver
from ..network import _network_wait, _add_receiver_addr
from ..network import _receiver_wait, _sender_connect
from ..network import _send_kv_msg, _recv_kv_msg
from ..network import KVMsgType, KVStoreMsg

import math
import torch

class KVServer(object):
    """KVServer is a lightweight key-value store service for DGL distributed training.

    In practice, developers can use KVServer to hold large-scale graph features or 
    graph embeddings across machines or storing them in one standalone machine with big memory 
    capability. DGL KVServer uses a very simple range-partition scheme to partition data 
    into different KVServer nodes. For example, if the total embedding size is 200 and we have 
    two KVServer nodes, the data (0~99) will be stored in kvserver_0, and the data (100~199) will 
    be stored in kvserver_1.

    Parameters
    ----------
    server_id : int
        KVServer's ID (start from 0). 
    client_namebook : dict
        IP address namebook of KVClient, where the key is client's ID 
        (start from 0) and the value is client's IP address, e.g.,

            { 0:'168.12.23.45:50051', 
              1:'168.12.23.21:50051', 
              2:'168.12.46.12:50051' }
    server_addr : str
        IP address of current KVServer node, e.g., '168.12.23.22:50051'
    """
    def __init__(self, server_id, client_namebook, server_addr):
        assert server_id >= 0, 'server_id cannot be a negative number.'
        assert len(client_namebook) > 0, 'client_namebook cannot be empty.'
        assert len(server_addr.split(':')) == 2, 'Please use right IP format, e.g., 127.0.0.1:50051'
        # self._data_store is a key-value store 
        # where the key is data name and value is a tensor 
        # (mx.ndarray or torch.tensor) partitioned into current node.
        self._is_init = False
        self._data_store = {}
        self._server_id = server_id
        self._client_namebook = client_namebook
        self._client_count = len(client_namebook)
        self._addr = server_addr
        self._sender = _create_sender()
        self._receiver = _create_receiver()

    def __del__(self):
        """Finalize KVServer
        """
        _finalize_sender(self._sender)
        _finalize_receiver(self._receiver)

    def start(self):
        """Start the service of KVServer
        """
        server_ip, server_port = self._addr.split(':')
        _receiver_wait(self._receiver, server_ip, int(server_port), self._client_count)
        _network_wait() # wait client's setup
        for ID, addr in self._client_namebook.items():
            client_ip, client_port = addr.split(':')
            _add_receiver_addr(self._sender, client_ip, int(client_port), ID)
        _sender_connect(self._sender)
        # Service loop
        while True:
            msg = _recv_kv_msg(self._receiver)
            if msg.type == KVMsgType.INIT:
                if self._is_init == False:
                    self._init_data(msg.name, msg.id.tolist())
                    self._is_init = True
            elif msg.type == KVMsgType.PUSH:
                ID = self._remap_id(msg.name, msg.id)
                self._push_handler(msg.name, ID, msg.data)
            elif msg.type == KVMsgType.PULL:
                ID = self._remap_id(msg.name, msg.id)
                res_tensor = self._pull_handler(msg.name, ID)
                back_msg = KVStoreMsg(
                    type=KVMsgType.PULL_BACK,
                    rank=self._server_id,
                    name=msg.name,
                    id=msg.id,
                    data=res_tensor)
                _send_kv_msg(self._sender, back_msg, msg.rank)
            elif msg.type == KVMsgType.FINAL:
                print("Exit KVStore service, server ID: %d" % self.get_id())
                break
            else:
                raise RuntimeError('Unknown type of kvstore message: %d' % msg.type.value)

    def get_id(self):
        """Get server id

        Return
        ------
        int
            KVServer ID
        """
        return self._server_id

    def _init_data(self, name, shape):
        """User-defined initialize method.

        On default, we initialize all data to zero.

        Parameters
        ----------
        name : str
            data str
        shape : list
            local shape of target tensor
        """
        self._data_store[name] = torch.zeros(shape, dtype=torch.float32)

    def _push_handler(self, name, ID, data):
        """User-defined handler for PUSH message. 

        On default, _push_handler perform ADD() operation for the PUSH message.

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the IDs that has been re-mapped
        data : tensor (mx.ndarray or torch.tensor)
            a matrix with the same row size of id
        """
        size = ID.shape[0]
        for idx in range(size):
            self._data_store[name][ID[idx]] += data[idx]

    def _pull_handler(self, name, ID):
        """User-defined handler for PUSH operation.

        On default, _pull_handler perform index_select() operation for the PULL message.

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the IDs that has been re-mapped

        Return
        ------
        tensor
            a matrix with the same row size of ID
        """
        new_tensor = self._data_store[name].index_select(0, ID)
        return new_tensor

    def _remap_id(self, name, ID):
        """Re-mapping global-ID to local-ID.

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the global data ID

        Return
        ------
        tensor
            re-mapped ID
        """
        return ID % self._data_store[name].shape[0]

class KVClient(object):
    """KVClient is used to push/pull tensors to/from KVServer on DGL trainer.

    There are three operations supported by KVClient:

      * init_data(name, shape): initialize data for KVServer
      * push(name, id, data): push data to KVServer
      * pull(name, id): pull data from KVServer
      * shut_down(): shut down all KVServer nodes

    Parameters
    ----------
    client_id : int
        KVClient's ID (start from 0)
    server_namebook: dict
        IP address namebook of KVServer, where key is the KVServer's ID 
        (start from 0) and value is the server's IP address, e.g.,

        { 0:'168.12.23.45:50051', 
          1:'168.12.23.21:50051', 
          2:'168.12.46.12:50051' }
    client_addr : str
        IP address of current KVClient, e.g., '168.12.23.22:50051'
    """
    def __init__(self, client_id, server_namebook, client_addr):
        assert client_id >= 0, 'client_id cannot be a nagative number.'
        assert len(server_namebook) > 0, 'server_namebook cannot be empty.'
        assert len(client_addr.split(':')) == 2, 'Please use right IP format, e.g., 127.0.0.1:50051'
        # self._data_size is a key-value store where the key is data name 
        # and value is the size of tensor. It is used to partition data into
        # different KVServer nodes.
        self._data_size = {}
        self._client_id = client_id
        self._server_namebook = server_namebook
        self._server_count = len(server_namebook)
        self._addr = client_addr
        self._sender = _create_sender()
        self._receiver = _create_receiver()

    def __del__(self):
        """Finalize KVClient
        """
        _finalize_sender(self._sender)
        _finalize_receiver(self._receiver)

    def connect(self):
        """Connect to all KVServer nodes
        """
        for ID, addr in self._server_namebook.items():
            server_ip, server_port = addr.split(':')
            _add_receiver_addr(self._sender, server_ip, int(server_port), ID)
        _sender_connect(self._sender)
        client_ip, client_port = self._addr.split(':')
        _receiver_wait(self._receiver, client_ip, int(client_port), self._server_count)

    def init_data(self, name, shape):
        """Initialize data tensor on KVServer. 
        We usually invoke this API by just one client (e.g., client_0).

        init_data() will automatically partition shape into different KVServer nodes.

        Parameters
        ----------
        name : str
            data name
        shape : list
            global shape of tensor
        """
        self._data_size[name] = shape[0]
        count = math.ceil(shape[0] / self._server_count)
        for server_id in range(self._server_count):
            par_shape = shape.copy()
            if shape[0] - server_id*count >= count:
                par_shape[0] = count
            else:
                par_shape[0] = shape[0] - server_id*count
            tensor_shape = torch.tensor(par_shape)
            msg = KVStoreMsg(
                type=KVMsgType.INIT,
                rank=self._client_id,
                name=name,
                id=tensor_shape,
                data=None) 
            _send_kv_msg(self._sender, msg, server_id)
        
    def push(self, name, ID, data):
        """Push message to KVServer

        The push() API will partition message and mapping them into
        different KVServer nodes automatically.

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the global IDs
        data : tensor (mx.ndarray or torch.tensor)
            a matrix with the same row size of id
        """
        assert ID.dim() == 1, 'ID must be a vector.'
        assert data.dim() == 2, 'data must be a matrix.'
        assert data.size(0) == ID.size(0), 'The data must has the same row size with ID vector.'
        group_size = [0] * self._server_count
        for id in ID:
            # mapping data to corresponded server nodes
            server_id = self._get_server_id(id.item(), name)
            group_size[server_id] += 1
        min_idx = 0
        max_idx = 0
        for idx in range(self._server_count):
            if group_size[idx] == 0:
                continue
            max_idx += group_size[idx]
            range_id = ID[min_idx:max_idx]
            range_data = data[min_idx:max_idx]
            min_idx = max_idx
            msg = KVStoreMsg(
                type=KVMsgType.PUSH,
                rank=self._client_id,
                name=name,
                id=range_id,
                data=range_data)
            _send_kv_msg(self._sender, msg, idx)

    def pull(self, name, ID):
        """Pull message from KVServer

        Parameters
        ----------
        name : str
            data name
        ID : tensor (mx.ndarray or torch.tensor)
            a vector storing the IDs

        Return
        ------
        tensor
            a matrix with the same row size of ID
        """
        assert ID.dim() == 1, 'ID must be a vector.'
        group_size = [0] * self._server_count
        for id in ID:
            server_id = self._get_server_id(id.item(), name)
            group_size[server_id] += 1
        min_idx = 0
        max_idx = 0
        server_count = 0
        for idx in range(self._server_count):
            if group_size[idx] == 0:
                continue
            server_count += 1
            max_idx += group_size[idx]
            range_id = ID[min_idx:max_idx]
            min_idx = max_idx
            msg = KVStoreMsg(
                type=KVMsgType.PULL,
                rank=self._client_id,
                name=name,
                id=range_id,
                data=None)
            _send_kv_msg(self._sender, msg, idx)
        # Recv back message
        msg_list = []
        for idx in range(self._server_count):
            if group_size[idx] == 0:
                continue
            msg = _recv_kv_msg(self._receiver)
            assert msg.type == KVMsgType.PULL_BACK
            msg_list.append(msg)

        return self._merge_msg(msg_list)
    
    def shut_down(self):
        """Shutdown all KVServer nodes

        We usually invoke this API by just one client (e.g., client_0).
        """
        for server_id in range(self._server_count):
            msg = KVStoreMsg(
                type=KVMsgType.FINAL,
                rank=self._client_id,
                name=None,
                id=None,
                data=None)
            _send_kv_msg(self._sender, msg, server_id)

    def get_id(self):
        """Get client id

        Return
        ------
        int
            KVClient ID
        """
        return self._client_id

    def _get_server_id(self, id, name):
        """Get target server id by given a data id

        Parameters
        ----------
        id : int
            data id
        name : str
            data name

        Return
        ------
        int
           target server id
        """
        count = math.ceil(self._data_size[name] / self._server_count)
        return int(id / count)

    def _sort_func(self, msg):
        """Sort function for KVStoreMsg: sort message by rank

        Parameters
        ----------
        msg : KVStoreMsg
            KVstore message
        """
        return msg.rank

    def _merge_msg(self, msg_list):
        """Merge separated message to a big matrix

        Parameters
        ----------
        msg_list : list
            a list of KVStoreMsg

        Return
        ------
        tensor (mx.ndarray or torch.tensor)
            a merged data matrix
        """
        msg_list.sort(key=self._sort_func)
        return torch.cat([msg.data for msg in msg_list], 0)