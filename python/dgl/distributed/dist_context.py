"""Initialize the distributed services"""

import multiprocessing as mp
import traceback
import atexit
import time
import os
import sys
from enum import Enum

from . import rpc
from .constants import MAX_QUEUE_SIZE
from .kvstore import init_kvstore, close_kvstore
from .rpc_client import connect_to_server, shutdown_servers
from .role import init_role
from .. import utils

SAMPLER_POOL = None
NUM_SAMPLER_WORKERS = 0
INITIALIZED = False


def set_initialized(value=True):
    """Set the initialized state of rpc"""
    global INITIALIZED
    INITIALIZED = value


def get_sampler_pool():
    """Return the sampler pool and num_workers"""
    return SAMPLER_POOL, NUM_SAMPLER_WORKERS


def _init_rpc(ip_config, num_servers, max_queue_size, net_type, role, num_threads):
    ''' This init function is called in the worker processes.
    '''
    try:
        utils.set_num_threads(num_threads)
        if os.environ.get('DGL_DIST_MODE', 'standalone') != 'standalone':
            connect_to_server(ip_config, num_servers, max_queue_size, net_type)
        init_role(role)
        init_kvstore(ip_config, num_servers, role)
    except Exception as e:
        print(e, flush=True)
        traceback.print_exc()
        raise e


class MpCommand(Enum):
    INIT_RPC = 0  # Not used in the task queue
    SET_COLLATE_FN = 1
    CALL_BARRIER = 2
    DELETE_COLLATE_FN = 3
    CALL_COLLATE_FN = 4
    CALL_FN_ALL_WORKERS = 5
    FINALIZE_POOL = 6


def init_process(_, rpc_config, mp_contexts):
    try:
        _init_rpc(*rpc_config)
        keep_polling = True
        data_queue, task_queue, barrier = mp_contexts
        collate_fn_dict = {}

        while keep_polling:
            command, args = task_queue.get(timeout=1800)
            if command == MpCommand.SET_COLLATE_FN:
                dataloader_name, func = args
                collate_fn_dict[dataloader_name] = func
            elif command == MpCommand.CALL_BARRIER:
                barrier.wait()
            elif command == MpCommand.DELETE_COLLATE_FN:
                dataloader_name, = args
                del collate_fn_dict[dataloader_name]
            elif command == MpCommand.CALL_COLLATE_FN:
                dataloader_name, collate_args = args
                data_queue.put(
                    (dataloader_name, collate_fn_dict[dataloader_name](collate_args)))
            elif command == MpCommand.CALL_FN_ALL_WORKERS:
                func, func_args = args
                func(func_args)
            elif command == MpCommand.FINALIZE_POOL:
                _exit()
                keep_polling = False
            else:
                raise Exception("Unknown command")
    except Exception as e:
        traceback.print_exc()
        raise e


class CustomPool:
    def __init__(self, num_workers, rpc_config):
        ctx = mp.get_context("spawn")
        self.num_workers = num_workers
        self.queue_size = num_workers * 4
        self.result_queue = ctx.Queue(self.queue_size)
        self.task_queues = []
        self.process_list = []
        self.current_proc_id = 0
        self.cache_result_dict = {}
        self.barrier = ctx.Barrier(num_workers)
        for i in range(num_workers):
            task_queue = ctx.Queue(self.queue_size)
            self.task_queues.append(task_queue)
            proc = ctx.Process(target=init_process, args=(
                i, rpc_config, (self.result_queue, task_queue, self.barrier)))
            proc.daemon = True
            proc.start()
            self.process_list.append(proc)

    def set_collate_fn(self, func, dataloader_name):
        for i in range(self.num_workers):
            self.task_queues[i].put(
                (MpCommand.SET_COLLATE_FN, (dataloader_name, func)))

    def submit_task(self, dataloader_name, args):
        # Round robin
        self.task_queues[self.current_proc_id].put(
            (MpCommand.CALL_COLLATE_FN, (dataloader_name, args)))
        self.current_proc_id = (self.current_proc_id + 1) % self.num_workers

    def submit_task_to_all_workers(self, func, args):
        for i in range(self.num_workers):
            self.task_queues[i].put(
                (MpCommand.CALL_FN_ALL_WORKERS, (func, args)))

    def get_result(self, dataloader_name):
        result_dataloader_name, result = self.result_queue.get(timeout=1800)
        assert result_dataloader_name == dataloader_name
        return result

    def delete_collate_fn(self, dataloader_name):
        for i in range(self.num_workers):
            self.task_queues[i].put(
                (MpCommand.DELETE_COLLATE_FN, (dataloader_name, )))

    def close(self):
        for i in range(self.num_workers):
            self.task_queues[i].put((MpCommand.FINALIZE_POOL, tuple()))

    def join(self):
        for i in range(self.num_workers):
            self.process_list[i].join()


def initialize(ip_config, num_servers=1, num_workers=0,
               max_queue_size=MAX_QUEUE_SIZE, net_type='socket',
               num_worker_threads=1):
    """Initialize DGL's distributed module

    This function initializes DGL's distributed module. It acts differently in server
    or client modes. In the server mode, it runs the server code and never returns.
    In the client mode, it builds connections with servers for communication and
    creates worker processes for distributed sampling. `num_workers` specifies
    the number of sampling worker processes per trainer process.
    Users also have to provide the number of server processes on each machine in order
    to connect to all the server processes in the cluster of machines correctly.

    Parameters
    ----------
    ip_config: str
        File path of ip_config file
    num_servers : int
        The number of server processes on each machine. This argument is deprecated in DGL 0.7.0.
    num_workers: int
        Number of worker process on each machine. The worker processes are used
        for distributed sampling. This argument is deprecated in DGL 0.7.0.
    max_queue_size : int
        Maximal size (bytes) of client queue buffer (~20 GB on default).

        Note that the 20 GB is just an upper-bound and DGL uses zero-copy and
        it will not allocate 20GB memory at once.
    net_type : str, optional
        Networking type. Currently the only valid option is ``'socket'``.

        Default: ``'socket'``
    num_worker_threads: int
        The number of threads in a worker process.

    Note
    ----
    Users have to invoke this API before any DGL's distributed API and framework-specific
    distributed API. For example, when used with Pytorch, users have to invoke this function
    before Pytorch's `pytorch.distributed.init_process_group`.
    """
    if os.environ.get('DGL_ROLE', 'client') == 'server':
        from .dist_graph import DistGraphServer
        assert os.environ.get('DGL_SERVER_ID') is not None, \
            'Please define DGL_SERVER_ID to run DistGraph server'
        assert os.environ.get('DGL_IP_CONFIG') is not None, \
            'Please define DGL_IP_CONFIG to run DistGraph server'
        assert os.environ.get('DGL_NUM_SERVER') is not None, \
            'Please define DGL_NUM_SERVER to run DistGraph server'
        assert os.environ.get('DGL_NUM_CLIENT') is not None, \
            'Please define DGL_NUM_CLIENT to run DistGraph server'
        assert os.environ.get('DGL_CONF_PATH') is not None, \
            'Please define DGL_CONF_PATH to run DistGraph server'
        formats = os.environ.get('DGL_GRAPH_FORMAT', 'csc').split(',')
        formats = [f.strip() for f in formats]
        serv = DistGraphServer(int(os.environ.get('DGL_SERVER_ID')),
                               os.environ.get('DGL_IP_CONFIG'),
                               int(os.environ.get('DGL_NUM_SERVER')),
                               int(os.environ.get('DGL_NUM_CLIENT')),
                               os.environ.get('DGL_CONF_PATH'),
                               graph_format=formats)
        serv.start()
        sys.exit()
    else:
        if os.environ.get('DGL_NUM_SAMPLER') is not None:
            num_workers = int(os.environ.get('DGL_NUM_SAMPLER'))
        else:
            num_workers = 0
        if os.environ.get('DGL_NUM_SERVER') is not None:
            num_servers = int(os.environ.get('DGL_NUM_SERVER'))
        else:
            num_servers = 1

        rpc.reset()
        global SAMPLER_POOL
        global NUM_SAMPLER_WORKERS
        is_standalone = os.environ.get(
            'DGL_DIST_MODE', 'standalone') == 'standalone'
        if num_workers > 0 and not is_standalone:
            SAMPLER_POOL = CustomPool(num_workers, (ip_config, num_servers, max_queue_size,
                                                    net_type, 'sampler', num_worker_threads))
        else:
            SAMPLER_POOL = None
        NUM_SAMPLER_WORKERS = num_workers
        if not is_standalone:
            assert num_servers is not None and num_servers > 0, \
                'The number of servers per machine must be specified with a positive number.'
            connect_to_server(ip_config, num_servers, max_queue_size, net_type)
        init_role('default')
        init_kvstore(ip_config, num_servers, 'default')


def finalize_client():
    """Release resources of this client."""
    if os.environ.get('DGL_DIST_MODE', 'standalone') != 'standalone':
        rpc.finalize_sender()
        rpc.finalize_receiver()
    global INITIALIZED
    INITIALIZED = False


def _exit():
    exit_client()
    time.sleep(1)


def finalize_worker():
    """Finalize workers
       Python's multiprocessing pool will not call atexit function when close
    """
    if SAMPLER_POOL is not None:
        SAMPLER_POOL.close()


def join_finalize_worker():
    """join the worker close process"""
    global SAMPLER_POOL
    if SAMPLER_POOL is not None:
        SAMPLER_POOL.join()
    SAMPLER_POOL = None


def is_initialized():
    """Is RPC initialized?
    """
    return INITIALIZED


def exit_client():
    """Trainer exits

    This function is called automatically when a Python process exits. Normally,
    the training script does not need to invoke this function at the end.

    In the case that the training script needs to initialize the distributed module
    multiple times (so far, this is needed in the unit tests), the training script
    needs to call `exit_client` before calling `initialize` again.
    """
    # Only client with rank_0 will send shutdown request to servers.
    finalize_worker()  # finalize workers should be earilier than barrier, and non-blocking
    if os.environ.get('DGL_DIST_MODE', 'standalone') != 'standalone':
        rpc.client_barrier()
        shutdown_servers()
    finalize_client()
    join_finalize_worker()
    close_kvstore()
    atexit.unregister(exit_client)
