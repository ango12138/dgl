"""PyTorch multiprocessing wrapper."""
from functools import wraps
import os
from collections import namedtuple
import traceback
from _thread import start_new_thread
import torch.multiprocessing as mp

def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = mp.Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:  # pylint: disable=broad-except
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function

# pylint: disable=missing-docstring
class Process(mp.Process):
    # pylint: disable=dangerous-default-value
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        target = thread_wrapped_func(target)
        super().__init__(group, target, name, args, kwargs, daemon=daemon)

ProcessContext = namedtuple('ProcessContext', ['queue', 'queue_ack', 'rank', 'nprocs'])
mp_timeout = int(os.environ.get('DGL_MP_TIMEOUT', '10'))
_process_context = None

def call_once_and_share(func, rank=0):
    """Invoke the function in a single process of the process group spawned by
    :func:`spawn`, and share the result to other processes.

    Parameters
    ----------
    func : callable
        Any callable that accepts no arguments and returns an arbitrary object.
    rank : int, optional
        The process ID to actually execute the function.
    """
    global _process_context
    if _process_context is None:
        raise RuntimeError(
            'call_once_and_share can only be called within processes spawned by '
            'dgl.multiprocessing.spawn() function.')

    if _process_context.rank == rank:
        result = func()
        for _ in range(_process_context.nprocs - 1):
            _process_context.queue.put(result)
        # Synchronize
        for _ in range(_process_context.nprocs - 1):
            _process_context.queue_ack.get(timeout=mp_timeout)
    else:
        result = _process_context.queue.get(timeout=mp_timeout)
        # Synchronize
        _process_context.queue_ack.put(None)
    return result

def _spawn_entry(rank, queue, queue_ack, nprocs, fn, *args):
    global _process_context
    _process_context = ProcessContext(queue, queue_ack, rank, nprocs)
    fn(rank, *args)

def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    """A wrapper around :func:`torch.multiprocessing.spawn` that allows calling
    DGL-specific multiprocessing functions in :mod:`dgl.multiprocessing` namespace."""
    ctx = mp.get_context(start_method)

    # The following two queues are for call_once_and_share
    queue = ctx.Queue()
    queue_ack = ctx.Queue()

    mp.spawn(_spawn_entry, args=(queue, queue_ack, nprocs, fn) + tuple(args), nprocs=nprocs,
             join=join, daemon=daemon, start_method=start_method)
