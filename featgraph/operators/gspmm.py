""" The compute function and schedules for GSpMM kernels written in TVM. """
import tvm
from tvm import te
from tvm import topi
from tvm.tir import IntImm, decl_buffer
from tvm.topi.utils import prod
from .utils import binary_op_map, reduce_op_map, TargetCode

__all__ = ['gspmm']


def _spmm(out_shp, binary_op, reduce_op,
          indptr, indices,
          ufeat, efeat):
    use_u = binary_op != 'copy_rhs'
    use_e = binary_op != 'copy_lhs'
    require_arg = reduce_op in ['min', 'max']
    n = indptr.shape[0] - 1
    def node_func(*args):
        row = args[0]
        start = indptr[row]
        end = indptr[row + 1]
        # The following line avoids the possible illegal memory access error
        # Related issue: https://github.com/apache/tvm/issues/6596
        length = te.if_then_else(row < n, end - start, 0)
        k = te.reduce_axis((0, length), name='k')
        eid = k + start
        uid = indices[eid]
        msg = []
        e_val, u_val = None, None
        if use_e:
            e_val = efeat.__getitem__((eid,) + args[1:])
            if require_arg:
                msg.append(eid)
        if use_u:
            u_val = ufeat.__getitem__((uid,) + args[1:])
            if require_arg:
                msg.append(uid)
        msg.append(binary_op_map[binary_op](u_val, e_val))
        return reduce_op_map[reduce_op](msg, axis=k)
        
    ret = te.compute(out_shp, node_func, name='out')
    if reduce_op == 'sum':
        ret = (ret,)
    return ret


def _spmm_cuda_general(sched, out):
    out_len = prod(out.shape[1:])
    node_axis = out.op.axis[0]
    feat_axis = sched[out].fuse(*out.op.axis[1:])
    #ntx = tvm.autotvm.task.space.get_pow2s(out_len)[-1]
    #ntx = 1024 if ntx > 1024 else ntx
    ntx = 1024
    feat_outer, feat_inner = sched[out].split(feat_axis, factor=ntx)
    #nbx = 2**31 - 1 if out.shape[0] > 2**31 - 1 else out.shape[0]
    #nbx = out.shape[0]
    #node_outer, _ = sched[out].split(node_axis, nparts=nbx)
    sched[out].bind(node_axis, te.thread_axis('blockIdx.x'))
    sched[out].bind(feat_outer, te.thread_axis('blockIdx.y'))
    sched[out].bind(feat_inner, te.thread_axis('threadIdx.x'))


def _spmm_cuda_tree_reduce(sched, out):
    reduce_axis = out.op.reduce_axis[0]
    node_axis = out.op.axis[0]
    feat_axis = sched[out].fuse(*out.op.axis[1:])
    _, red_inner = sched[out].split(reduce_axis, factor=32)
    # TODO(zihao): the following line is problematic, will fix.
    nbx = 2**31 - 1 if out.shape[0] > 2**31 - 1 else out.shape[0]
    node_outer, _ = sched[out].split(node_axis, nparts=nbx)
    sched[out].bind(red_inner, te.thread_axis('threadIdx.x'))
    sched[out].bind(feat_axis, te.thread_axis('threadIdx.y'))
    sched[out].bind(node_outer, te.thread_axis('blockIdx.x'))


_gspmm_cuda_schedule = {
    'general': _spmm_cuda_general,
    'tree': _spmm_cuda_tree_reduce
}


def gspmm(binary_op, reduce_op,
          ndim,
          indice_type, feat_type,
          schedule_type='general',
          target='cuda'):
    """
    Compile GSPMM kernels using TVM.
    """
    num_rows = te.var('num_rows', indice_type)
    num_cols = te.var('num_cols', indice_type)
    nnz = te.var('nnz', indice_type)

    indptr = te.placeholder((num_rows + 1,), indice_type, 'indptr')
    indices = te.placeholder((nnz,), indice_type, 'indices')
    ufeat_shp = [te.var('d_u{}'.format(i)) for i in range(ndim)]
    efeat_shp = [te.var('d_e{}'.format(i)) for i in range(ndim)]
    out_feat_shp = [te.var('d_v{}'.format(i)) for i in range(ndim)]

    ufeat = te.placeholder([num_cols] + ufeat_shp, feat_type, 'ufeat')
    efeat = te.placeholder([nnz] + efeat_shp, feat_type, 'efeat')

    ret = _spmm([num_rows] + out_feat_shp,
                binary_op, reduce_op,
                indptr, indices, ufeat, efeat)
    out = ret[-1]
    sched = te.create_schedule(out.op)
    
    if target == 'cuda':
        _gspmm_cuda_schedule[schedule_type](sched, out)
    else:
        raise NotImplementedError("CPU schedule not implemented yet.")
    
    f_input = [indptr, indices]
    f_name = '_'.join(str(x) for x in [
        'spmm', binary_op, reduce_op, ndim,
        indice_type, feat_type,
        schedule_type, target
    ])

    use_u = binary_op != 'copy_rhs'
    use_e = binary_op != 'copy_lhs'
    if use_u:
        f_input.append(ufeat)
    if use_e:
        f_input.append(efeat)

    f_input += ret
    # bind autobcast buffer
    u_buffer = decl_buffer(ufeat.shape, ufeat.dtype, name='u_buf',
                           buffer_type='auto_broadcast')
    e_buffer = decl_buffer(efeat.shape, efeat.dtype, name='e_buf',
                           buffer_type='auto_broadcast')
    binds = {ufeat: u_buffer, efeat: e_buffer}
    return tvm.lower(sched, f_input, name=f_name, binds=binds)
