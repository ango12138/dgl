"""Implementation for core graph computation."""

from .base import DGLError, is_all, NID, EID, ALL
from . import function as fn
from .udf import NodeBatch, EdgeBatch
from . import ops

def is_builtin(func):
    return isinstance(func, fn.BuiltinFunction)

def invoke_node_udf(graph, nodes, ntype, func, ndata):
    if is_all(nodes):
        nodes = graph.nodes(ntype=ntype)
    nbatch = NodeBatch(graph, nodes, ntype, ndata)
    return func(nbatch)

def invoke_edge_udf(graph, eid, etype, func, orig_eid=None):
    etid = graph.get_etype_id(etype)
    stid, dtid = graph._graph.metagraph.find_edge(etid)
    if is_all(eid):
        u, v, eid = graph.edges(form='all', etype=etype)
        edata = graph._edge_frames[etid]
    else:
        u, v = graph.find_edges(eid, etype=etype)
        edata = graph._edge_frames[etid].subframe(eid)
    srcdata = graph._node_frames[stid].subframe(u)
    dstdata = graph._node_frames[dtid].subframe(v)
    ebatch = EdgeBatch(graph, eid if orig_eid is None else orig_eid,
                       etype, srcdata, edata, dstdata)
    return func(ebatch)

def invoke_udf_reduce(graph, func, edata=None):
    pass

def invoke_gsddmm(graph, func):
    alldata = [graph.srcdata, graph.dstdata, graph.edata]
    if isinstance(func, fn.BinaryMessageFunction):
        x = alldata[func.lhs][func.lhs_field]
        y = alldata[func.rhs][func.rhs_field]
        op = getattr(ops, func.name)
        z = op(graph, x, y)
    else:
        x = alldata[func.target][func.in_field]
        op = getattr(ops, func.name)
        z = op(graph, x)
    return {func.out_field : z}

def invoke_gspmm(graph, mfunc, rfunc, *, srcdata=None, dstdata=None, edata=None):
    # sanity check
    if mfunc.out_field != rfunc.msg_field:
        raise DGLError('Invalid message ({}) and reduce ({}) function pairs.'
                       ' The output field of the message function must be equal to the'
                       ' message field of the reduce function.'.format(mfunc, rfunc))
    if edata is None:
        edata = graph.edata
    if srcdata is None:
        srcdata = graph.srcdata
    if dstdata is None:
        dstdata = graph.dstdata
    alldata = [srcdata, dstdata, edata]

    if isinstance(mfunc, fn.BinaryMessageFunction):
        x = alldata[mfunc.lhs][mfunc.lhs_field]
        y = alldata[mfunc.rhs][mfunc.rhs_field]
        op = getattr(ops, '{}_{}'.format(mfunc.name, rfunc.name))
        z = op(graph, x, y)
    else:
        x = alldata[mfunc.target][mfunc.in_field]
        op = getattr(ops, '{}_{}'.format(mfunc.name, rfunc.name))
        z = op(graph, x)
    return {rfunc.out_field : z}

def message_passing(g, mfunc, rfunc, afunc):
    if g.number_of_edges() == 0:
        # No message passing is triggered.
        return
    if is_builtin(mfunc) and is_builtin(rfunc):
        ndata = invoke_gspmm(g, mfunc, rfunc)
    else:
        # message phase
        if is_builtin(mfunc):
            edata = invoke_gsddmm(g, mfunc)
        else:
            orig_eid = g.edata.get(EID, None)
            edata = invoke_edge_udf(g, ALL, g.canonical_etypes[0], mfunc, orig_eid=orig_eid)
        # reduce phase
        if is_builtin(rfunc):
            msg = rfunc.msg_field
            ndata = invoke_gspmm(g, fn.copy_e(msg, msg), rfunc, edata=edata)
        else:
            edata.update(g.edata)  # incorporate original edge features
            ndata = invoke_udf_reduce(g, rfunc, edata=edata)
    # apply phase
    if afunc is not None:
        ndata.update(g.dstdata)  # incorporate original node features
        dstnodes = g.dstdata.get(NID, ALL)
        ndata = invoke_node_udf(g, dstnodes, g.dsttypes[0], afunc, ndata)
    return ndata
