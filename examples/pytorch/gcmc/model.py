"""NN modules"""
import torch as th
import torch.nn as nn
import dgl.function as fn

from utils import get_activation

class GCMCLayer(nn.Module):
    r"""GCMC layer

    .. math::
        z_j^{(l+1)} = \sigma_{agg}\left[\mathrm{agg}\left(
        \sum_{j\in\mathcal{N}_1}\frac{1}{c_{ij}}W_1h_j, \ldots,
        \sum_{j\in\mathcal{N}_R}\frac{1}{c_{ij}}W_Rh_j
        \right)\right]

    After that, apply an extra output projection:

    .. math::
        h_j^{(l+1)} = \sigma_{out}W_oz_j^{(l+1)}

    The equation is applied to both user nodes and movie nodes and the parameters
    are not shared unless ``share_user_item_param`` is true.

    Parameters
    ----------
    rating_vals : list of int or float
        Possible rating values.
    user_in_units : int
        Size of user input feature
    movie_in_units : int
        Size of movie input feature
    msg_units : int
        Size of message :math:`W_rh_j`
    out_units : int
        Size of of final output user and movie features
    dropout_rate : float, optional
        Dropout rate (Default: 0.0)
    agg : str, optional
        Function to aggregate messages of different ratings.
        Could be any of the supported cross type reducers:
        "sum", "max", "min", "mean", "stack".
        (Default: "stack")
    agg_act : callable, str, optional
        Activation function :math:`sigma_{agg}`. (Default: None)
    out_act : callable, str, optional
        Activation function :math:`sigma_{agg}`. (Default: None)
    share_user_item_param : bool, optional
        If true, user node and movie node share the same set of parameters.
        Require ``user_in_units`` and ``move_in_units`` to be the same.
        (Default: False)
    """
    def __init__(self,
                 rating_vals,
                 user_in_units,
                 movie_in_units,
                 msg_units,
                 out_units,
                 dropout_rate=0.0,
                 agg='stack',  # or 'sum'
                 agg_act=None,
                 out_act=None,
                 share_user_item_param=False):
        super(GCMCLayer, self).__init__()
        self.rating_vals = rating_vals
        self.agg = agg
        self.share_user_item_param = share_user_item_param
        self.ufc = nn.Linear(msg_units, out_units)
        if share_user_item_param:
            self.ifc = self.ufc
        else:
            self.ifc = nn.Linear(msg_units, out_units)
        if agg == 'stack':
            # divide the original msg unit size by number of ratings to keep
            # the dimensionality
            assert msg_units % len(rating_vals) == 0
            msg_units = msg_units // len(rating_vals)
        self.dropout = nn.Dropout(dropout_rate)
        self.W_r = nn.ParameterDict()
        for rating in rating_vals:
            # PyTorch parameter name can't contain "."
            rating = str(rating).replace('.', '_')
            if share_user_item_param and user_in_units == movie_in_units:
                self.W_r[rating] = nn.Parameter(th.randn(user_in_units, msg_units))
                self.W_r['rev-%s' % rating] = self.W_r[rating]
            else:
                self.W_r[rating] = nn.Parameter(th.randn(user_in_units, msg_units))
                self.W_r['rev-%s' % rating] = nn.Parameter(th.randn(movie_in_units, msg_units))
        self.agg_act = get_activation(agg_act)
        self.out_act = get_activation(out_act)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat=None, ifeat=None):
        """Forward function

        Normalizer constant :math:`c_{ij}` is stored as two node data "ci"
        and "cj".

        Parameters
        ----------
        graph : DGLHeteroGraph
            User-movie rating graph. It should contain two node types: "user"
            and "movie" and many edge types each for one rating value.
        ufeat : torch.Tensor, optional
            User features. If None, using an identity matrix.
        ifeat : torch.Tensor, optional
            Movie features. If None, using an identity matrix.

        Returns
        -------
        new_ufeat : torch.Tensor
            New user features
        new_ifeat : torch.Tensor
            New movie features
        """
        num_u = graph.number_of_nodes('user')
        num_i = graph.number_of_nodes('movie')
        funcs = {}
        for i, rating in enumerate(self.rating_vals):
            rating = str(rating)
            # W_r * x
            x_u = dot_or_identity(ufeat, self.W_r[rating.replace('.', '_')])
            x_i = dot_or_identity(ifeat, self.W_r['rev-%s' % rating.replace('.', '_')])
            # left norm and dropout
            x_u = x_u * self.dropout(graph.nodes['user'].data['cj'])
            x_i = x_i * self.dropout(graph.nodes['movie'].data['cj'])
            graph.nodes['user'].data['h%d' % i] = x_u
            graph.nodes['movie'].data['h%d' % i] = x_i
            funcs[rating] = (fn.copy_u('h%d' % i, 'm'), fn.sum('m', 'h'))
            funcs['rev-%s' % rating] = (fn.copy_u('h%d' % i, 'm'), fn.sum('m', 'h'))
        # message passing
        graph.multi_update_all(funcs, self.agg)
        ufeat = graph.nodes['user'].data.pop('h').view(num_u, -1)
        ifeat = graph.nodes['movie'].data.pop('h').view(num_i, -1)
        # right norm
        ufeat = ufeat * graph.nodes['user'].data['ci']
        ifeat = ifeat * graph.nodes['movie'].data['ci']
        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        return self.out_act(ufeat), self.out_act(ifeat)

class SampleGCMCLayer(GCMCLayer):
    r"""Sample based GCMC layer

    Implemented a sample based GCMC algorithm. It accepts two minibatch
    graph: ugraph for generating user embeddings and igraph for generating
    item embeddings.

    Parameters
    ----------
    rating_vals : list of int or float
        Possible rating values.
    user_in_units : int
        Size of user input feature
    movie_in_units : int
        Size of movie input feature
    msg_units : int
        Size of message :math:`W_rh_j`
    out_units : int
        Size of of final output user and movie features
    dropout_rate : float, optional
        Dropout rate (Default: 0.0)
    agg : str, optional
        Function to aggregate messages of different ratings.
        Could be any of the supported cross type reducers:
        "sum", "max", "min", "mean", "stack".
        (Default: "stack")
    agg_act : callable, str, optional
        Activation function :math:`sigma_{agg}`. (Default: None)
    out_act : callable, str, optional
        Activation function :math:`sigma_{agg}`. (Default: None)
    share_user_item_param : bool, optional
        If true, user node and movie node share the same set of parameters.
        Require ``user_in_units`` and ``move_in_units`` to be the same.
        (Default: False)
    """
    def __init__(self,
                 rating_vals,
                 user_in_units,
                 movie_in_units,
                 msg_units,
                 out_units,
                 dropout_rate=0.0,
                 agg='stack',  # or 'sum'
                 agg_act=None,
                 out_act=None,
                 share_user_item_param=False,
                 device=None):
        super(SampleGCMCLayer, self).__init__(rating_vals,
                                              user_in_units,
                                              movie_in_units,
                                              msg_units,
                                              out_units,
                                              dropout_rate,
                                              agg,  # or 'sum'
                                              agg_act,
                                              out_act,
                                              share_user_item_param)
        # move part of mode params into GPU when required
        self.device = device
        self.share_user_item_param
        
    def partial_to(self, device):
        """Put parameters into device except W_r
        
        Parameters
        ----------
        device : torch device
            Which device the parameters are put in.
        """
        self.device = device
        if device is not None:
            self.ufc.to(device)
            if self.share_user_item_param is False:
                self.ifc.to(device)
            self.dropout.to(device)

    def forward(self, ugraph, igraph, ufeat=None, ifeat=None):
        """Forward function

        Normalizer constant :math:`c_{ij}` is stored as two node data "ci"
        and "cj".

        Parameters
        ----------
        ugraph : DGLHeteroGraph
            User-item rating graph. It should be a bipartite graph containing
            edges only from item to user. Used in generating user embeddings.
        igraph : DGLHeteroGraph
            User-item rating graph. It should be a bipartite graph containing
            edges only from user to item. Used in generating item embeddings.
        ufeat : torch.Tensor, optional
            User features. If None, using an identity matrix.
        ifeat : torch.Tensor, optional
            Movie features. If None, using an identity matrix.

        Returns
        -------
        new_ufeat : torch.Tensor
            New user features
        new_ifeat : torch.Tensor
            New movie features
        """
        num_u = ugraph.number_of_nodes('user')
        num_i = igraph.number_of_nodes('movie')
        ufuncs = {}
        ifuncs = {}
        for i, rating in enumerate(self.rating_vals):
            rating = str(rating)
            # W_r * x
            x_u = dot_or_identity(ufeat, self.W_r[rating.replace('.', '_')], self.device)
            x_i = dot_or_identity(ifeat, self.W_r['rev-%s' % rating.replace('.', '_')], self.device)
            # left norm and dropout
            x_u = x_u * self.dropout(igraph.srcnodes['user'].data['cj'].to(self.device))
            x_i = x_i * self.dropout(ugraph.srcnodes['movie'].data['cj'].to(self.device))
            igraph.srcnodes['user'].data['h%d' % i] = x_u
            ugraph.srcnodes['movie'].data['h%d' % i] = x_i
            ifuncs[rating] = (fn.copy_u('h%d' % i, 'm'), fn.sum('m', 'h'))
            ufuncs['rev-%s' % rating] = (fn.copy_u('h%d' % i, 'm'), fn.sum('m', 'h'))
        # message passing
        ugraph.multi_update_all(ufuncs, self.agg)
        igraph.multi_update_all(ifuncs, self.agg)
        ufeat = ugraph.dstnodes['user'].data.pop('h').view(num_u, -1)
        ifeat = igraph.dstnodes['movie'].data.pop('h').view(num_i, -1)
        # right norm
        ufeat = ufeat * ugraph.dstnodes['user'].data['ci'].to(self.device)
        ifeat = ifeat * igraph.dstnodes['movie'].data['ci'].to(self.device)
        # fc and non-linear
        ufeat = self.agg_act(ufeat)
        ifeat = self.agg_act(ifeat)
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        ufeat = self.ufc(ufeat)
        ifeat = self.ifc(ifeat)
        return self.out_act(ufeat), self.out_act(ifeat)

class BiDecoder(nn.Module):
    r"""Bilinear decoder.

    .. math::
        p(M_{ij}=r) = \text{softmax}(u_i^TQ_rv_j)

    The trainable parameter :math:`Q_r` is further decomposed to a linear
    combination of basis weight matrices :math:`P_s`:

    .. math::
        Q_r = \sum_{s=1}^{b} a_{rs}P_s

    Parameters
    ----------
    rating_vals : list of int or float
        Possible rating values.
    in_units : int
        Size of input user and movie features
    num_basis_functions : int, optional
        Number of basis. (Default: 2)
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    """
    def __init__(self,
                 rating_vals,
                 in_units,
                 num_basis_functions=2,
                 dropout_rate=0.0):
        super(BiDecoder, self).__init__()
        self.rating_vals = rating_vals
        self._num_basis_functions = num_basis_functions
        self.dropout = nn.Dropout(dropout_rate)
        self.Ps = nn.ParameterList()
        for i in range(num_basis_functions):
            self.Ps.append(nn.Parameter(th.randn(in_units, in_units)))
        self.rate_out = nn.Linear(self._num_basis_functions, len(rating_vals), bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat, ifeat):
        """Forward function.

        Parameters
        ----------
        graph : DGLHeteroGraph
            "Flattened" user-movie graph with only one edge type.
        ufeat : th.Tensor
            User embeddings. Shape: (|V_u|, D)
        ifeat : th.Tensor
            Movie embeddings. Shape: (|V_m|, D)

        Returns
        -------
        th.Tensor
            Predicting scores for each user-movie edge.
        """
        graph = graph.local_var()
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        graph.nodes['movie'].data['h'] = ifeat
        basis_out = []
        for i in range(self._num_basis_functions):
            graph.nodes['user'].data['h'] = ufeat @ self.Ps[i]
            graph.apply_edges(fn.u_dot_v('h', 'h', 'sr'))
            basis_out.append(graph.edata['sr'].unsqueeze(1))
        out = th.cat(basis_out, dim=1)
        out = self.rate_out(out)
        return out

class SampleBiDecoder(BiDecoder):
    r"""Sample based Bilinear decoder.

    Implemented a sample based GCMC algorithm. It accepts user embeddings and
    item embeddings and output a score for each possible label.

    Parameters
    ----------
    rating_vals : list of int or float
        Possible rating values.
    in_units : int
        Size of input user and movie features
    num_basis_functions : int, optional
        Number of basis. (Default: 2)
    dropout_rate : float, optional
        Dropout raite (Default: 0.0)
    """
    def __init__(self,
                 rating_vals,
                 in_units,
                 num_basis_functions=2,
                 dropout_rate=0.0):
        super(SampleBiDecoder, self).__init__(rating_vals,
                                              in_units,
                                              num_basis_functions,
                                              dropout_rate)

    def forward(self, ufeat, ifeat):
        """Forward function.

        Parameters
        ----------
        ufeat : th.Tensor
            User embeddings. Shape: (Batsh_Size, D)
        ifeat : th.Tensor
            Movie embeddings. Shape: (Batsh_Size, D)

        Returns
        -------
        th.Tensor
            Predicting scores for each user-movie edge.
        """
        ufeat = self.dropout(ufeat)
        ifeat = self.dropout(ifeat)
        basis_out = []
        for i in range(self._num_basis_functions):
            ufeat_i = ufeat @ self.Ps[i]
            out = th.einsum('ab,ab->a', ufeat_i, ifeat)
            basis_out.append(out.unsqueeze(1))
        out = th.cat(basis_out, dim=1)
        out = self.rate_out(out)
        return out

def dot_or_identity(A, B, device=None):
    # if A is None, treat as identity matrix
    if A is None:
        return B
    elif len(A.shape) == 1:
        if device is None:
            return B[A]
        else:
            return B[A].to(device)
    else:
        return A @ B
