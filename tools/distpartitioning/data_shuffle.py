import os
import sys
import constants
import numpy as np
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import dgl
import logging

from timeit import default_timer as timer
from datetime import timedelta
from dataset_utils import get_dataset
from utils import read_ntype_partition_files, read_json, get_node_types, \
                    augment_edge_data, get_gnid_range_map, \
                    write_dgl_objects, write_metadata_json, get_ntype_featnames, \
                    get_idranges
from gloo_wrapper import allgather_sizes, gather_metadata_json,\
                    alltoallv_cpu
from globalids import assign_shuffle_global_nids_nodes, \
                    assign_shuffle_global_nids_edges, \
                    lookup_shuffle_global_nids_edges
from convert_partition import create_dgl_object, create_metadata_json
from dist_lookup import DistLookupService

def gen_node_data(rank, world_size, id_lookup, ntid_ntype_map, schema_map):
    '''
    For this data processing pipeline, reading node files is not needed. All the needed information about
    the nodes can be found in the metadata json file. This function generates the nodes owned by a given
    process, using metis partitions. 

    Parameters: 
    -----------
    rank : int
        rank of the process
    world_size : int
        total no. of processes
    id_lookup : instance of class DistLookupService
       Distributed lookup service used to map global-nids to respective partition-ids and 
       shuffle-global-nids
    ntid_ntype_map : 
        a dictionary where keys are node_type ids(integers) and values are node_type names(strings).
    schema_map: 
        dictionary formed by reading the input metadata json file for the input dataset.

        Please note that, it is assumed that for the input graph files, the nodes of a particular node-type are
        split into `p` files (because of `p` partitions to be generated). On a similar node, edges of a particular
        edge-type are split into `p` files as well. 

        #assuming m nodetypes present in the input graph
        "num_nodes_per_chunk" : [
            [a0, a1, a2, ... a<p-1>], 
            [b0, b1, b2, ... b<p-1>], 
            ...
            [m0, m1, m2, ... m<p-1>]
        ]
        Here, each sub-list, corresponding a nodetype in the input graph, has `p` elements. For instance [a0, a1, ... a<p-1>] 
        where each element represents the number of nodes which are to be processed by a process during distributed partitioning.

        In addition to the above key-value pair for the nodes in the graph, the node-features are captured in the
        "node_data" key-value pair. In this dictionary the keys will be nodetype names and value will be a dictionary which 
        is used to capture all the features present for that particular node-type. This is shown in the following example:

        "node_data" : {
            "paper": {       # node type
                "feat": {   # feature key
                    "format": {"name": "numpy"},
                    "data": ["node_data/paper-feat-part1.npy", "node_data/paper-feat-part2.npy"]
                },
                "label": {   # feature key
                    "format": {"name": "numpy"},
                    "data": ["node_data/paper-label-part1.npy", "node_data/paper-label-part2.npy"]
                },
                "year": {   # feature key
                    "format": {"name": "numpy"},
                    "data": ["node_data/paper-year-part1.npy", "node_data/paper-year-part2.npy"]
                }
            }
        }
        In the above textual description we have a node-type, which is paper, and it has 3 features namely feat, label and year. 
        Each feature has `p` files whose location in the filesystem is the list for the key "data" and "foramt" is used to 
        describe storage format. 

    Returns:
    --------
    dictionary : 
        dictionary where keys are column names and values are numpy arrays, these arrays are generated by 
        using information present in the metadata json file

    '''
    local_node_data = { constants.GLOBAL_NID : [], 
                        constants.NTYPE_ID : [], 
                        constants.GLOBAL_TYPE_NID : []
                        }

    type_nid_dict, global_nid_dict = get_idranges(schema_map[constants.STR_NODE_TYPE], 
                                        schema_map[constants.STR_NUM_NODES_PER_CHUNK])

    for ntype_id, ntype_name in ntid_ntype_map.items(): 
        type_start, type_end = type_nid_dict[ntype_name][0][0], type_nid_dict[ntype_name][-1][1]
        gnid_start, gnid_end = global_nid_dict[ntype_name][0, 0], global_nid_dict[ntype_name][0, 1]

        node_partid_slice = id_lookup.get_partition_ids(np.arange(gnid_start, gnid_end, dtype=np.int64)) #exclusive
        cond = node_partid_slice == rank
        own_gnids = np.arange(gnid_start, gnid_end, dtype=np.int64)
        own_gnids = own_gnids[cond]

        own_tnids = np.arange(type_start, type_end, dtype=np.int64)
        own_tnids = own_tnids[cond]

        local_node_data[constants.NTYPE_ID].append(np.ones(own_gnids.shape, dtype=np.int64)*ntype_id)
        local_node_data[constants.GLOBAL_NID].append(own_gnids)
        local_node_data[constants.GLOBAL_TYPE_NID].append(own_tnids)

    for k in local_node_data.keys():
        local_node_data[k] = np.concatenate(local_node_data[k])

    return local_node_data

def exchange_edge_data(rank, world_size, edge_data):
    """
    Exchange edge_data among processes in the world.
    Prepare list of sliced data targeting each process and trigger
    alltoallv_cpu to trigger messaging api

    Parameters:
    -----------
    rank : int
        rank of the process
    world_size : int
        total no. of processes
    edge_data : dictionary
        edge information, as a dicitonary which stores column names as keys and values
        as column data. This information is read from the edges.txt file.

    Returns:
    --------
    dictionary : 
        the input argument, edge_data, is updated with the edge data received by other processes
        in the world.
    """

    input_list = []
    start = timer()
    for i in np.arange(world_size):
        send_idx = (edge_data[constants.OWNER_PROCESS] == i)
        send_idx = send_idx.reshape(edge_data[constants.GLOBAL_SRC_ID].shape[0])
        filt_data = np.column_stack((edge_data[constants.GLOBAL_SRC_ID][send_idx == 1], \
                                    edge_data[constants.GLOBAL_DST_ID][send_idx == 1], \
                                    edge_data[constants.GLOBAL_TYPE_EID][send_idx == 1], \
                                    edge_data[constants.ETYPE_ID][send_idx == 1], \
                                    edge_data[constants.GLOBAL_EID][send_idx == 1]))
        if(filt_data.shape[0] <= 0):
            input_list.append(torch.empty((0,5), dtype=torch.int64))
        else:
            input_list.append(torch.from_numpy(filt_data))
    end = timer()
    
    dist.barrier ()
    output_list = alltoallv_cpu(rank, world_size, input_list)
    end = timer()
    logging.info(f'[Rank: {rank}] Time to send/rcv edge data: {timedelta(seconds=end-start)}')

    #Replace the values of the edge_data, with the received data from all the other processes.
    rcvd_edge_data = torch.cat(output_list).numpy()
    edge_data[constants.GLOBAL_SRC_ID] = rcvd_edge_data[:,0]
    edge_data[constants.GLOBAL_DST_ID] = rcvd_edge_data[:,1]
    edge_data[constants.GLOBAL_TYPE_EID] = rcvd_edge_data[:,2]
    edge_data[constants.ETYPE_ID] = rcvd_edge_data[:,3]
    edge_data[constants.GLOBAL_EID] = rcvd_edge_data[:,4]
    edge_data.pop(constants.OWNER_PROCESS)
    return edge_data

def exchange_node_features(rank, world_size, node_feature_tids, ntype_gnid_map, id_lookup, node_features):
    """
    This function is used to shuffle node features so that each process will receive
    all the node features whose corresponding nodes are owned by the same process. 
    The mapping procedure to identify the owner process is not straight forward. The
    following steps are used to identify the owner processes for the locally read node-
    features. 
    a. Compute the global_nids for the locally read node features. Here metadata json file
        is used to identify the corresponding global_nids. Please note that initial graph input
        nodes.txt files are sorted based on node_types. 
    b. Using global_nids and metis partitions owner processes can be easily identified. 
    c. Now each process sends the global_nids for which shuffle_global_nids are needed to be 
        retrieved. 
    d. After receiving the corresponding shuffle_global_nids these ids are added to the 
        node_data and edge_data dictionaries
    
    This pipeline assumes all the input data in numpy format, except node/edge features which
    are maintained as tensors throughout the various stages of the pipeline execution.

    Parameters: 
    -----------
    rank : int
        rank of the current process
    world_size : int
        total no. of participating processes. 
    node_feature_tids : dictionary
        dictionary with keys as node-type names and value is a dictionary. This dictionary
        contains information about node-features associated with a given node-type and value
        is a list.  This list contains a of indexes, like [starting-idx, ending-idx) which 
        can be used to index into the node feature tensors read from corresponding input files.
    ntypes_gnid_map : dictionary
        mapping between node type names and global_nids which belong to the keys in this dictionary
    id_lookup : instance of class DistLookupService
       Distributed lookup service used to map global-nids to respective partition-ids and 
       shuffle-global-nids
    node_feautres: dicitonary
        dictionry where node_features are stored and this information is read from the appropriate
        node features file which belongs to the current process

    Returns:
    --------
    dictionary : 
        node features are returned as a dictionary where keys are node type names and node feature names 
        and values are tensors
    dictionary : 
        a dictionary of global_nids for the nodes whose node features are received during the data shuffle 
        process
    """
    start = timer()
    own_node_features = {}
    own_global_nids = {}
    #To iterate over the node_types and associated node_features
    for ntype_name, ntype_info in node_feature_tids.items():

        #To iterate over the node_features, of a given node_type 
        #ntype_info is a list of 3 elements
        #[node-feature-name, starting-idx, ending-idx]
        #node-feature-name is the name given to the node-feature, read from the input metadata file
        #[starting-idx, ending-idx) specifies the range of indexes associated with the node-features read from
        #the associated input file. Note that the rows of node-features read from the input file should be same
        #as specified with this range. So no. of rows = ending-idx - starting-idx.
        for feat_info in ntype_info:

            #determine the owner process for these node features. 
            node_feats_per_rank = []
            global_nid_per_rank = []
            feat_name = feat_info[0]
            feat_key = ntype_name+'/'+feat_name
            logging.info(f'[Rank: {rank}] processing node feature: {feat_key}')

            #compute the global_nid range for this node features
            type_nid_start = int(feat_info[1])
            type_nid_end = int(feat_info[2])
            begin_global_nid = ntype_gnid_map[ntype_name][0]
            gnid_start = begin_global_nid + type_nid_start
            gnid_end = begin_global_nid + type_nid_end

            #type_nids for this feature subset on the current rank
            gnids_feat = np.arange(gnid_start, gnid_end)
            tnids_feat = np.arange(type_nid_start, type_nid_end)
            local_idx = np.arange(0, type_nid_end - type_nid_start)

            #check if node features exist for this ntype_name + feat_name
            #this check should always pass, because node_feature_tids are built
            #by reading the input metadata json file for existing node features.
            assert(feat_key in node_features)

            node_feats = node_features[feat_key]
            for part_id in range(world_size):
                partid_slice = id_lookup.get_partition_ids(np.arange(gnid_start, gnid_end, dtype=np.int64))
                cond = (partid_slice == part_id)
                gnids_per_partid = gnids_feat[cond]
                tnids_per_partid = tnids_feat[cond]
                local_idx_partid = local_idx[cond]

                if (gnids_per_partid.shape[0] == 0):
                    node_feats_per_rank.append(torch.empty((0,1), dtype=torch.float))
                    global_nid_per_rank.append(np.empty((0,1), dtype=np.int64))
                else:
                    node_feats_per_rank.append(node_feats[local_idx_partid])
                    global_nid_per_rank.append(torch.from_numpy(gnids_per_partid).type(torch.int64))

            #features (and global nids) per rank to be sent out are ready
            #for transmission, perform alltoallv here.
            output_feat_list = alltoallv_cpu(rank, world_size, node_feats_per_rank)
            output_nid_list = alltoallv_cpu(rank, world_size, global_nid_per_rank)

            #stitch node_features together to form one large feature tensor
            own_node_features[feat_key] = torch.cat(output_feat_list)
            own_global_nids[feat_key] = torch.cat(output_nid_list).numpy()

    end = timer()
    logging.info(f'[Rank: {rank}] Total time for node feature exchange: {timedelta(seconds = end - start)}')
    return own_node_features, own_global_nids

def exchange_graph_data(rank, world_size, node_features, node_feat_tids, edge_data,
        id_lookup, ntypes_ntypeid_map, ntypes_gnid_range_map, ntid_ntype_map, schema_map):
    """
    Wrapper function which is used to shuffle graph data on all the processes. 

    Parameters: 
    -----------
    rank : int
        rank of the current process
    world_size : int
        total no. of participating processes. 
    node_feautres: dicitonary
        dictionry where node_features are stored and this information is read from the appropriate
        node features file which belongs to the current process
    node_feat_tids: dictionary
        in which keys are node-type names and values are triplets. Each triplet has node-feature name
        and the starting and ending type ids of the node-feature data read from the corresponding
        node feature data file read by current process. Each node type may have several features and
        hence each key may have several triplets.
    edge_data : dictionary
        dictionary which is used to store edge information as read from the edges.txt file assigned
        to each process.
    id_lookup : instance of class DistLookupService
       Distributed lookup service used to map global-nids to respective partition-ids and 
       shuffle-global-nids
    ntypes_ntypeid_map : dictionary
        mappings between node type names and node type ids
    ntypes_gnid_range_map : dictionary
        mapping between node type names and global_nids which belong to the keys in this dictionary
    ntid_ntype_map : dictionary
        mapping between node type id and no of nodes which belong to each node_type_id
    schema_map : dictionary
        is the data structure read from the metadata json file for the input graph

    Returns:
    --------
    dictionary : 
        the input argument, node_data dictionary, is updated with the node data received from other processes
        in the world. The node data is received by each rank in the process of data shuffling.
    dictionary : 
        node features dictionary which has node features for the nodes which are owned by the current 
        process
    dictionary : 
        list of global_nids for the nodes whose node features are received when node features shuffling was 
        performed in the `exchange_node_features` function call
    dictionary : 
        the input argument, edge_data dictionary, is updated with the edge data received from other processes
        in the world. The edge data is received by each rank in the process of data shuffling.
    """
    rcvd_node_features, rcvd_global_nids = exchange_node_features(rank, world_size, node_feat_tids, \
                                                ntypes_gnid_range_map, id_lookup, node_features)
    logging.info(f'[Rank: {rank}] Done with node features exchange.')

    node_data = gen_node_data(rank, world_size, id_lookup, ntid_ntype_map, schema_map)
    edge_data = exchange_edge_data(rank, world_size, edge_data)
    return node_data, rcvd_node_features, rcvd_global_nids, edge_data

def read_dataset(rank, world_size, id_lookup, params, schema_map):
    """
    This function gets the dataset and performs post-processing on the data which is read from files.
    Additional information(columns) are added to nodes metadata like owner_process, global_nid which 
    are later used in processing this information. For edge data, which is now a dictionary, we add new columns
    like global_edge_id and owner_process. Augmenting these data structure helps in processing these data structures
    when data shuffling is performed. 

    Parameters:
    -----------
    rank : int
        rank of the current process
    world_size : int
        total no. of processes instantiated
    id_lookup : instance of class DistLookupService
       Distributed lookup service used to map global-nids to respective partition-ids and 
       shuffle-global-nids
    params : argparser object 
        argument parser object to access command line arguments
    schema_map : dictionary
        dictionary created by reading the input graph metadata json file

    Returns : 
    ---------
    dictionary
        in which keys are node-type names and values are are tuples representing the range of ids
        for nodes to be read by the current process
    dictionary
        node features which is a dictionary where keys are feature names and values are feature
        data as multi-dimensional tensors 
    dictionary
        in which keys are node-type names and values are triplets. Each triplet has node-feature name
        and the starting and ending type ids of the node-feature data read from the corresponding
        node feature data file read by current process. Each node type may have several features and
        hence each key may have several triplets.
    dictionary
        edge data information is read from edges.txt and additional columns are added such as 
        owner process for each edge. 
    dictionary
        edge features which is also a dictionary, similar to node features dictionary
    """
    edge_features = {}
    #node_tids, node_features, edge_datadict, edge_tids
    node_tids, node_features, node_feat_tids, edge_data, edge_tids = \
        get_dataset(params.input_dir, params.graph_name, rank, world_size, schema_map)
    logging.info(f'[Rank: {rank}] Done reading dataset deom {params.input_dir}')

    edge_data = augment_edge_data(edge_data, id_lookup, edge_tids, rank, world_size)
    logging.info(f'[Rank: {rank}] Done augmenting edge_data: {len(edge_data)}, {edge_data[constants.GLOBAL_SRC_ID].shape}')

    return node_tids, node_features, node_feat_tids, edge_data, edge_features

def gen_dist_partitions(rank, world_size, params):
    """
    Function which will be executed by all Gloo processes to begin execution of the pipeline. 
    This function expects the input dataset is split across multiple file format. 

    Input dataset and its file structure is described in metadata json file which is also part of the 
    input dataset. On a high-level, this metadata json file contains information about the following items
    a) Nodes metadata, It is assumed that nodes which belong to each node-type are split into p files
       (wherer `p` is no. of partitions). 
    b) Similarly edge metadata contains information about edges which are split into p-files. 
    c) Node and Edge features, it is also assumed that each node (and edge) feature, if present, is also
       split into `p` files.

    For example, a sample metadata json file might be as follows: :
    (In this toy example, we assume that we have "m" node-types, "k" edge types, and for node_type = ntype0-name
     we have two features namely feat0-name and feat1-name. Please note that the node-features are also split into 
     `p` files. This will help in load-balancing during data-shuffling phase).

    Terminology used to identify any particular "id" assigned to nodes, edges or node features. Prefix "global" is
    used to indicate that this information is either read from the input dataset or autogenerated based on the information
    read from input dataset files. Prefix "type" is used to indicate a unique id assigned to either nodes or edges. 
    For instance, type_node_id means that a unique id, with a given node type,  assigned to a node. And prefix "shuffle" 
    will be used to indicate a unique id, across entire graph, assigned to either a node or an edge. For instance, 
    SHUFFLE_GLOBAL_NID means a unique id which is assigned to a node after the data shuffle is completed. 

    Some high-level notes on the structure of the metadata json file. 
    1. path(s) mentioned in the entries for nodes, edges and node-features files can be either absolute or relative. 
       if these paths are relative, then it is assumed that they are relative to the folder from which the execution is
       launched. 
    2. The id_startx and id_endx represent the type_node_id and type_edge_id respectively for nodes and edge data. This 
       means that these ids should match the no. of nodes/edges read from any given file. Since these are type_ids for 
       the nodes and edges in any given file, their global_ids can be easily computed as well. 

    {
        "graph_name" : xyz,
        "node_type" : ["ntype0-name", "ntype1-name", ....], #m node types
        "num_nodes_per_chunk" : [
            [a0, a1, ...a<p-1>], #p partitions
            [b0, b1, ... b<p-1>], 
            ....
            [c0, c1, ..., c<p-1>] #no, of node types
        ],
        "edge_type" : ["src_ntype:edge_type:dst_ntype", ....], #k edge types
        "num_edges_per_chunk" : [
            [a0, a1, ...a<p-1>], #p partitions
            [b0, b1, ... b<p-1>], 
            ....
            [c0, c1, ..., c<p-1>] #no, of edge types
        ],
        "node_data" : {
            "ntype0-name" : {
                "feat0-name" : {
                    "format" : {"name": "numpy"},
                    "data" :   [ #list of lists
                        ["<path>/feat-0.npy", 0, id_end0],
                        ["<path>/feat-1.npy", id_start1, id_end1],
                        ....
                        ["<path>/feat-<p-1>.npy", id_start<p-1>, id_end<p-1>]                
                    ]
                },
                "feat1-name" : {
                    "format" : {"name": "numpy"}, 
                    "data" : [ #list of lists
                        ["<path>/feat-0.npy", 0, id_end0],
                        ["<path>/feat-1.npy", id_start1, id_end1],
                        ....
                        ["<path>/feat-<p-1>.npy", id_start<p-1>, id_end<p-1>]                
                    ]
                }
            }
        },
        "edges": { #k edge types 
            "src_ntype:etype0-name:dst_ntype" : {
                "format": {"name" : "csv", "delimiter" : " "},
                "data" : [
                    ["<path>/etype0-name-0.txt", 0, id_end0], #These are type_edge_ids for edges of this type
                    ["<path>/etype0-name-1.txt", id_start1, id_end1],
                    ...,
                    ["<path>/etype0-name-<p-1>.txt", id_start<p-1>, id_end<p-1>]
                ]
            }, 
            ..., 
            "src_ntype:etype<k-1>-name:dst_ntype" : {
                "format": {"name" : "csv", "delimiter" : " "},
                "data" : [
                    ["<path>/etype<k-1>-name-0.txt", 0, id_end0],
                    ["<path>/etype<k-1>-name-1.txt", id_start1, id_end1],
                    ...,
                    ["<path>/etype<k-1>-name-<p-1>.txt", id_start<p-1>, id_end<p-1>]
                ]
            },         
        }, 
    }

    The function performs the following steps: 
    1. Reads the metis partitions to identify the owner process of all the nodes in the entire graph.
    2. Reads the input data set, each partitipating process will map to a single file for the edges, 
        node-features and edge-features for each node-type and edge-types respectively. Using nodes metadata
        information, nodes which are owned by a given process are generated to optimize communication to some
        extent.
    3. Now each process shuffles the data by identifying the respective owner processes using metis
        partitions. 
        a. To identify owner processes for nodes, metis partitions will be used. 
        b. For edges, the owner process of the destination node will be the owner of the edge as well. 
        c. For node and edge features, identifying the owner process is a little bit involved. 
            For this purpose, graph metadata json file is used to first map the locally read node features
            to their global_nids. Now owner process is identified using metis partitions for these global_nids
            to retrieve shuffle_global_nids. A similar process is used for edge_features as well. 
        d. After all the data shuffling is done, the order of node-features may be different when compared to 
            their global_type_nids. Node- and edge-data are ordered by node-type and edge-type respectively. 
            And now node features and edge features are re-ordered to match the order of their node- and edge-types. 
    4. Last step is to create the DGL objects with the data present on each of the processes. 
        a. DGL objects for nodes, edges, node- and edge- features. 
        b. Metadata is gathered from each process to create the global metadata json file, by process rank = 0. 

    Parameters:
    ----------
    rank : int
        integer representing the rank of the current process in a typical distributed implementation
    world_size : int
        integer representing the total no. of participating processes in a typical distributed implementation
    params : argparser object
        this object, key value pairs, provides access to the command line arguments from the runtime environment
    """
    global_start = timer()
    logging.info(f'[Rank: {rank}] Starting distributed data processing pipeline...')

    #init processing
    schema_map = read_json(os.path.join(params.input_dir, params.schema))

    #Initialize distributed lookup service for partition-id and shuffle-global-nids mappings
    #for global-nids
    _, global_nid_ranges = get_idranges(schema_map[constants.STR_NODE_TYPE], 
                                        schema_map[constants.STR_NUM_NODES_PER_CHUNK])
    id_map = dgl.distributed.id_map.IdMap(global_nid_ranges)
    id_lookup = DistLookupService(os.path.join(params.input_dir, params.partitions_dir),\
                                    schema_map[constants.STR_NODE_TYPE],\
                                    id_map, rank, world_size)

    ntypes_ntypeid_map, ntypes, ntypeid_ntypes_map = get_node_types(schema_map)
    logging.info(f'[Rank: {rank}] Initialized metis partitions and node_types map...')

    #read input graph files and augment these datastructures with
    #appropriate information (global_nid and owner process) for node and edge data
    node_tids, node_features, node_feat_tids, edge_data, edge_features = read_dataset(rank, world_size, id_lookup, params, schema_map)
    logging.info(f'[Rank: {rank}] Done augmenting file input data with auxilary columns')

    #send out node and edge data --- and appropriate features. 
    #this function will also stitch the data recvd from other processes
    #and return the aggregated data
    ntypes_gnid_range_map = get_gnid_range_map(node_tids)
    node_data, rcvd_node_features, rcvd_global_nids, edge_data  = \
                    exchange_graph_data(rank, world_size, node_features, node_feat_tids, \
                                        edge_data, id_lookup, ntypes_ntypeid_map, ntypes_gnid_range_map, \
                                        ntypeid_ntypes_map, schema_map)
    logging.info(f'[Rank: {rank}] Done with data shuffling...')

    #sort node_data by ntype
    idx = node_data[constants.NTYPE_ID].argsort()
    for k, v in node_data.items():
        node_data[k] = v[idx]
    logging.info(f'[Rank: {rank}] Sorted node_data by node_type')

    #resolve global_ids for nodes
    assign_shuffle_global_nids_nodes(rank, world_size, node_data)
    logging.info(f'[Rank: {rank}] Done assigning global-ids to nodes...')

    #shuffle node feature according to the node order on each rank. 
    for ntype_name in ntypes:
        featnames = get_ntype_featnames(ntype_name, schema_map)
        for featname in featnames:
            #if a feature name exists for a node-type, then it should also have 
            #feature data as well. Hence using the assert statement.
            assert(ntype_name+'/'+featname in rcvd_global_nids)
            global_nids = rcvd_global_nids[ntype_name+'/'+featname]

            common, idx1, idx2 = np.intersect1d(node_data[constants.GLOBAL_NID], global_nids, return_indices=True)
            shuffle_global_ids = node_data[constants.SHUFFLE_GLOBAL_NID][idx1]
            feature_idx = shuffle_global_ids.argsort()
            rcvd_node_features[ntype_name+'/'+featname] = rcvd_node_features[ntype_name+'/'+featname][feature_idx]

    #sort edge_data by etype
    sorted_idx = edge_data[constants.ETYPE_ID].argsort()
    for k, v in edge_data.items():
        edge_data[k] = v[sorted_idx]

    shuffle_global_eid_start = assign_shuffle_global_nids_edges(rank, world_size, edge_data)
    logging.info(f'[Rank: {rank}] Done assigning global_ids to edges ...')

    #determine global-ids for edge end-points
    edge_data = lookup_shuffle_global_nids_edges(rank, world_size, edge_data, id_lookup, node_data)
    logging.info(f'[Rank: {rank}] Done resolving orig_node_id for local node_ids...')

    #create dgl objects here
    start = timer()
    num_nodes = 0
    num_edges = shuffle_global_eid_start
    graph_obj, ntypes_map_val, etypes_map_val, ntypes_ntypeid_map, etypes_map = create_dgl_object(\
            params.graph_name, params.num_parts, \
            schema_map, rank, node_data, edge_data, num_nodes, num_edges)
    write_dgl_objects(graph_obj, rcvd_node_features, edge_features, params.output, rank)

    #get the meta-data 
    json_metadata = create_metadata_json(params.graph_name, len(node_data[constants.NTYPE_ID]), len(edge_data[constants.ETYPE_ID]), \
                            rank, world_size, ntypes_map_val, \
                            etypes_map_val, ntypes_ntypeid_map, etypes_map, params.output)

    if (rank == 0):
        #get meta-data from all partitions and merge them on rank-0
        metadata_list = gather_metadata_json(json_metadata, rank, world_size)
        metadata_list[0] = json_metadata
        write_metadata_json(metadata_list, params.output, params.graph_name)
    else:
        #send meta-data to Rank-0 process
        gather_metadata_json(json_metadata, rank, world_size)
    end = timer()
    logging.info(f'[Rank: {rank}] Time to create dgl objects: {timedelta(seconds = end - start)}')

    global_end = timer()
    logging.info(f'[Rank: {rank}] Total execution time of the program: {timedelta(seconds = global_end - global_start)}')

def single_machine_run(params):
    """ Main function for distributed implementation on a single machine

    Parameters:
    -----------
    params : argparser object
        Argument Parser structure with pre-determined arguments as defined
        at the bottom of this file.
    """
    log_params(params)
    processes = []
    mp.set_start_method("spawn")

    #Invoke `target` function from each of the spawned process for distributed
    #implementation
    for rank in range(params.world_size):
        p = mp.Process(target=run, args=(rank, params.world_size, gen_dist_partitions, params))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def run(rank, world_size, func_exec, params, backend="gloo"):
    """
    Init. function which is run by each process in the Gloo ProcessGroup

    Parameters:
    -----------
    rank : integer
        rank of the process
    world_size : integer
        number of processes configured in the Process Group
    proc_exec : function name
        function which will be invoked which has the logic for each process in the group
    params : argparser object
        argument parser object to access the command line arguments
    backend : string
        string specifying the type of backend to use for communication
    """
    os.environ["MASTER_ADDR"] = '127.0.0.1'
    os.environ["MASTER_PORT"] = '29500'

    #create Gloo Process Group
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timedelta(seconds=5*60))

    #Invoke the main function to kick-off each process
    func_exec(rank, world_size, params)

def multi_machine_run(params):
    """
    Function to be invoked when executing data loading pipeline on multiple machines

    Parameters:
    -----------
    params : argparser object
        argparser object providing access to command line arguments.
    """
    rank = int(os.environ["RANK"])

    #init the gloo process group here.
    dist.init_process_group(
            backend="gloo",
            rank=rank,
            world_size=params.world_size,
            timeout=timedelta(seconds=constants.GLOO_MESSAGING_TIMEOUT))
    logging.info(f'[Rank: {rank}] Done with process group initialization...')

    #invoke the main function here.
    gen_dist_partitions(rank, params.world_size, params)
    logging.info(f'[Rank: {rank}] Done with Distributed data processing pipeline processing.')
