import os
import argparse
import time

import dgl
from dgl.contrib import KVClient

import torch as th

partition = th.tensor([0,0,1,1,2,2,3,3])

ID = []
ID.append(th.tensor([0,1]))
ID.append(th.tensor([2,3]))
ID.append(th.tensor([4,5]))
ID.append(th.tensor([6,7]))

DATA = []
DATA.append(th.tensor([[1.,1.,1.,],[1.,1.,1.,]]))
DATA.append(th.tensor([[2.,2.,2.,],[2.,2.,2.,]]))
DATA.append(th.tensor([[3.,3.,3.,],[3.,3.,3.,]]))
DATA.append(th.tensor([[4.,4.,4.,],[4.,4.,4.,]]))

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--ip_config', type=str, default='ip_config.txt',
                          help='IP configuration file of kvstore.')
        self.add_argument('--num_worker', type=int, default=2,
                          help='Number of worker (client nodes) on single-machine.')

def start_client(args):
    """Start client
    """
    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)

    my_client = KVClient(server_namebook=server_namebook)

    my_client.connect()

    if my_client.get_id() % args.num_worker == 0:
        my_client.set_partition_book(name='entity_embed', partition_book=partition)
    else:
        time.sleep(3)
        my_client.set_partition_book(name='entity_embed')

    for i in range(4):
        my_client.push(name='entity_embed', id_tensor=ID[i], data_tensor=DATA[i])

    my_client.barrier()

    if my_client.get_id() % args.backup_count == 0:
        res = my_client.pull(name='entity_embed', id_tensor=th.tensor([0,1,2,3,4,5,6,7]))
        print(res)

        my_client.shut_down()


if __name__ == '__main__':
    args = ArgParser().parse_args()
    start_client(args)