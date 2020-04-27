from modules import *
from loss import *
from optims import *
from dataset import *
from modules.config import *
#from modules.viz import *
import numpy as np
import argparse
import torch
from functools import partial
import torch.distributed as dist


# TODO: need refactor using real pytorch dataset
def run_epoch(epoch, train_data_iter, graph_pool, dataset, device, dev_rank, ndev, model, loss_compute, test_loss_compute, log_interval=10, is_train=True, log_f=None):
    with loss_compute:
        for i, g in enumerate(train_data_iter):
            with T.set_grad_enabled(is_train):
                output = model(g)
                tgt_y = g.tgt_y
                n_tokens = g.n_tokens
                loss = loss_compute(output, tgt_y, n_tokens)
            if i % log_interval == 0:
            #if True:
                print (i, loss)
                if log_f:
                    info = 'train,' + str(epoch) + ',' + str(i) + ',' + str(loss) + '\n'
                    log_f.write(info)
                    log_f.flush()
                # Run a random test set
                test_g = dataset.random_batch(graph_pool, mode='test', batch_size=4,
                                              device=device, dev_rank=dev_rank, ndev=ndev)
                model.eval()
                with T.set_grad_enabled(False):
                    output = model(test_g)
                    tgt_y = test_g.tgt_y
                    n_tokens = test_g.n_tokens
                    loss = test_loss_compute(output, tgt_y, n_tokens)
                    print ('test', loss)
                    if log_f:
                        info = 'test,' + str(epoch) + ',' + str(i) + ',' + str(loss) + '\n'
                        log_f.write(info)
                        log_f.flush()
                model.train(True)

    print('Epoch {} {}: Dev {} average loss: {}, accuracy {}'.format(
        epoch, "Training" if is_train else "Evaluating",
        dev_rank, loss_compute.avg_loss, loss_compute.accuracy))

def run(dev_id, args):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip=args.master_ip, master_port=args.master_port)
    world_size = args.ngpu
    torch.distributed.init_process_group(backend="nccl",
                                         init_method=dist_init_method,
                                         world_size=world_size,
                                         rank=dev_id)
    gpu_rank = torch.distributed.get_rank()
    assert gpu_rank == dev_id
    main(dev_id, args)

def main(dev_id, args):
    if dev_id == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:{}'.format(dev_id))
    # Create ckpt dir
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train_log_path = os.path.join(args.ckpt_dir, 'log.txt.train')
    print (train_log_path)
    train_log_f = open(train_log_path, 'w')
    test_log_path = os.path.join(args.ckpt_dir, 'log.txt.test')
    test_log_f = open(test_log_path, 'w')

    # Set current device
    th.cuda.set_device(device)
    # Prepare dataset
    dataset = get_dataset('face')
    criterion = torch.nn.NLLLoss()
    dim_model = 256
    # Build graph pool
    graph_pool = FaceGraphPool()
    # Create model
    model = make_face_model(N=args.N, dim_model=dim_model,
                       universal=args.universal)
    # Move model to corresponding device
    model, criterion = model.to(device), criterion.to(device)
    # Loss function
    if args.ngpu > 1:
        dev_rank = dev_id # current device id
        ndev = args.ngpu # number of devices (including cpu)
        loss_compute = partial(MultiGPULossCompute, criterion, args.ngpu,
                               args.grad_accum, model)
    else: # cpu or single gpu case
        dev_rank = 0
        ndev = 1
        loss_compute = partial(SimpleLossCompute, criterion, args.grad_accum)

    if ndev > 1:
        for param in model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= ndev

    # Optimizer
    model_opt = NoamOpt(dim_model, 0.1, 4000,
                        T.optim.Adam(model.parameters(), lr=3e-4,
                                     betas=(0.9, 0.98), eps=1e-9))

    # Train & evaluate
    for epoch in range(100):
        start = time.time()
        train_iter = dataset(graph_pool, mode='train', batch_size=args.batch,
                             device=device, dev_rank=dev_rank, ndev=ndev)

        model.train(True)
        run_epoch(epoch, train_iter, graph_pool, dataset, device, dev_rank, ndev, model,
                  loss_compute(opt=model_opt), loss_compute(opt=None), is_train=True, log_f=train_log_f)
        if dev_rank == 0:
            ckpt_path = os.path.join(args.ckpt_dir, 'ckpt.'+str(epoch)+'.pt')
            print (ckpt_path)
            torch.save(model.state_dict(), ckpt_path)


if __name__ == '__main__':
    np.random.seed(1111)
    argparser = argparse.ArgumentParser('training translation model')
    argparser.add_argument('--gpus', default='-1', type=str, help='gpu id')
    argparser.add_argument('--N', default=6, type=int, help='enc/dec layers')
    argparser.add_argument('--dataset', default='face', help='dataset')
    argparser.add_argument('--batch', default=128, type=int, help='batch size')
    argparser.add_argument('--ckpt-dir', default='.', type=str, help='checkpoint path')
    argparser.add_argument('--viz', action='store_true',
                           help='visualize attention')
    argparser.add_argument('--universal', action='store_true',
                           help='use universal transformer')
    argparser.add_argument('--master-ip', type=str, default='127.0.0.1',
                           help='master ip address')
    argparser.add_argument('--master-port', type=str, default='12345',
                           help='master port')
    argparser.add_argument('--grad-accum', type=int, default=1,
                           help='accumulate gradients for this many times '
                                'then update weights')
    args = argparser.parse_args()
    print(args)

    devices = list(map(int, args.gpus.split(',')))
    if len(devices) == 1:
        args.ngpu = 0 if devices[0] < 0 else 1
        main(devices[0], args)
    else:
        args.ngpu = len(devices)
        mp = torch.multiprocessing.get_context('spawn')
        procs = []
        for dev_id in devices:
            procs.append(mp.Process(target=run, args=(dev_id, args),
                                    daemon=True))
            procs[-1].start()
        for p in procs:
            p.join()

