from models import KEModel

import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd

import os
import logging
import time
import json

def load_model(logger, args, n_entities, n_relations, ckpt=None):
    model = KEModel(args, args.model_name, n_entities, n_relations,
                    args.hidden_dim, args.gamma,
                    double_entity_emb=args.double_ent, double_relation_emb=args.double_rel)
    if ckpt is not None:
        # TODO: loading model emb only work for genernal Embedding, not for ExternalEmbedding
        if args.gpu >= 0:
            model.load_parameters(ckpt, ctx=mx.gpu(args.gpu))
        else:
            model.load_parameters(ckpt, ctx=mx.cpu())

    logger.info('Load model {}'.format(args.model_name))
    return model

def load_model_from_checkpoint(logger, args, n_entities, n_relations, ckpt_path):
    model = load_model(logger, args, n_entities, n_relations)
    model.load_emb(ckpt_path, args.dataset)
    return model

def train(args, model, train_sampler, valid_samplers=None):
    if args.num_proc > 1:
        os.environ['OMP_NUM_THREADS'] = '1'
    logs = []

    for arg in vars(args):
        logging.info('{:20}:{}'.format(arg, getattr(args, arg)))

    start = time.time()
    for step in range(args.init_step, args.max_step):
        (pos_g, neg_g), neg_head = next(train_sampler)
        # TODO If the batch isn't divisible by negative sample size,
        # we need to pad them. Let's ignore them for now.
        if pos_g.number_of_edges() % args.neg_sample_size > 0:
            continue

        args.step = step
        with mx.autograd.record():
            loss, log = model.forward(pos_g, neg_g, neg_head, args.gpu)
        loss.backward()
        logs.append(log)
        model.update()

        if step % args.log_interval == 0:
            for k in logs[0].keys():
                v = sum(l[k] for l in logs) / len(logs)
                print('[Train]({}/{}) average {}: {}'.format(step, args.max_step, k, v))
            logs = []
            print(time.time() - start)
            start = time.time()

        if args.valid and step % args.eval_interval == 0 and step > 1 and valid_samplers is not None:
            start = time.time()
            test(args, model, valid_samplers, mode='Valid')
            print('test:', time.time() - start)
    # clear cache
    logs = []

def test(args, model, test_samplers, mode='Test'):
    logs = []

    for sampler in test_samplers:
        #print('Number of tests: ' + len(sampler))
        count = 0
        for pos_g, neg_g in sampler:
            # The last batch may not have the batch size expected.
            # Let's ignore it for now.
            if pos_g.number_of_edges() != args.batch_size_eval:
                continue

            model.forward_test(pos_g, neg_g, sampler.neg_head,
                                sampler.neg_sample_size, logs, args.gpu)

    metrics = {}
    if len(logs) > 0:
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

    for k, v in metrics.items():
        print('{} average {} at [{}/{}]: {}'.format(mode, k, args.step, args.max_step, v))
    for i in range(len(test_samplers)):
        test_samplers[i] = test_samplers[i].reset()
