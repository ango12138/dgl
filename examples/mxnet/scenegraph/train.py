import dgl
import mxnet as mx
import numpy as np
import logging, time
from operator import itemgetter
from mxnet import nd, gluon
from mxnet.gluon import nn
from dgl.utils import toindex
from dgl.nn.mxnet import GraphConv
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Pad
from gluoncv.utils import makedirs, LRSequential, LRScheduler

from model import EdgeGCN, faster_rcnn_resnet101_v1d_custom
from utils import *
from data import *

filehandler = logging.FileHandler('output.log')
streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

# Hyperparams
num_gpus = 1
batch_size = num_gpus * 8
ctx = [mx.gpu(i+1) for i in range(num_gpus)]
nepoch = 10
N_relations = 50
N_objects = 150
save_dir = 'params'
makedirs(save_dir)
batch_verbose_freq = 100
bbox_augmentation = False

net = EdgeGCN(in_feats=49, n_hidden=32, n_classes=N_relations,
              n_layers=2, activation=nd.relu, pretrained_base=False, ctx=ctx)
# net.initialize(ctx=ctx)
net.edge_mlp.initialize(ctx=ctx)
net.edge_link_mlp.initialize(ctx=ctx)
net.layers.initialize(ctx=ctx)
net_trainer = gluon.Trainer(net.collect_params(), 'adam', 
                            {'learning_rate': 0.01, 'wd': 0.00001})

# dataset and dataloader
vg_train = VGRelation(top_frequent_rel=N_relations, top_frequent_obj=N_objects,
                      balancing='weight', split='train')
logger.info('data loaded!')
train_data = gluon.data.DataLoader(vg_train, batch_size=batch_size, shuffle=True, num_workers=8*num_gpus,
                                   batchify_fn=dgl_mp_batchify_fn)
n_batches = len(train_data)

detector = faster_rcnn_resnet101_v1d_custom(classes=vg_train._obj_classes,
                                            pretrained_base=False, pretrained=False,
                                            additional_output=True)
params_path = 'faster_rcnn_resnet101_v1d_custom_best.params'
detector.load_parameters(params_path, ctx=ctx, ignore_extra=True, allow_missing=True)
for k, v in detector.collect_params().items():
    v.grad_req = 'null'

def get_data_batch(g_list, img_list, ctx_list):
    if g_list is None or len(g_list) == 0:
        return None, None
    n_gpu = len(ctx_list)
    size = len(g_list)
    if size < n_gpu:
        raise Exception("too small batch")
    step = size // n_gpu
    G_list = [g_list[i*step:(i+1)*step] if i < n_gpu - 1 else g_list[i*step:size] for i in range(n_gpu)]
    img_list = [img_list[i*step:(i+1)*step] if i < n_gpu - 1 else img_list[i*step:size] for i in range(n_gpu)]

    for G_slice, ctx in zip(G_list, ctx_list):
        for G in G_slice:
            G.ndata['bbox'] = G.ndata['bbox'].as_in_context(ctx)
            G.ndata['node_class_ids'] = G.ndata['node_class_ids'].as_in_context(ctx)
            G.ndata['node_class_vec'] = G.ndata['node_class_vec'].as_in_context(ctx)
            G.edata['classes'] = G.edata['classes'].as_in_context(ctx)
            G.edata['link'] = G.edata['link'].as_in_context(ctx)
            G.edata['weights'] = G.edata['weights'].expand_dims(1).as_in_context(ctx)
    img_list = [img.as_in_context(ctx) for img in img_list]
    return G_list, img_list

L_link = gluon.loss.SoftmaxCELoss()
L_rel = gluon.loss.SoftmaxCELoss()
# L_cls = gluon.loss.SoftmaxCELoss()

train_metric = mx.metric.Accuracy(name='rel')
train_metric_top5 = mx.metric.TopKAccuracy(5, name='rel_top')
train_metric_node = mx.metric.Accuracy(name='obj')
train_metric_node_top5 = mx.metric.TopKAccuracy(5, name='obj_top')
train_metric_f1 = mx.metric.F1(name='link_f1')
train_metric_auc = AUCMetric(name='link_auc')
train_metric_r100 = PredCls(100, iou_thresh=0.9)

for epoch in range(nepoch):
    loss_val = 0
    tic = time.time()
    btic = time.time()
    train_metric.reset()
    train_metric_top5.reset()
    train_metric_node.reset()
    train_metric_node_top5.reset()
    train_metric_f1.reset()
    train_metric_auc.reset()
    train_metric_r100.reset()
    if epoch == 8:
        net_trainer.set_learning_rate(net_trainer.learning_rate*0.1)
        # cls_trainer.set_learning_rate(cls_trainer.learning_rate*0.1)
    for i, (G_list, img_list) in enumerate(train_data):
        G_list, img_list = get_data_batch(G_list, img_list, ctx)
        if G_list is None or img_list is None:
            continue

        loss = []
        detector_res_list = []
        G_batch = []
        bbox_pad = Pad(axis=(0))
        loss_rel_val = 0
        loss_link_val = 0
        # loss_cls_val = 0
        with mx.autograd.record():
            for G_slice, img in zip(G_list, img_list):
                cur_ctx = img.context
                bbox_list = [G.ndata['bbox'] for G in G_slice]
                bbox_stack = bbox_pad(bbox_list).as_in_context(cur_ctx)
                if bbox_augmentation:
                    dis = nd.random.normal(0, 5, shape=bbox_stack.shape,
                                           ctx=bbox_stack.context)
                    bbox_stack += dis

                bbox, spatial_feat, cls_pred = detector((img, bbox_stack))
                G_batch.append(build_graph_gt(G_slice, img, bbox,
                                              spatial_feat, cls_pred,
                                              training=True))

            G_batch = [net(G) for G in G_batch]

            for G_slice, G_pred, img in zip(G_list, G_batch, img_list):
                if G_pred is None or G_pred.number_of_nodes() == 0:
                    continue
                G_gt = dgl.batch(G_slice)
                loss_rel = L_rel(G_pred.edata['preds'], G_gt.edata['classes'], G_gt.edata['link'])
                loss_link = L_link(G_pred.edata['link_preds'], G_gt.edata['link'], G_gt.edata['weights'])
                # loss_cls = L_cls(G_pred.ndata['node_class_pred'], G_gt.ndata['node_class_ids'])
                # loss.append(loss_rel.sum() + loss_link.sum() + loss_cls.sum())
                loss.append(loss_rel.sum() + loss_link.sum())
                loss_rel_val += loss_rel.sum().asscalar() / num_gpus
                loss_link_val += loss_link.sum().asscalar() / num_gpus
                # loss_cls_val += loss_cls.sum().asscalar() / num_gpus

        for l in loss:
            l.backward()
        net_trainer.step(batch_size)
        # cls_trainer.step(batch_size)
        # loss_val += sum([l.mean().asscalar() for l in loss]) / num_gpus
        for G_slice, G_pred, img_slice in zip(G_list, G_batch, img_list):
            for G_gt, G_pred_one in zip(G_slice, dgl.unbatch(G_pred)):
                if G_pred_one is None or G_pred_one.number_of_nodes() == 0:
                    continue
                link_ind = np.where(G_gt.edata['link'].asnumpy() == 1)[0]
                if len(link_ind) == 0:
                    continue
                train_metric.update([G_gt.edata['classes'][link_ind]], [G_pred_one.edata['preds'][link_ind]])
                train_metric_top5.update([G_gt.edata['classes'][link_ind]], [G_pred_one.edata['preds'][link_ind]])
                train_metric_node.update([G_gt.ndata['node_class_ids']], [G_pred_one.ndata['node_class_pred']])
                train_metric_node_top5.update([G_gt.ndata['node_class_ids']], [G_pred_one.ndata['node_class_pred']])
                train_metric_f1.update([G_gt.edata['link']], [G_pred_one.edata['link_preds']])
                train_metric_auc.update([G_gt.edata['link']], [G_pred_one.edata['link_preds']])
                gt_objects, gt_triplet = extract_gt(G_gt, img_slice.shape[2:4])
                pred_objects, pred_triplet = extract_pred(G_pred)
                train_metric_r100.update(gt_triplet, pred_triplet)
        if (i+1) % batch_verbose_freq == 0:
            # print_txt = 'Epoch[%d] Batch[%d/%d], time: %d, loss_rel=%.4f loss_link=%.4f loss_cls=%.4f '%\
            print_txt = 'Epoch[%d] Batch[%d/%d], time: %d, loss_rel=%.4f loss_link=%.4f '%\
                (epoch, i, n_batches, int(time.time() - btic),
                loss_rel_val / (i+1), loss_link_val / (i+1))#, loss_cls_val / (i+1))
            metric_list = [train_metric, train_metric_top5, train_metric_node, train_metric_node_top5,
                           train_metric_f1, train_metric_auc, train_metric_r100]
            for metric in metric_list:
                metric_name, metric_val = metric.get()
                print_txt +=  '%s=%.4f '%(metric_name, metric_val)
            logger.info(print_txt)
            btic = time.time()
    # print_txt = 'Epoch[%d] Batch[%d/%d], time: %d, loss_rel=%.4f, loss_link=%.4f, loss_cls=%.4f,'%\
    print_txt = 'Epoch[%d], time: %d, loss_rel=%.4f, loss_link=%.4f,'%\
        (epoch, int(time.time() - btic),
        loss_rel_val / (i+1), loss_link_val / (i+1))#, loss_cls_val / (i+1))
    metric_list = [train_metric, train_metric_top5, train_metric_node, train_metric_node_top5,
                   train_metric_f1, train_metric_auc, train_metric_r100]
    for metric in metric_list:
        metric_name, metric_val = metric.get()
        print_txt +=  '%s=%.4f '%(metric_name, metric_val)
    logger.info(print_txt)
    net.save_parameters('%s/model-%d.params'%(save_dir, epoch))
    detector.class_predictor.save_parameters('%s/class_predictor-%d.params'%(save_dir, epoch))
