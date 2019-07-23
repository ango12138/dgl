# -*- coding:utf-8 -*-
"""Sample training code
"""

import torch as th
import torch.nn as nn
from sch import SchNetModel
from mgcn import MGCNModel
from torch.utils.data import DataLoader
from Alchemy_dataset import TencentAlchemyDataset, batcher


def train(model="sch", epochs=80, device=th.device("cpu")):
    alchemy_dataset = TencentAlchemyDataset()
    alchemy_loader = DataLoader(dataset=alchemy_dataset,
                                batch_size=20,
                                collate_fn=batcher(device),
                                shuffle=False,
                                num_workers=0)

    if model == "sch":
        model = SchNetModel(norm=True, output_dim=12)
    elif model == "mgcn":
        model = MGCNModel(norm=True, output_dim=12)

    model.set_mean_std(alchemy_dataset.mean, alchemy_dataset.std)
    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    optimizer = th.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):

        w_loss, w_mae = 0, 0
        model.train()

        for batch in alchemy_loader:

            res = model(batch.graph)
            loss = loss_fn(res, batch.label)
            mae = MAE_fn(res, batch.label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            w_mae += mae.detach().item()
            w_loss += loss.detach().item()

        print("Epoch {:2d}, loss: {:.7f}, mae: {:.7f}".format(
            epoch, w_loss, w_mae))


if __name__ == "__main__":
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    train("sch", 80, device)
