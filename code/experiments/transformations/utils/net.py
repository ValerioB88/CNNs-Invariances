import torch
from torch import nn as nn

from framework_utils import make_cuda


def self_supervised_step(data, model, loss_fn, optimizer, use_cuda, loader, train):
    if train:
        optimizer.zero_grad()

    logs = {}
    x, lc, info = data
    lo = info['label_object']
    batch_size = len(lo) // 2
    # Watch out! Same objects is y=0, different is y=1
    y_matching_correct = torch.tensor([1 if i[0] != i[1] else 0 for i in lo.reshape(2, batch_size).T], dtype=torch.float)
    logs['images'] = x
    # Pairs {1_n, 2_n} are organized in this way: {1_1, 1_2, 1_3, 2_1, 2_2, 2_3}
    # If you want to debug, uncomment the following lines. The first window will have all the first elemenets of each pair, the second window the second elements:
    # framework_utils.imshow_batch(x[:batch_size], labels=[f'C{c}_O{o}' for c, o in zip(lc[:batch_size].numpy(), lo[:batch_size].numpy())], stats=loader.dataset.stats)
    # framework_utils.imshow_batch(x[batch_size:], labels=[f'C{c}_O{o}' for c, o in zip(lc[batch_size:].numpy(), lo[batch_size:].numpy())], stats=loader.dataset.stats)
    x = make_cuda(x, use_cuda)
    relation_scores = model((x, batch_size, 1, 1, use_cuda))
    rsm = make_cuda(relation_scores.squeeze(1), use_cuda)
    lb = make_cuda(y_matching_correct, use_cuda)
    loss = loss_fn(rsm, lb)
    y_matching_predicted = (rsm > 0.5).int()

    logs['output'] = relation_scores
    logs['labels'] = lo
    logs.update(info)

    if train:
        loss.backward()
        optimizer.step()
    return loss, y_matching_correct, y_matching_predicted, logs


class SameDifferentNetwork(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.backbone = net
        self.relation_module = nn.Sequential(nn.Linear(512, 1, bias=True),
                                             nn.Sigmoid())

    def forward(self, input):
        x, k, nSt, nFt, use_cuda = input
        emb_all = self.backbone(x)
        emb_candidates = emb_all[:k]  # this only works for nSt, nFt, nSc and nFc = 1!
        emb_trainings = emb_all[k:]
        return self.relation_module(torch.abs(emb_candidates - emb_trainings))