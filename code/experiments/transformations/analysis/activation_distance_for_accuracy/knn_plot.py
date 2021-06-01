import matplotlib.pyplot as plt
import pickle
import numpy as np
from experiments.transformations.utils.misc import get_fulltrain_strings_shapenet

def get_file(pt_transf, pt_objs, transf):
    pt_num_v = 50 if 'v' in pt_transf else 1

    if pt_transf == 'vanilla' or pt_transf == 'ImageNet':
        name_net = pt_transf
        folder = 'others'
    else:
        _, _, _, name_net = get_fulltrain_strings_shapenet(pt_transf, pt_objs, pt_num_v, 'set1', mat=0, network_name='vgg11bn')
        folder = 'fulltrain'
    return f'./results/transformations/knn-test/ShapeNet/vgg11bn/{folder}/knn_T{transf}_objs3_vp1_shapeNetSet2_{name_net}.pickle'

all_objs = []

pt_objs_list = [1, 5,  25, 50, 100, 250, 500]
def plot_transf_objs(pt_transf, transf, ax, **kwargs):
    c = [pickle.load(open(get_file(pt_transf, pt_objs, transf), 'rb'))[f'acc_{metric}'] for pt_objs in pt_objs_list]
    ax.plot(pt_objs_list, c, label=transf, markersize=8, **kwargs)

def plot_all_transf(ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True)

    [plot_transf_objs(t, t, ax=ax, color=color_cycle[idx], **kwargs) for idx, t in enumerate(['t', 'r', 's'])]
    ax.grid('on')
    ax.set_ylim([0, 1.1])
    plt.setp(ax.get_yticklabels(), fontsize=20)
    plt.setp(ax.get_xticklabels(), fontsize=20)
    plt.axhline(1/30, color='k', linestyle='--', linewidth=2)

##
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.close('all')
fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True)
metric = 'cossim'
plot_all_transf(ax=ax, linestyle='-', marker='o')
plt.legend(prop={'size': 20})

##
# metric='euclidean'
# plot_all_transf(ax=ax, linestyle='--', marker='o')
# plt.legend(prop={'size': 20})

##
metric = 'cossim'
[plot_transf_objs('vanilla', t, ax=ax, color=color_cycle[idx], linestyle='--', linewidth=3) for idx, t in enumerate(['t', 'r', 's'])]

metric = 'cossim'
[plot_transf_objs('ImageNet', t, ax=ax, color=color_cycle[idx], linestyle='-.', linewidth=3) for idx, t in enumerate(['t', 'r', 's'])]

##


##

