import matplotlib.pyplot as plt
import pickle
import numpy as np
from experiments.transformations.utils.misc import project_path, get_fulltrain_strings_shapenet
import seaborn as sn
import sty

sn.set(style="dark")
sn.axes_style()
size_text = 20


sn.set_style('dark', {'axes.edgecolor': 'black',
                      'xtick.bottom': True,
                      'xtick.top': True,
                      'ytick.left': True,
                      'ytick.right': True,
                      'xtick.direction': 'in',
                      'ytick.direction': 'in',
                      'grid.color': "1",
                      'grid.linestyle': '-',
                      "axes.facecolor": "0.9"})

def get_file(pt_transf, pt_objs, transf, seed, dataset, network_name):
    pt_num_v = 50 if 'v' in pt_transf else 1

    if pt_transf == 'vanilla' or pt_transf == 'ImageNet':
        name_net = pt_transf
    else:
        _, _, _, name_net = get_fulltrain_strings_shapenet(pt_transf, pt_objs, pt_num_v, 'set1', mat=0, network_name='vgg11bn', seed=seed)
    return f'./results/transformations/activation_distance/{dataset}/{network_name}/{folder}/cossim_{transf}_{name_net}.pickle'

def get_values(pt_transf, transf, objs=100, dataset='ShapeNet', network_name=None, seed=None):
    try:
        ff = pickle.load(open(get_file(pt_transf, objs, transf, seed, dataset, network_name), 'rb'))
    except FileNotFoundError as err:
        print(sty.fg.yellow + f'NOT FOUND: {err.filename}' + sty.rs.fg)
        return None, None
    mean_acc = ff['acc_net']
    return mean_acc


folder = 'nway_test'
all_objs = []
seed = [1, 2, 3]
network_list = ['alexnet', 'resnet18', 'resnet50', 'vgg11bn', 'vgg19bn', 'googlenet', 'densenet201', 'samediffnet']

def get_acc(dataset_name, seed):
    untransformed = {}
    transformed = {}
    for network_name in network_list:
        untransformed[network_name] = np.array([get_values('', tt, 250, dataset_name, network_name, seed) for tt in ['v', 't', 'r', 's', 'b', 'c']])
        transformed[network_name] = np.array([get_values(tt, tt, 250, dataset_name, network_name, seed) for tt in ['v', 't', 'r', 's', 'b', 'c']])
    return untransformed, transformed

tr_ETH = [get_acc('ETH_grayscale_darkened_nobkg', s)[1] for s in [1,2,3]]
tr_shapenet = [get_acc('ShapeNet', s)[1] for s in [1, 2, 3]]


##

diff = np.array([np.array([tr_ETH[s][n]-tr_shapenet[s][n] for n in network_list]) for s in [0, 1, 2]])
diff.shape
m = np.mean(diff, axis=0)
s = np.std(diff, axis=0)
np.mean(diff)  # drop across all networks and transformation

plt.close('all')
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

space_between_bars = 0.25
width_bar = 0.20
values = m
span = space_between_bars * len(values)
values_mean = m
values_std = s
fig, ax = plt.subplots(1, 1, figsize=(22, 5), sharex=True, sharey=True)
plt.rcParams["figure.figsize"] = (25, 5)
x = np.array(np.arange(0, 3*len(values[0]), 3))

for i in range(len(values_mean)):
    ax.bar(x - span / 2 + space_between_bars / 2 + space_between_bars * i, values_mean[i],  color=color_cycle[i], width=width_bar, yerr=s[i] , error_kw=dict(lw=3, capsize=4, capthick=3))


ax.legend(prop={'size': 20}, bbox_to_anchor=(-0.1, 1.18), ncol=8, loc='upper left')
plt.axhline(0, linestyle='--', color='k', lw=2)
plt.xticks(fontsize=size_text)
plt.yticks(fontsize=size_text)
ax.set_xticks(x)
ax.set_xticklabels(['viewpoint', 'translate', 'rotate', 'scale', 'brightness', 'contrast'], rotation=0)
plt.grid(True)
plt.subplots_adjust(top=0.88,
                    bottom=0.14,
                    left=0.055,
                    right=0.9,
                    hspace=0.185,
                    wspace=0.185)

plt.ylabel(r"$\Delta$ Accuracy", size=20)

save_path = './results/transformations/figures/single_figs/'
plt.savefig(save_path + f'nway_plot_diff_eth_shapenet.svg', format='svg')

##

