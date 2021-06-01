import matplotlib.pyplot as plt
import pickle
import numpy as np
from experiments.transformations.utils.misc import get_fulltrain_strings_shapenet
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

def get_file(pt_transf, pt_objs, transf, seed):
    pt_num_v = 50 if 'v' in pt_transf else 1

    if pt_transf == 'vanilla' or pt_transf == 'ImageNet':
        name_net = pt_transf
    else:

        _, _, _, name_net = get_fulltrain_strings_shapenet(pt_transf, pt_objs, pt_num_v, 'set1', mat=0, network_name='vgg11bn', seed=seed)
    return f'./results/transformations/activation_distance/{dataset}/{network_name}/{folder}/cossim_{transf}_{name_net}.pickle'

##
folder = 'nway_test'
all_objs = []
seed = [1, 2, 3]
def get_values(pt_transf, transf, objs=100):
    try:
        ff = [pickle.load(open(get_file(pt_transf, objs, transf, s), 'rb')) for s in seed]
    except FileNotFoundError as err:
        print(sty.fg.yellow + f'NOT FOUND: {err.filename}' + sty.rs.fg)
        return None, None
    mean_acc = np.mean([f['acc_net'] for f in ff])
    std_acc = np.std([f['acc_net'] for f in ff])
    return mean_acc, std_acc


# dataset = 'ETH_grayscale_darkened_nobkg'
dataset = 'ShapeNet'

untransformed = {}
transformed = {}
network_list = ['alexnet', 'resnet18', 'resnet50', 'vgg11bn', 'vgg19bn', 'googlenet', 'densenet201', 'samediffnet']
# network_list = ['relationNet']
for network_name in network_list:
    untransformed[network_name] = np.array([get_values('', tt, 250) for tt in ['v', 't', 'r', 's', 'b', 'c']])
    transformed[network_name] = np.array([get_values(tt, tt, 250) for tt in ['v', 't', 'r', 's', 'b', 'c']])


[np.mean([v[t][0] for k, v in transformed.items()]) for t in range(6)]
[np.mean([v[t][0] for k, v in untransformed.items()]) for t in range(6)]

plt.close('all')
# color_cycle = sn.color_palette("hls",8)
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

space_between_bars = 0.25
width_bar = 0.20
labels = [k for k, v in transformed.items()]
values = [v for k, v in transformed.items()]
span = space_between_bars * len(values)
values_mean = [[i[0] for i in c] for c in values]
values_std = [[i[1] for i in c] for c in values]
# fig, ax = plt.subplots(1, 1, figsize=(22, 5), sharex=True, sharey=True)
plt.rcParams["figure.figsize"] = (25, 5)
ax = plt.subplot2grid((1, 5), (0, 0), colspan=4)
# x = np.arange(np.max([len(i) for i in values]))
x = np.array(np.arange(0, 3*len(values[0]), 3))

for i in range(len(values_mean)):
    ax.bar(x - span / 2 + space_between_bars / 2 + space_between_bars * i, values_mean[i],  color=color_cycle[i], width=width_bar, yerr=values_std[i], label=labels[i], error_kw=dict(lw=3, capsize=4, capthick=3))

labels = [k for k, v in untransformed.items()]
values = [v for k, v in untransformed.items()]
span = space_between_bars * len(values)
values_mean = [[i[0] for i in c] for c in values]
values_std = [[i[1] for i in c] for c in values]

for i in range(len(values_mean)):
    ax.plot(x - span / 2 + space_between_bars / 2 + space_between_bars * i, values_mean[i], 'o', color='k')
    ax.errorbar(x - span / 2 + space_between_bars / 2 + space_between_bars * i, values_mean[i], yerr=values_std[i],color='k', linestyle='')
ax.legend(prop={'size': 20}, bbox_to_anchor=(-0.1, 1.18), ncol=8, loc='upper left')

plt.axhline(0.2, color='k', linestyle='--', linewidth=2)
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


## CROSS TESTS - we only computed the data for the ShapeNet cross tests
if dataset == 'ShapeNet':
    ax = plt.subplot2grid((1,5), (0, 4), colspan=4)

    folder = 'nway_test_cross'
    untransformed = {}
    transformed = {}
    col = []
    network_name = 'vgg11bn'
    tv = get_values('t', 'v', 250)
    col.append(color_cycle[network_list.index(network_name)])

    network_name = 'googlenet'
    vt = get_values('v', 't', 250)
    col.append(color_cycle[network_list.index(network_name)])

    network_name = 'samediffnet'
    rt = get_values('r', 't', 250)
    col.append(color_cycle[network_list.index(network_name)])

    network_name = 'alexnet'
    sr = get_values('s', 'r', 250)
    col.append(color_cycle[network_list.index(network_name)])

    network_name = 'resnet50'
    bs = get_values('b', 's', 250)
    col.append(color_cycle[network_list.index(network_name)])

    network_name = 'densenet201'
    cb = get_values('c', 'b', 250)
    col.append(color_cycle[network_list.index(network_name)])

    # transformed[network_name] = np.array([get_values(tt, tt, 250) for tt in ['v', 't', 'r', 's', 'b', 'c']])
    # fig, ax = plt.subplots(1, 1, figsize=(5, 8), sharex=True, sharey=True)

    values = [tv, vt, rt, sr, bs, cb]
    x = np.arange(0, 1.5*len(values), 1.5)
    ax.bar(x, [i[0] for i in values], yerr=[i[1] for i in values], color=col,  error_kw=dict(lw=3, capsize=4, capthick=3))
    ax.set_xticks(x)

    plt.axhline(0.2, color='k', linestyle='--', linewidth=2)
    plt.xticks(fontsize=size_text-5)
    plt.yticks(fontsize=size_text)
    ax.set_yticklabels([])
    plt.grid(True)
    ax.set_xticklabels([r't$\rightarrow$v', r'v$\rightarrow$t', r'r$\rightarrow$t', r's$\rightarrow$r', r'b$\rightarrow$s', r'c$\rightarrow$b'], rotation=0)
    plt.ylim([0, 1])

save_path = './results/transformations/figures/single_figs/'
plt.savefig(save_path + f'nway_plot_{dataset}.svg', format='svg')

## Some more (not used cross comparison, still relevant)
network_name = 'vgg11bn'
get_values('t', 'c', 250)


network_name = 'densenet201'
cb = get_values('c', 'b', 250)