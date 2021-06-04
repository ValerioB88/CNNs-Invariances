import matplotlib
import matplotlib.pyplot as plt
import pickle
import numpy as np
from experiments.transformations.utils.misc import get_fulltrain_strings_shapenet
import seaborn as sn
import sty

sn.set(style="dark")
sn.axes_style()
size_text = 20


def map_string_to_label(str):
    if str == 'vanilla':
        return 'Untrained'
    if str == 'ImageNet':
        return 'ImageNet'
    if str == '':
        return 'No Transformation'
    if str == 'r':
        return 'Rotation'
    if str == 's':
        return 'Scale'
    if str == 't':
        return 'Translate'
    if str == 'b':
        return 'Brightness'
    if str == 'c':
        return 'Contrast'
    if str == 'vtrsbc':
        return 'All'
    if str == 'v':
        return 'Viewpoint'
    else:
        return str


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
    return f'./results/transformations/activation_distance/ShapeNet/{network_name}/{folder}/cossim_{transf}_{name_net}.pickle'

##
plt.close('all')
folder = 'nway_test'
all_objs = []
seed = [1, 2, 3]
def get_values(pt_transf, transf, objs=100):
    result_seed = []
    for s in seed:
        try:
            result_seed.append(pickle.load(open(get_file(pt_transf, objs, transf, s), 'rb')))
        except FileNotFoundError as err:
            print(sty.fg.yellow + f'NOT FOUND: {err.filename}' + sty.rs.fg)
    mean_acc = np.nanmean([f['acc_net'] for f in result_seed])
    std_acc = np.nanstd([f['acc_net'] for f in result_seed])
    return mean_acc, std_acc


untransformed = {}
transformed = {}
obj_list = [5, 50, 100, 250, 500]
network_list = ['alexnet', 'resnet18', 'resnet50', 'vgg11bn', 'vgg19bn', 'googlenet', 'densenet201', 'samediffnet']
# network_list = ['vgg19bn']
transformation_list = ['v', 't', 'r', 's', 'b', 'c']
for network_name in network_list:
    untransformed.update({network_name: []})
    transformed.update({network_name: []})
    for tt in transformation_list:
        untransformed[network_name].append([get_values('', tt, obj) for obj in obj_list])
        transformed[network_name].append([get_values(tt, tt, obj) for obj in obj_list])

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']


##
plt.close('all')
# plt.rcParams["figure.figsize"] = (20, 7)
fig = plt.figure(figsize=(20, 5))
for i, t in enumerate(transformation_list):
    ax = plt.subplot2grid((1, 6), (0, i), colspan=1)

    for idx, n in enumerate(network_list):
        def plot(x, **kwargs):
            m = x[:, 0]
            s = x[:, 1]
            # plt.plot(obj_list, m, lw=2, color=color_cycle[idx], **kwargs)
            plt.errorbar(obj_list, m, yerr=s, color=color_cycle[idx], elinewidth=2, capsize=5, **kwargs)
            # plt.fill_between(obj_list, m - s, m + s, alpha=0.1, color=color_cycle[idx])
        plot(np.array(untransformed[n][i]), linestyle='--')
        plot(np.array(transformed[n][i]), linestyle='-')

    plt.ylim([0.15, 1.01])
    plt.axhline(0.2, color='k', linestyle='--', lw=3)
    plt.gca().set_xscale('log')
    plt.xlim([4, 600])
    plt.gca().set_xticks([5, 25, 100, 500])
    plt.gca().set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks(fontsize=size_text)
    plt.yticks(fontsize=size_text)
    if i != 0:
        plt.gca().set_yticklabels([])
    else:
        plt.ylabel('5ACT Accuracy', size=size_text)

    plt.grid(True)
    plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.title(map_string_to_label(t), size=size_text)

fig.text(0.5, 0.04, 'Number of Trained Objects per class', ha='center', size=size_text)
plt.tight_layout()
plt.subplots_adjust(top=0.934,
                    bottom=0.160,
                    left=0.048,
                    right=0.983,
                    hspace=0.2,
                    wspace=0.218)

save_path = './results/transformations/figures/single_figs/'
plt.savefig(save_path + 'multi_objs_acc.svg', format='svg')

##
plt.rcParams["figure.figsize"] = (10, 5)

for idx, network in enumerate(network_list):
    ax = plt.subplot2grid((1, len(network_list)), (0, idx), colspan=1)

    for i, _ in enumerate(transformed[network]):
        def plot(x, **kwargs):
            m = np.array([i[0] for i in x[network][i]])
            s = np.array([i[1] for i in x[network][i]])
            plt.plot(obj_list, m, lw=3, color=color_cycle[i], **kwargs)
            plt.fill_between(obj_list, m - s, m + s, alpha=0.2, color=color_cycle[i])
        plot(transformed, linestyle='-')
        plot(untransformed, linestyle='--')

    plt.ylim([0, 1.01])
    plt.gca().set_xscale('log')
    plt.xlim([5, 500])
    plt.gca().set_xticks([5, 25, 100, 500])
    plt.gca().set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xticks(fontsize=size_text)
    plt.yticks(fontsize=size_text)
    plt.grid(True)
    plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.title(network)


##


##

