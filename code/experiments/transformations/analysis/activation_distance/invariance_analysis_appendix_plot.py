"""
Analysis of the results obtained in run_distances_fulltrain.ipynb. In particular, we plot the Invariance Analysis Figure.
You can select any number of networks, objects, etc.
"""
from experiments.transformations.analysis.activation_distance.activation_distance_utils import *
import framework_utils
import seaborn as sn
import pickle
import matplotlib.pyplot as plt
import sty


def rearrange_multiplots():
    axm = np.reshape(ax, (4, 4))
    cc = axm[:, 1]
    for i in range(len(cc)):
        axx = cc[i]
        pp = axx.get_position()
        pp.x0 -= 0.02
        axx.set_position(pp)

    cc = axm[:, 2]
    for i in range(len(cc)):
        axx = cc[i]
        pp = axx.get_position()
        pp.x1 += 0.02
        axx.set_position(pp)

    cc = [x for idx, x in enumerate(ax) if idx in range(0, 16, 2)]
    for idx, c in enumerate(cc):
        t = c.set_title(map_net_to_netstring(all_nets[idx]), size=size_text, loc='center')
        t.set_position((1.0, -1.55))


def map_net_to_netstring(s):
    if s == 'alexnet':
        return 'Alexnet'
    if s == 'resnet18':
        return 'ResNet-18'
    if s == 'resnet50':
        return 'ResNet-50'
    if s == 'googlenet':
        return 'GoogLeNet'
    if s == 'densenet201':
        return 'DenseNet-201'
    if s == 'vgg11bn':
        return 'VGG11'
    if s == 'vgg19bn':
        return 'VGG19'
    if s == 'samediffnet':
        return 'Same/Different Network'

sn.set(style="dark")
sn.axes_style()
sn.set_style('dark', {'axes.edgecolor': 'black',
                      'xtick.bottom': True,
                      'xtick.top': True,
                      'ytick.left': True,
                      'ytick.right': True,
                      'xtick.direction': 'in',
                      'ytick.direction': 'in',
                      'axes.facecolor': '.9'})


def get_filename(dataset, distance, pt_transf, transf, objs, vp, seed):
    evaluated_activation = 'ShapeNetSet2' if dataset == 'ShapeNet' else 'test'

    if pt_transf == 'ImageNet' or pt_transf == 'vanilla':
        return f'./results/transformations/activation_distance/{dataset}/vgg11bn/others/{distance}_{transf}_{evaluated_activation}_{pt_transf}.pickle'
    if dataset == 'ShapeNet':
        return f'./results/transformations/activation_distance/{dataset}/{network_name}/fulltrain/{distance}_{transf}_shapeNetSet2_' + \
               (f'FULLTRAIN_T{pt_transf}_objs{objs}_vp{vp}_mat0_trset1_S{seed}' if pt_transf != 'ImageNet' and pt_transf != 'vanilla' else 'S0') + '.pickle'
    if dataset == 'cifar10':
        return f'./results/transformations/activation_distance/{dataset}/{distance}_T{pt_transf}_{transf}' + \
               (f'_imgs{objs}' if pt_transf != 'ImageNet' and pt_transf != 'vanilla' else '') + '.pickle'


def plot_glob(pt_transf, transf, folder, distance, type_plot, ax=None, label=None):
    vp = 50 if 'v' in pt_transf else 1
    mean_seed = []
    for s in [1, 2, 3]:
        try:
            activationT = pickle.load(open(get_filename(folder, distance, pt_transf, transf, objs, vp, s), 'rb'))
        except FileNotFoundError as err:
            print(sty.fg.yellow + f'NOT FOUND: {err.filename}' + sty.rs.fg)
            continue
        activation_net_same_obj = activationT['distance_net_same_obj']
        activation_net_same_class_diff_objs = activationT['distance_net_same_class_diff_obj']
        activation_net_any_obj = activationT['distance_net_any_obj']
        activation_img = activationT['distance_img']
        x_values = activationT['x_values']

        mean_any_obj, std_any_obj, _ = DistanceActivation.get_average_cossim_across_classes(activation_net_any_obj)
        mean_same_class_diff_obj, std_same_class_diff_obj, _ = DistanceActivation.get_average_cossim_across_classes(activation_net_same_class_diff_objs)
        mean_act_img, std_act_img, _ = DistanceActivation.get_average_cossim_across_classes(activation_img)
        mean_same_obj, std_same_obj, _ = DistanceActivation.get_average_cossim_across_classes(activation_net_same_obj)

        I = mean_same_obj[layer_index]
        D = mean_any_obj[layer_index]
        D2 = mean_same_class_diff_obj[layer_index]
        G = (I - D) / (1 - D)

        if type_plot == 'image':
            # I forgot to do the comparisong with ITSELF at alpha level (rot=0, scale etc =1), which cossim is obviously 1 - but not having that values create ugly looking plots, so let's add it manually. We do this for each layer.
            idx = np.argmin(np.abs(x_values)) if transf == 'Rotate' else np.argmin(np.abs(x_values-1))
            x_values = np.insert(x_values, idx+1, 0 if transf == 'Rotate' else 1)
            mean_act_img = np.insert(mean_act_img, idx+1, 1)
            mean_seed.append(mean_act_img)
        elif type_plot == 'G':
            mean_seed.append(G)
        elif type_plot == 'I':
            mean_seed.append(I)
        elif type_plot == 'D':
            mean_seed.append(D)
        elif type_plot == 'D2':
            mean_seed.append(D2)

    v = np.nanmean(mean_seed, axis=0)
    std = np.nanstd(mean_seed, axis=0)
    if transf == 'Translate' or transf == 'VP':
        incl = (np.array(x_values[0] * 128 + 128/2) if transf == 'Translate' else np.array(x_values[0]))
        azi = (np.array(x_values[1] * 128 + 128/2) if transf == 'Translate' else np.array(x_values[1]))

        if transf == 'VP':
            v = np.hstack((v, v[azi == 0]))
            incl = np.hstack((incl, np.unique(incl)))
            azi = np.hstack((azi, np.repeat(359, len(np.unique(azi)))))  # np.unique(values[1])))
        ax, im = framework_utils.imshow_density((incl, azi, v),
                                                plot_args={'size_canvas': (128, 128) if transf == 'Translate' else (120, 360),
                                                           'interpolate': True}, ax=ax, vmin=0, vmax=1)
        if transf == 'VP':
            ax.plot(36, 75, marker='x', markersize=8, markeredgecolor='k', markeredgewidth=3)
            ax.set_axisbelow(True)
            ax.set_xticks([0, 90, 180, 270, 360])
            ax.set_yticks([30, 110])
            ax.set_ylim([0, 111+30])
            ax.set_xlim([0, 360])
        if transf == 'Translate':
            ax.plot(64, 64, marker='x', markersize=8, markeredgecolor='k', markeredgewidth=3)
            # ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_xlim([0, 124])
            ax.set_ylim([0, 124])
            # ticks = list(range(0, 125, 25))
            # ticks.append(124)

    else:
        ax.plot(x_values, v,
                linewidth=1 if type_plot == 'image' else 2,
                linestyle='--' if type_plot == 'D' else ('-' if type_plot == 'I' else ':'),
                label=label,
                color='k' if type_plot == 'image' else color_net[network_name],
                marker='o' if type_plot == 'image' else None,
                markersize=4)
        ax.fill_between(x_values, v - std, v + std, alpha=0.1,
                        color='k' if type_plot == 'image' else color_net[network_name])

    return True


def plot_set(pt_list, diffAct, type_plot, ax=None):
    multiax = False
    if ax is None:
        if diffAct == 'Translate' or diffAct == 'VP':
            if diffAct == 'Translate':
                size = (8.81, 4.1)
            else:
                size = (12.9, 2.45)
            fig, ax = plt.subplots(1, len(pt_list), figsize=size, sharex=True, sharey=True)
            axes = ax.flatten()
            multiax = True
        else:
            fig, ax = plt.subplots(1, 1, figsize=(8, 5.5), sharex=True, sharey=True)

    for idx, p in enumerate(pt_list):
        ax = axes[idx]  if multiax else ax
        f = plot_glob(p, diffAct, folder, metric, type_plot, ax=ax, label=map_string_to_label(network_name) if idx == 0 else None)
        if (diffAct == 'Translate' or diffAct == 'VP') and f:
            # ax.figure.colorbar(im)
            # plt.suptitle(network_name, size=size_text)
            plt.grid(False)
            plt.setp(ax.get_xticklabels(), fontsize=size_text)
            plt.setp(ax.get_yticklabels(), fontsize=size_text)


    if not (diffAct == 'Translate' or diffAct == 'VP') and f:
        ax.set_title(diffAct, size=size_text)
        # plt.grid('on')
        ax.grid(True)
        ax.set_yticks([0, 0.5, 1])
        plt.setp(ax.get_xticklabels(), fontsize=size_text)
        plt.setp(ax.get_yticklabels(), fontsize=size_text)
    # ax.set_xticklabels(fontsize=size_text)
    # ax.set_yticklabels(fontsize=size_text)


layer_index = -1

##
size_text = 20
adjust = True
ax = None
metric = 'cossim'
objs = 250
folder = 'ShapeNet'
save_path = './results/transformations/figures/single_figs/'
pt_list = ['']
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
all_nets = ['alexnet', 'resnet18', 'resnet50', 'vgg11bn', 'vgg19bn', 'googlenet',  'densenet201', 'samediffnet']
color_net = {k: v for k, v in zip(all_nets, color_cycle)}

# type = None
type = 'TRANSF'
## Rotate
fig, ax = plt.subplots(2, 2, figsize=(8*2, 5.5*2), sharex=False, sharey=True)
ax = ax.flatten()
i = 0
x = ['r'] if type == 'TRANSF' else ['']
tp = 'D'
for network_name in all_nets:
    plot_set(x, 'Rotate', type_plot=tp, ax=ax[0]); i += 1
tp = 'I'
for network_name in all_nets:
    plot_set(x, 'Rotate', type_plot=tp, ax=ax[0]); i += 1
tp = 'G'
for network_name in all_nets:
    plot_set(x, 'Rotate', type_plot=tp, ax=ax[0]); i += 1


ax[0].set_xlim(-180, 180)
ax[0].set_ylim(-0.2, 1.1)
ax[0].set_xticks([-180, -90, 0, 90, 180])

# ## Scale
i = 0
x = ['s'] if type == 'TRANSF' else ['']

tp = 'D'
for network_name in all_nets:
    plot_set(x, 'Scale', type_plot=tp, ax=ax[1]); i += 1
tp = 'I'
for network_name in all_nets:
    plot_set(x, 'Scale', type_plot=tp, ax=ax[1]); i += 1
tp = 'G'
for network_name in all_nets:
    plot_set(x, 'Scale', type_plot=tp, ax=ax[1]); i += 1

ax[1].set_xticks([0.2, 0.6, 1, 1.4, 1.8])
ax[1].set_xlim(0.2, 1.8)
ax[1].set_ylim(-0.2, 1.1)

# ## Brightness
i = 0
x = ['b'] if type == 'TRANSF' else ['']

tp = 'D'
for network_name in all_nets:
    plot_set(x, 'Brightness', type_plot=tp, ax=ax[2]); i += 1
tp = 'I'
for network_name in all_nets:
    plot_set(x, 'Brightness', type_plot=tp, ax=ax[2]); i += 1
tp = 'G'
for network_name in all_nets:
    plot_set(x, 'Brightness', type_plot=tp, ax=ax[2]); i += 1

ax[2].set_xticks([0.2, 0.6, 1, 1.4, 1.8])
ax[2].set_xlim(0.2, 1.8)
ax[2].set_ylim(-0.1, 1.1)

# #
# ## Contrast
i = 0
x = ['c'] if type == 'TRANSF' else ['']

tp = 'D'
for network_name in all_nets:
    plot_set(x, 'Contrast', type_plot=tp, ax=ax[3]); i += 1
tp = 'I'
for network_name in all_nets:
    plot_set(x, 'Contrast', type_plot=tp, ax=ax[3]); i += 1
tp = 'G'
for network_name in all_nets:
    plot_set(x, 'Contrast', type_plot=tp, ax=ax[3]); i += 1

ax[3].set_xticks([0.2, 0.6, 1, 1.4, 1.8])
ax[3].set_xlim(0.2, 1.8)
ax[3].set_ylim(-0.1, 1.1)


plt.savefig(save_path + f'SUPPL_rscb_{type}.svg', format='svg')


## VP
x = ['v'] if type == 'TRANSF' else ['']
i = 0
fig, ax = plt.subplots(4, 4, figsize=(21.50, 12.14), sharex=True, sharey=True)
ax = ax.flatten()
for network_name in all_nets:
    tp = 'I'
    plot_set(x, 'VP', type_plot=tp, ax=ax[i]); i += 1
    tp = 'D'
    plot_set(x, 'VP', type_plot=tp, ax=ax[i]); i += 1
    tp = 'G'
    plot_set(x, 'VP', type_plot=tp, ax=ax[i]);
    i += 1
rearrange_multiplots()
plt.savefig(save_path + f'VP_allnet_{type}.svg', format='svg')


## Translate
plt.close('all')
i = 0
x = ['t'] if type == 'TRANSF' else ['']
fig, ax = plt.subplots(3, 6, figsize=(19.89, 10.51), sharex=True, sharey=True)
ax = ax.flatten()
for network_name in all_nets:
    tp = 'I'
    plot_set(x, 'Translate', type_plot=tp, ax=ax[i]); i += 1
    tp = 'D'
    plot_set(x, 'Translate', type_plot=tp, ax=ax[i]); i += 1
    tp = 'G'
    plot_set(x, 'Translate', type_plot=tp, ax=ax[i]);
    i += 1

axm = np.reshape(ax, (3, 6))
cc = axm[:, 1]
for i in range(len(cc)):
    axx = cc[i]
    pp = axx.get_position()
    pp.x0 -= 0.02
    axx.set_position(pp)

cc = axm[:, 2]
for i in range(len(cc)):
    axx = cc[i]
    pp = axx.get_position()
    pp.x1 += 0.01
    axx.set_position(pp)


cc = axm[:, 3]
for i in range(len(cc)):
    axx = cc[i]
    pp = axx.get_position()
    pp.x0 -= 0.01
    axx.set_position(pp)


cc = axm[:, 4]
for i in range(len(cc)):
    axx = cc[i]
    pp = axx.get_position()
    pp.x1 += 0.02
    axx.set_position(pp)
ax[-1].remove()
ax[-2].remove()


cc = [x for idx, x in enumerate(ax) if idx in range(0, 16, 2)]
for idx, c in enumerate(cc):
    t = c.set_title(map_net_to_netstring(all_nets[idx]), size=size_text, loc='center')
    t.set_position((1.0, -1.55))

# plt.subplots_adjust(top=0.88,
# bottom=0.11,
# left=0.11,
# right=0.9,
# hspace=0.36,
# wspace=0.035)
#
plt.savefig(save_path + f'Translate_allnets_{type}.svg', format='svg')
#


##

