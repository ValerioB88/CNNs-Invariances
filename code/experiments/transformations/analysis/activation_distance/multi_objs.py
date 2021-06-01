## Analysis of cossim distance computed in the ipynb nearby.

import seaborn as sn
from experiments.transformations.analysis.activation_distance.activation_distance_utils import *
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sty

def map_string_to_label(str):
    if str == 'vanilla':
        return 'Untrained'
    if str == 'ImageNet':
        return 'ImageNet'
    if str == '':
        return 'No Transformation'
    if str ==  'r':
        return 'Rotation'
    if str == 's':
        return 'Scale'
    if str == 't':
        return 'Translate'
    if str == 'vtrsbc':
        return 'All'
    if str == 'v':
        return 'Viewpoint'
    else:
        return str



sn.set(style="dark")
sn.axes_style()
sn.set_style('dark', {'axes.edgecolor': 'black',
                      'xtick.bottom': True,
                      'xtick.top': True,
                      'ytick.left': True,
                      'ytick.right': True,
                      'xtick.direction': 'in',
                      'ytick.direction': 'in',
                    "axes.facecolor": ".9"})


def get_filename(dataset, distance, pt_transf, transf, objs, vp, seed=0):
    evaluated_activation = 'ShapeNetSet2' if dataset == 'ShapeNet' else 'test'

    if pt_transf == 'ImageNet' or pt_transf == 'vanilla':
        return f'./results/transformations/activation_distance/{dataset}/{network_name}/others/{distance}_{transf}_{evaluated_activation}_{pt_transf}_S0.pickle'
    if dataset == 'ShapeNet':
        return f'./results/transformations/activation_distance/{dataset}/{network_name}/fulltrain/{distance}_{transf}_shapeNetSet2_' + \
               (f'FULLTRAIN_T{pt_transf}_objs{objs}_vp{vp}_mat0_trset1_S{seed}' if pt_transf != 'ImageNet' and pt_transf != 'vanilla' else '') + '.pickle'
    if dataset == 'cifar10':
        return f'./results/transformations/activation_distance/{dataset}/{distance}_T{pt_transf}_{transf}_S{seed}' + \
               (f'_imgs{objs}' if pt_transf != 'ImageNet' and pt_transf != 'vanilla' else '') + '.pickle'


def get_values(pt_transf, transf, folder, distance, objs=None, seed=0):
    vp = 50 if 'v' in pt_transf  else 1
    try:
        filename = get_filename(folder, distance, pt_transf, transf, objs, vp, seed)
        activationT = pickle.load(open(filename, 'rb'))
    except FileNotFoundError as err:
        print(sty.fg.yellow + f'NOT FOUND: {err.filename}' + sty.rs.fg)
        return None, None
    # activationDC = pickle.load(open(get_filename(folder, distance, pt_transf, 'DiffClasses', objs, vp), 'rb'))

    activation_net_same_obj = activationT['distance_net_same_obj']
    activation_net_same_class_diff_objs = activationT['distance_net_same_class_diff_obj']
    activation_net_any_obj = activationT['distance_net_any_obj']

    activation_img = activationT['distance_img']
    x_values = activationT['x_values']


    mean_any_obj, std_any_obj, _ = DistanceActivation.get_average_cossim_across_classes(activation_net_any_obj)
    mean_same_class_diff_obj, std_same_class_diff_obj, _ = DistanceActivation.get_average_cossim_across_classes(activation_net_same_class_diff_objs)
    mean_act_img, std_act_img, _ = DistanceActivation.get_average_cossim_across_classes(activation_img)
    mean_same_obj, std_same_obj, _ = DistanceActivation.get_average_cossim_across_classes(activation_net_same_obj)

    return mean_same_obj[layer_index] - np.mean(mean_any_obj[layer_index])
    # return adjust_metric(mean[layer_index], meanDC[layer_index], distance) if adjust else mean[-1], std

import matplotlib
def arrange_plot():
    plt.ylim([0, 1])
    plt.gca().set_xscale('log')
    plt.xlim([1, 500])
    plt.gca().set_xticks([1, 5, 25, 100, 250, 500])
    plt.gca().set_yticks([0, 0.2, 0.4, 0.6, 0.8])

    plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    leg1 = ax.legend(prop={'size': 20}, bbox_to_anchor=(0, 0), framealpha=1, loc="lower left")

    custom_lines = [Line2D([0], [0], color='k', ls='-', markersize=marker_size),
                    Line2D([0], [0], color='k', linestyle='--', markersize=marker_size)]

    leg1 = ax.legend(custom_lines, ['Trained', 'Untrained'], prop={'size': 20}, framealpha=1, bbox_to_anchor=(0.0, 1.0), ncol=3, loc="lower left")

    custom_lines_datasets = [Line2D([0], [0], color=color, lw=3) for color in color_cycle]

    leg2 = ax.legend(custom_lines_datasets, ['Depth-Rotation', '2D Rotate', 'Scale', 'Translate'], prop={'size': 20}, ncol=2, bbox_to_anchor=(0, 0.1), framealpha=1, loc="lower left")
    ax.add_artist(leg1)
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=size_text)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=size_text)
    # plt.subplots_adjust(top=0.88,
    #                     bottom=0.11,
    #                     left=0.055,
    #                     right=0.9,
    #                     hspace=0.185,
    #                     wspace=0.185)
    plt.grid('on')
    plt.xlabel('Num. Training Objects', fontsize=size_text)
    plt.ylabel('Invariance Metric', fontsize=size_text)



def get_all_objs(pt_transf, transf, folder, distance):
    mean_all_objs = []
    std_all_objs = []
    for objs in all_objs:
        mean_and_std = [get_values(pt_transf, transf, folder, distance, objs, seed) for seed in range(1,4)]
        tot_mean_across_seeds = [np.mean(mean_and_std[s][0]) for s in range(0, 3)]
        if mean_and_std[0] is None:
            mean_all_objs.append(None)
            std_all_objs.append(None)
        else:
            mean_all_objs.append(np.mean(tot_mean_across_seeds))
            std_all_objs.append(np.std(tot_mean_across_seeds))

    return np.array(mean_all_objs), np.array(std_all_objs)

def plot_trained_untrained(pt_transf, transf, folder):
    global idx
    m, s = get_all_objs('', transf, folder, metric)
    # plt.plot(all_objs, m, color=color_cycle[idx], marker='o', linestyle='--', linewidth=2)
    ax.semilogx(all_objs, m,
            linewidth=3, linestyle='--' if pt_transf != 'ImageNet' and pt_transf != 'vanilla' and pt_transf != '' else '--',
            label=map_string_to_label(pt_transf),
            marker='o',
            color=color_cycle[idx])
    ax.fill_between(all_objs, m - s, m + s, alpha=0.2, color=color_cycle[idx])

    m, s = get_all_objs(pt_transf,  transf, folder, metric)
    # plt.plot(all_objs, m, color=color_cycle[idx], marker='o', linewidth=2)
    ax.semilogx(all_objs, m,
            linewidth=3, linestyle='-' if pt_transf != 'ImageNet' and pt_transf != 'vanilla' and pt_transf != '' else '--',
            label=map_string_to_label(pt_transf),
            marker='o',
            color=color_cycle[idx])
    ax.fill_between(all_objs, m - s, m + s, alpha=0.2, color=color_cycle[idx])

    idx += 1
    # plt.legend()

size_text = 20
layer_index = -1
adjust = True
ax = None
all_objs = [1, 5, 25, 50, 100, 250, 500]
global idx
marker_size = 20
network_name = 'samediffnet'

metric = 'cossim'
##
# plt.close('all')
idx = 0
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(1, 1, figsize=(8, 8))
dataset = 'ShapeNet'

plot_trained_untrained('v', 'VP', dataset)
plot_trained_untrained('t', 'Translate', dataset)
plot_trained_untrained('s', 'Scale', dataset)
plot_trained_untrained('r', 'Rotate', dataset)
# plot_trained_untrained('vtrsbc', 'Brightness', dataset)
# plot_trained_untrained('vtrsbc', 'Contrast', dataset)


# idx = 0
# m, s = get_all_objs('ImageNet', 'VP', dataset, metric)
# plt.plot(all_objs, m, color=color_cycle[idx], marker='o', linewidth=2)
# idx += 1
# m, s = get_all_objs('ImageNet', 'Translate', dataset, metric)
# plt.plot(all_objs, m, color=color_cycle[idx], marker='o', linewidth=2)
# idx += 1
# m, s = get_all_objs('ImageNet', 'Scale', dataset, metric)
# plt.plot(all_objs, m, color=color_cycle[idx], marker='o', linewidth=2)
# idx += 1
# m, s = get_all_objs('ImageNet', 'Rotate', dataset, metric)
# plt.plot(all_objs, m, color=color_cycle[idx], marker='o', linewidth=2)
#
# plot_trained_untrained('ImageNet', 'Rotate', dataset)
# plot_trained_untrained('ImageNet', 'Scale', dataset)

arrange_plot()
save_path = './results/transformations/figures/single_figs/'
plt.savefig(save_path + 'multi_obj.svg', format='svg')

##
idx = 0
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
dataset = 'cifar10'
plot_trained_untrained('s', 'Scale', dataset)
plot_trained_untrained('t', 'Translate', dataset)
plot_trained_untrained('r', 'Rotate', dataset)
arrange_plot()

##

