import re
from experiments.transformations.analysis.activation_distance.activation_distance_utils import *
import framework_utils
import pickle
import matplotlib.pyplot as plt
import sty
import seaborn as sn
sn.set(style="dark")
sn.axes_style()
sn.set_style('dark', {'axes.edgecolor': 'black',
                      'xtick.bottom': False,
                      'xtick.top': False,
                      'ytick.left': True,
                      'ytick.right': True,
                      'xtick.direction': 'in',
                      'ytick.direction': 'in',
                      'axes.facecolor': '.9'})
plt.rcParams['xtick.bottom'] = False

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
        return 'Same/Diff. Net.'

def only_get_interesting_labels(layer):
    p = ''
    labels = []
    for l in layer:
        l = re.search('\d+: (\w+)', l).groups()[0]
        if l != p:
            labels.append(l)
        else:
            labels.append('')
        p = l
    return labels


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

def get_activations(pt_transf, transf, folder, distance):
    vp = 50 if pt_transf == 'v' or pt_transf == 'vtrs' else 1
    mean_seed = []
    for s in [1 ,2 ,3]:
        try:
            activationT = pickle.load(open(get_filename(folder, distance, pt_transf, transf, objs, vp, s), 'rb'))
        except FileNotFoundError as err:
            print(sty.fg.yellow + f'NOT FOUND: {err.filename}' + sty.rs.fg)
            return False
        activation_net_same_obj = activationT['distance_net_same_obj']
        activation_net_same_class_diff_objs = activationT['distance_net_same_class_diff_obj']
        activation_net_any_obj = activationT['distance_net_any_obj']

        activation_img = activationT['distance_img']
        x_values = activationT['x_values']

        mean_any_obj, std_any_obj, _ = DistanceActivation.get_average_cossim_across_classes(activation_net_any_obj)
        mean_same_class_diff_obj, std_same_class_diff_obj, _ = DistanceActivation.get_average_cossim_across_classes(activation_net_same_class_diff_objs)
        mean_act_img, std_act_img, _ = DistanceActivation.get_average_cossim_across_classes(activation_img)
        mean_same_obj, std_same_obj, _ = DistanceActivation.get_average_cossim_across_classes(activation_net_same_obj)

        if transf == 'VP':
            idx = [True if i and j else False for i, j in zip([True if i == 180 else False for i in x_values[1]],  [True if i == 48 else False for i in x_values[0]])].index(True)
        if transf == 'Translate':
            idx = [True if i and j else False for i, j in zip([True if i == -0.2 else False for i in x_values[1]],  [True if i == -0.2 else False for i in x_values[0]])].index(True)
        if transf == 'Rotate':
            idx = [idx for idx, i in enumerate(x_values) if np.isclose(i, -113.877551)][0]
        if transf == 'Scale':
            idx = [idx for idx, i in enumerate(x_values) if np.isclose(i, 0.526530)][0]
        if transf == 'Brightness':
            idx = [idx for idx, i in enumerate(x_values) if np.isclose(i, 0.56842105)][0]
        if transf == 'Contrast':
            idx = [idx for idx, i in enumerate(x_values) if np.isclose(i, 0.5368421)][0]
        I = np.array([i[idx] for i in mean_same_obj])
        D = np.array([i[idx] for i in mean_any_obj])
        D2 = np.array([i[idx] for i in mean_same_class_diff_obj])
        G = (I - D)/(1 - D)
        if type_plot == 'image':
            # I forgot to do the comparisong with ITSELF at alpha level (rot=0, scale etc =1), which cossim is obviously 1 - but not having that values create ugly looking plots, so let's add it manually. We do this for each layer.
            idx = np.argmin(np.abs(x_values)) if transf == 'Rotate' else np.argmin(np.abs(x_values - 1))
            x_values = np.insert(x_values, idx + 1, 0 if transf == 'Rotate' else 1)
            mean_act_img = np.insert(mean_act_img, idx + 1, 1)
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

    return v, std

def plot_set(pt_list, diffAct, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    lss = ['-','--']
    for idx, p in enumerate(pt_list):
        mean, std = get_activations(p, diffAct, folder, metric)
        ax.set_xticks(range(len(layer)))
        ax.set_xticklabels(only_get_interesting_labels(layer), rotation=90)
        plt.setp(ax.get_xticklabels(), rotation=90)

        plt.subplots_adjust(bottom=0.190)
        plt.ylim(0, 1)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticklabels(['0', '', '0.5', '', '1'])
        ax.yaxis.grid(True)
        ax.set_xlim([0, len(layer)-1])
        plt.setp(ax.get_xticklabels(), fontsize=size_text)
        plt.setp(ax.get_yticklabels(), fontsize=size_text)

    # plt.legend()
    ax.set_title(map_net_to_netstring(network_name), size=size_text)
    return mean, std, layer
##
type_dataset = 'different_classes'
exp, dataset_no_transf = get_experiment_shapeNet(type_dataset)

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
size_text = 20
plt.close('all')
type_plot = 'G'
objs = 250
layers = {}
all_nets = ['alexnet', 'resnet18', 'resnet50', 'vgg11bn', 'vgg19bn', 'googlenet',  'densenet201', 'samediffnet']

for network_name in all_nets:
# network_name = 'googlenet'
    net = exp_categorization_task(network_name=network_name, use_weblog=False, verbose=False).get_net(10)
    d = DistanceActivationScale(net)
    # d.calculate_distance_dataloader(dataset_no_transf)
    layers[network_name] = DistanceActivationScale(net).all_layers_name

# We need to do some layer wrangling with the layers' names because googlenet is splitting at some point
layers['googlenet'] = layers['googlenet'][:57] + [layers['googlenet'][-1]]
layers['samediffnet'] = layers['samediffnet'][:-1]
##
plt.close('all')
folder = 'ShapeNet'

all_nets = ['alexnet', 'resnet18', 'resnet50', 'vgg11bn', 'vgg19bn', 'googlenet',  'samediffnet']
fig, ax = plt.subplots(1, len(all_nets), figsize=(15.85, 4.14), sharey=True)
ax = ax.flatten()
metric = 'cossim'
for idx, network_name in enumerate(all_nets):
    i = 0
    layer = layers[network_name]
    mv, _, l = plot_set(['v'], 'VP', ax=ax[idx]); i+=1
    mr, _, _ = plot_set(['r'], 'Rotate', ax=ax[idx]); i+=1
    ms, _, _ = plot_set(['s'], 'Scale', ax=ax[idx]); i+=1
    mc, _, _ = plot_set(['c'], 'Contrast', ax=ax[idx]); i+=1
    mb, _, _ = plot_set(['b'], 'Brightness', ax=ax[idx]); i+=1
    mt, _, _ = plot_set(['t'], 'Translate', ax=ax[idx]); i+=1
    average = np.mean(np.array([mv, mr, ms, mc, mb, mt]), axis=0)
    ax[idx].plot(l, average, 'k--', lw=2, label='Cosine Similarity')

metric = 'euclidean'

for idx, network_name in enumerate(all_nets):
    i = 0
    layer = layers[network_name]
    mv, _, l = plot_set(['v'], 'VP', ax=ax[idx]); i+=1
    mr, _, _ = plot_set(['r'], 'Rotate', ax=ax[idx]); i+=1
    ms, _, _ = plot_set(['s'], 'Scale', ax=ax[idx]); i+=1
    mc, _, _ = plot_set(['c'], 'Contrast', ax=ax[idx]); i+=1
    mb, _, _ = plot_set(['b'], 'Brightness', ax=ax[idx]); i+=1
    mt, _, _ = plot_set(['t'], 'Translate', ax=ax[idx]); i+=1
    average = np.mean(np.array([mv, mr, ms, mc, mb, mt]), axis=0)
    ax[idx].plot(l, average, 'r--', lw=2, label='Euclidean')
leg1 = ax[idx].legend(prop={'size': 20}, bbox_to_anchor=(0, 0), framealpha=1, loc="lower left")
##
save_path = './results/transformations/figures/single_figs/'

plt.savefig(save_path + 'along_net_hernandez.svg')