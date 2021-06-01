from experiments.transformations.analysis.activation_distance.activation_distance_utils import *
import framework_utils
import pickle
import matplotlib.pyplot as plt
import sty
import seaborn as sn
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

def get_filename(dataset, distance, pt_transf, transf, objs, vp):
    evaluated_activation = 'ShapeNetSet2' if dataset == 'ShapeNet' else 'test'

    if pt_transf == 'ImageNet' or pt_transf == 'vanilla':
        return f'./results/transformations/activation_distance/{dataset}/vgg11bn/others/{distance}_{transf}_{evaluated_activation}_{pt_transf}.pickle'
    if dataset == 'ShapeNet':
        return f'./results/transformations/activation_distance/{dataset}/{network_name}/fulltrain/{distance}_{transf}_shapeNetSet2_' + \
               (f'FULLTRAIN_T{pt_transf}_objs{objs}_vp{vp}_mat0_trset1_S1' if pt_transf != 'ImageNet' and pt_transf != 'vanilla' else 'S0') + '.pickle'
    if dataset == 'cifar10':
        return f'./results/transformations/activation_distance/{dataset}/{distance}_T{pt_transf}_{transf}' + \
               (f'_imgs{objs}' if pt_transf != 'ImageNet' and pt_transf != 'vanilla' else '') + '.pickle'

def get_activations(pt_transf, transf, folder, distance, ax=None):
    vp = 50 if pt_transf == 'v' or pt_transf == 'vtrs' else 1
    try:
        activationT = pickle.load(open(get_filename(folder, distance, pt_transf, transf, objs, vp), 'rb'))
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
    if type_plot == 'image':
        v = mean_act_img
    elif type_plot == 'same':
        v = mean_same_obj

    if ax is None:
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    mean_activation = np.array([np.mean(i) for i in v])
    std_activation = np.array([np.std(i) for i in v])

    return mean_activation, std_activation

def plot_set(pt_list, diffAct, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    lss = ['-','--']
    for idx, p in enumerate(pt_list):
        mean, std = get_activations(p, diffAct, folder, metric, ax=ax)
        ax.plot(layer, mean, linewidth=1, label=map_string_to_label(diffAct) if idx == 0 else None, color=color_cycle[i], ls=lss[idx],**kwargs)
        # ax.fill_between(layer, mean - std, mean + std, alpha=0.1, color=color_cycle[i], **kwargs)
        ax.set_xticks(range(len(layer)))
        ax.set_xticklabels(layer, rotation=90)
        plt.setp(ax.get_xticklabels(), rotation=90)

        plt.title(network_name)
        plt.grid(False)
        plt.subplots_adjust(bottom=0.190)
        plt.ylim(0, 1)

    plt.legend()
##
type_dataset = 'different_classes'
exp, dataset_no_transf = get_experiment_shapeNet(type_dataset)

color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.close('all')
type_plot = 'same'
metric = 'cossim'
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
for network_name in all_nets:
    i = 0
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    layer = layers[network_name]
    plot_set(['v', ''], 'VP', ax=ax); i+=1
    plot_set(['r', ''], 'Rotate', ax=ax); i+=1
    plot_set(['s', ''], 'Scale', ax=ax); i+=1
    plot_set(['c', ''], 'Contrast', ax=ax); i+=1
    plot_set(['b', ''], 'Brightness', ax=ax); i+=1
    plot_set(['t', ''], 'Translate', ax=ax); i+=1

