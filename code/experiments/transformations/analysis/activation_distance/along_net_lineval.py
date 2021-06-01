
# exp.pretraining = 'vanilla'
# exp.pretrainig = f'./models/transformations/fulltrain/FULLTRAIN_Tr_objs500_vp6_mat0_vgg11bn_trset1.pt'
# exp.get_net(10)

## Analysis done in the ipynb nearby.
from experiments.transformations.analysis.activation_distance.activation_distance_utils import *
import framework_utils
import pickle
import matplotlib.pyplot as plt
import sty


def get_filename_lineval(*args):
    folder, distance, pt_transf, dist_transf = args
    return f'./results/transformations/activation_distance/{folder}/{distance}_{dist_transf}_shapeNetSetRest_' + ('vanilla' if pt_transf == 'vanilla' else ('ImageNet'
    if pt_transf == 'ImageNet' else f'LINEVAL_pt[T{pt_transf}+set1+objs10]_objs10_vp1_mat0_trset2')) + '.pickle'
    # if folder == 'cifar10':
    #     return f'./results/transformations/activation_distance/{folder}/{distance}_T{pt_transf}_{transf}' + \
    #            (f'_imgs{objs}' if pt_transf != 'ImageNet' and pt_transf != 'vanilla' else '') + \
    #            (f'_{network_name}' if pt_transf !='ImageNet' else '') + '.pickle'


def plot_glob(pt_transf, distance_transf, folder, distance, ax=None, label=''):
    vp = 50 if pt_transf == 'v' or pt_transf == 'vtrs' else 1
    try:
        activationT = pickle.load(open(get_filename_lineval(folder, distance, pt_transf, distance_transf), 'rb'))
    except FileNotFoundError as err:
        print(sty.fg.yellow + f'NOT FOUND: {err.filename}' + sty.rs.fg)
        return False
    activationDC = pickle.load(open(get_filename_lineval(folder, distance, pt_transf, 'DiffClasses'), 'rb'))

    activation_net = activationT['distance_net']
    activation_img = activationT['distance_img']
    x_values = activationT['x_values']

    activation_netDC = activationDC['distance_net']
    activation_imgDC = activationDC['distance_img']
    x_valuesDC = activationDC['x_values']

    meanDC, stdDC, _ = DistanceActivation.get_average_cossim_across_classes(activation_netDC)
    meanCosSinOtherClasses = [np.mean(i) for i in meanDC]
    mean, std, _ = DistanceActivation.get_average_cossim_across_classes(activation_net)
    if ax is None:
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    def adjust_metric(dist, other_dist, metric):
        if metric == 'cossim':
            return 1 - (dist - 1)/(other_dist - 1)
        if metric == 'euclidean':
            return 1 - (dist / other_dist)
        return None

    mean_activation = [np.mean(x) for x in [adjust_metric(mean[i], meanCosSinOtherClasses[i], distance) if adjust else mean[i] for i in range(len(mean))]]
    ax.errorbar(layers, mean_activation,
                    yerr=None, marker='.', capsize=5, linewidth=3, markersize=8, linestyle='-',
                    label='Vanilla' if pt_transf == 'vanilla' else ('ImageNet' if pt_transf == 'ImageNet' else f'T{pt_transf}') + label)
    ax.set_xticklabels(layers, rotation=90)

    return True


def get_related_distance_string(t):
    if t == 'v':
        return 'VP'
    if t == 't':
        return 'Translate'
    if t == 'r':
        return 'Rotate'
    if t == 's':
        return 'Scale'


def plot_set(pt_list, test_transf):
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    for p in pt_list:
        f = plot_glob(p, get_related_distance_string(test_transf), f'ShapeNet/{network_name}/lineval/others' if p == 'vanilla' or p == 'ImageNet' else folder, metric, ax=ax)
        plt.title(get_related_distance_string(test_transf))
        plt.grid('on')
        plt.subplots_adjust(bottom=0.190)
        plt.ylim(-0.5, 1)

    plt.legend()

##
plt.close('all')
adjust = True
ax = None
metric = 'cossim'
objs = 5
network_name = 'vgg11bn'
net = exp_categorization_task(network_name=network_name, use_weblog=False, verbose=False).get_net(10)
layers = DistanceActivationScale(net).all_layers_name
##
plt.close('all')
folder = f'ShapeNet/{network_name}/lineval/freeze_all_but_one'

pt_list = ['ImageNet', 'vanilla', '', 'vtrs']
plot_set(pt_list + ['v'], 'v')
plot_set(pt_list + ['t'], 't')
plot_set(pt_list + ['r'], 'r')
plot_set(pt_list + ['s'], 's')

folder = f'ShapeNet/{network_name}/lineval/freeze_bkbn'

pt_list = ['ImageNet', 'vanilla', '', 'vtrs']
plot_set(pt_list + ['v'], 'v')
plot_set(pt_list + ['t'], 't')
plot_set(pt_list + ['r'], 'r')
plot_set(pt_list + ['s'], 's')

# framework_utils.save_figure_layout('along_net_lineval_ShapeNet.l')
framework_utils.load_figure_layout('along_net_lineval_ShapeNet.l')

##

def plot_two_types_net(transf):
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    folder = 'ShapeNet/lineval/freeze_all_but_one'
    f = plot_glob(transf, get_related_distance_string(transf), folder, metric, ax=ax, label='Freeze All -1')
    folder = 'ShapeNet/lineval/freeze_bkbn'
    f = plot_glob(transf, get_related_distance_string(transf), folder, metric, ax=ax, label='Freeze Bkbn')
    folder = 'ShapeNet/lineval/others'

    f = plot_glob('vanilla', get_related_distance_string(transf), folder, metric, ax=ax, label='')

    plt.legend()
    plt.title(get_related_distance_string(transf))
    plt.ylim([-0.2, 1])
    plt.grid('on')

plt.close('all')

plot_two_types_net('v')
plot_two_types_net('t')
plot_two_types_net('r')
plot_two_types_net('s')
# framework_utils.save_figure_layout('along_net_lineval_ShapeNet2.l')
framework_utils.load_figure_layout('along_net_lineval_ShapeNet2.l')


##

