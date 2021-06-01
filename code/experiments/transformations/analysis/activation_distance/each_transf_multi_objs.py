
# exp.pretraining = 'vanilla'
# exp.pretrainig = f'./models/transformations/fulltrain/FULLTRAIN_Tr_objs500_vp6_mat0_vgg11bn_trset1.pt'
# exp.get_net(10)

## Analysis done in the ipynb nearby.
from experiments.transformations.analysis.activation_distance.activation_distance_utils import *
import pickle
import matplotlib.pyplot as plt
import sty

def get_filename(dataset, distance, pt_transf, transf, objs, vp):
    evaluated_activation = 'ShapeNetSet2' if dataset == 'ShapeNet' else 'test'

    if pt_transf == 'ImageNet' or pt_transf == 'vanilla':
        return f'./results/transformations/activation_distance/{dataset}/{network_name}/others/{distance}_{transf}_{evaluated_activation}_{pt_transf}.pickle'
    if dataset == 'ShapeNet':
        return f'./results/transformations/activation_distance/{dataset}/{network_name}/fulltrain/{distance}_{transf}_shapeNetSet2_' + \
               (f'FULLTRAIN_T{pt_transf}_objs{objs}_vp{vp}_mat0_trset1' if pt_transf != 'ImageNet' and pt_transf != 'vanilla' else '') + '.pickle'
    if dataset == 'cifar10':
        return f'./results/transformations/activation_distance/{dataset}/{distance}_T{pt_transf}_{transf}' + \
               (f'_imgs{objs}' if pt_transf != 'ImageNet' and pt_transf != 'vanilla' else '') + '.pickle'


def plot_glob(pt_transf, transf, folder, distance, ax=None):
    vp = 50 if pt_transf == 'v' or pt_transf == 'vtrs' else 1
    if ax is None:
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)

    try:
        activationT = pickle.load(open(get_filename(folder, distance, pt_transf, transf, objs, vp), 'rb'))
    except FileNotFoundError as err:
        print(sty.fg.yellow + f'NOT FOUND: {err.filename}' + sty.rs.fg)
        return False
    # activationDC = pickle.load(open(get_filename(folder, distance, pt_transf, 'DiffClasses', objs, vp), 'rb'))

    activation_net = activationT['distance_net_same']
    activation_img = activationT['distance_img']
    x_values = activationT['x_values']

    activation_netDC = activationT['distance_net_diff']
    # activation_imgDC = activationDC['distance_img']
    # x_valuesDC = activationDC['x_values']
    #
    meanDC, stdDC, _ = DistanceActivation.get_average_cossim_across_classes(activation_netDC)
    # meanCosSinOtherClasses = [np.mean(i) for i in meanDC]
    mean, std, _ = DistanceActivation.get_average_cossim_across_classes(activation_net)

    def adjust_metric(dist, other_dist, metric):
        if metric == 'cossim':
            return (dist-other_dist)
            # return (dist-other_dist)/(1-other_dist)
            return other_dist
            # return 1 - (dist - 1)/(other_dist - 1)
        if metric == 'euclidean':
            return 1 - (dist / other_dist)
        return None
    if transf == 'Translate' or transf == 'VP':
        # x = (np.array(x_values[0] * 128 + 128/2) if transf == 'Translate' else np.array(x_values[0]))
        # y = (np.array(x_values[1] * 128 + 128/2) if transf == 'Translate' else np.array(x_values[1]))
        # ax, im = framework_utils.imshow_density((x, y, mean[layer_index]-meanDC[layer_index]),
        #                                         plot_args={'size_canvas': (128, 128) if transf == 'Translate' else (120, 330),
        #                                                    'interpolate': True}, ax=ax, vmin=0, vmax=1)
        # ax.figure.colorbar(im)

        plt.plot(mean[layer_index])
        plt.plot(meanDC[layer_index])
        # plt.plot((mean[layer_index]-meanDC[layer_index]))
        plt.ylim([0.2, 1])
    else:
        # ax.errorbar(x_values, adjust_metric(mean[layer_index], meanDC[layer_index], distance) if adjust else mean[-1],
        #             yerr=None, marker='.', capsize=5, linewidth=3, markersize=8, linestyle='-',
        #             label='Vanilla' if pt_transf == 'vanilla' else ('ImageNet' if pt_transf == 'ImageNet' else f'T{pt_transf}, objs{objs}'))
        plt.plot(mean[layer_index])
        plt.plot(meanDC[layer_index])
        plt.plot((mean[layer_index]-meanDC[layer_index]), label='Vanilla' if pt_transf == 'vanilla' else ('ImageNet' if pt_transf == 'ImageNet' else f'T{pt_transf}, objs{objs}'))

    return True

def plot_set(pt_list, diffAct):
    if diffAct == 'Translate' or diffAct == 'VP':
        ax = None
    else:
        fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    for p in pt_list:
        f = plot_glob(p, diffAct, folder, metric, ax=ax)
        if (diffAct == 'Translate' or diffAct == 'VP') and f:
            plt.title(f'Pretrained T{p}')
    if not (diffAct == 'Translate' or diffAct == 'VP') and f:
        plt.title(diffAct)
        plt.grid('on')
        plt.ylim(-0.2, 1)
    plt.legend()
layer_index = -1

##
plt.close('all')
adjust = True
ax = None
metric = 'cossim'
network_name = 'vgg11bn'
folder = 'ShapeNet'

# pt_list = ['ImageNet', 'vanilla', '', 'vtrs']
pt_list = []
# plot_set(pt_list + ['r'], 'Rotate')
objs=None

# objs_list = [1, 5, 25, 50, 100, 250, 500]
objs_list = [500]
##
plt.close('all')
f = plot_glob('ImageNet', 'Translate', folder, metric, ax=ax)
plt.title('ImageNet'); plt.grid('on')

for objs in objs_list:
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
    f = plot_glob('t', 'Translate', folder, metric, ax=ax)
    plt.title(f'Num Objs {objs}')
    plt.grid('on')
    plt.legend()


##
fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
f = plot_glob('ImageNet', 'Rotate', folder, metric, ax=ax)
for objs in objs_list:
    f = plot_glob('r', 'Rotate', folder, metric, ax=ax)
    plt.grid('on')
    plt.title('Rotate')
    plt.ylim(0, 1)
    plt.legend()

fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
f = plot_glob('ImageNet', 'Scale', folder, metric, ax=ax)
for objs in objs_list:
    f = plot_glob('s', 'Scale', folder, metric, ax=ax)
    plt.grid('on')
    plt.title('scale')
    plt.ylim(0.0, 1)
    plt.legend()

plt.grid('on')
plt.legend()
#
# plot_set(pt_list + ['s'], 'Scale')
# plot_set(pt_list + ['t'], 'Translate')
# plt.figure(1)
# mng = plt.get_current_fig_manager()

# framework_utils.save_figure_layout(project_path + 'all_layers.l')
# framework_utils.load_figure_layout(project_path + 'all_layers.l')

#
##


##

