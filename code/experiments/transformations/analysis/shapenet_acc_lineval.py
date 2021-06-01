import matplotlib.pyplot as plt
import numpy as np
import pickle
result_folder = './results/transformations/ShapeNet/'

get_path = lambda type_net, pt_transf, pt_objs, test_transf, objs:  f'{result_folder}/{network_name}/lineval/{type_net}/LINEVAL_' + (f'pt[T{pt_transf}+set1+objs{pt_objs}]' if pt_transf is not None else 'vanilla') + f'_T-{test_transf}_objs{objs}_vp1_mat0_trset2_test.pickle'

def plot_bar_values(values, labels):

    width_bar = 0.15
    span = width_bar * len(values)
    values_test = [[i[0] for i in c] for c in values]
    values_train = [[i[1] for i in c] for c in values]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True)
    x = np.arange(np.max([len(i) for i in values]))
    for i in range(len(values_train)):
        ax.stem(x - span / 2 + width_bar * i + width_bar/2, values_train[i], linefmt='k', markerfmt='ko')
    for i in range(len(values_test)):
        ax.bar(x - span / 2 + width_bar * i + width_bar/2, values_test[i], width_bar, yerr=None, label=labels[i], error_kw=dict(lw=3, capsize=6, capthick=3))
    ax.set_xticks(x)
    ax.set_xticklabels(transf, rotation=0, size=20)
    plt.setp(ax.get_yticklabels(), fontsize=20)

    plt.axhline(0, color='k', linewidth=2)
    # leg1 = ax.legend(prop={'size': 20}, bbox_to_anchor=(0, 0), framealpha=0, loc="lower left")
    ax.legend(prop={'size': 20}, bbox_to_anchor=(-0.1, 1.18), ncol=2, loc='upper left')
    plt.xlabel('Tested Transformations', size=20)
    plt.ylim(0, 100)
    plt.subplots_adjust(top=0.855)
    plt.show()
    plt.grid(axis='y')


def get_acc(type_net, pt_transf=None, test_transf=None, pt_objs=10, objs=10):
    try:
        results = pickle.load(open(get_path(type_net, pt_transf, pt_objs, test_transf, objs), 'rb'))
        acc = results[0]['test same objs diff vp set2' if test_transf == 'v' else 'transformed train set2']['total_accuracy']
        train_acc = results[0]['test like train set2']['total_accuracy']
    except FileNotFoundError as c:
        acc = 0
        print(c)
    return acc, train_acc


def get_transf_notransf(type_net, pt_objs=10, objs=10):
    transf = ['t', 'r', 's']
    acc_t = [get_acc(type_net=type_net, pt_transf=t, pt_objs=pt_objs, objs=objs, test_transf=t) for t in transf]
    acc_nt = [get_acc(type_net=type_net, pt_transf='', pt_objs=pt_objs, objs=objs, test_transf=t) for t in transf]
    return acc_t, acc_nt



transf = ['t', 'r', 's']
acc_vanilla = [get_acc(type_net='vanilla', test_transf=t) for t in transf]




##

# All results
# values = [acc_lineval_notransf, acc_lineval_notransf_freezebkbn, acc_lineval_transf, acc_lineval_transf_freezebkbn, acc_vanilla, ]
# labels = ['ptT_freeze_all-1', 'ptT_freezebkbn', 'ptTx_freeze_all-1', 'ptTx_freezebkbn', 'vanilla']

# Only the interesting ones
# values = [acc_lineval_transf, acc_lineval_transf_freezebkbn, acc_vanilla, ]
# labels = ['ptTx_freeze_all', 'ptTx_freezebkbn', 'vanilla']
# plot_bar_values(values, labels)
network_name = 'vgg11bn'
plt.close('all')
acc_tA, acc_ntA = get_transf_notransf('freeze_all_but_one')

values = [acc_ntA, acc_tA]  #, acc_vanilla]
labels = ['Pretrained', 'Pretrained Transformed'] # , 'vanilla']
plot_bar_values(values, labels)
plt.title('Freeze all but last linear', size=20)

acc_tB, acc_ntB = get_transf_notransf('freeze_backbone')
values = [acc_ntB, acc_tB]
labels = ['Pretrained', 'Pretrained Transformed']
plot_bar_values(values, labels)
plt.title('Freeze Backbone', size=20)

acc_tB, acc_ntB = get_transf_notransf('scramble_fc1')
values = [acc_ntB, acc_tB]
labels = ['Pretrained', 'Pretrained Transformed']
plot_bar_values(values, labels)
plt.title('Freeze Backbone + Scramble FC', size=20)


###
plt.close('all')
transf = ['t', 'r', 's']

acc_tA, acc_ntA = get_transf_notransf('add_cat_module', objs=500)
values = [acc_ntA, acc_tA]
labels = ['Pretrained', 'Pretrained Transformed']
plot_bar_values(values, labels)
plt.title('Freeze all, Add CONV+FC module - 500 Objects', size=20)

acc_tB, acc_ntB = get_transf_notransf('add_cat_module')
values = [acc_ntB, acc_tB]
labels = ['Pretrained', 'Pretrained Transformed']
plot_bar_values(values, labels)
plt.title('Freeze all, Add CONV+FC module - 10 Objects', size=20)

##
plt.close('all')
acc_tA, acc_ntA = get_transf_notransf('add_cat_module', pt_objs=10, objs=500)
acc_tB, acc_ntB = get_transf_notransf('add_cat_module', pt_objs=500, objs=500)
plot_bar_values([acc_ntA, acc_tA], ['Pretrained', 'Pretrained Transformed'])
plt.title('Add Conv+FC - Pt on 10 Objs', size=20)
plot_bar_values([acc_ntB, acc_tB], ['Pretrained', 'Pretrained Transformed'])
plt.title('Add Conv+FC - Pt on 500 Objs', size=20)

##
plt.close('all')
acc_tA, acc_ntA = get_transf_notransf('add_cat_module', pt_objs=10, objs=500)
acc_tB, acc_ntB = get_transf_notransf('add_cat_module', pt_objs=500, objs=500)
plot_bar_values([acc_tA, acc_tB], ['Pt 10 Objs', 'Pt 500 Objs'])
plt.title('Add Conv+FC, retrained on 500 Objs', size=20)
##
plt.close('all')
acc_tA, acc_ntA = get_transf_notransf('scramble_fc1')  # scramble_fc1 also has freeze_backbone
acc_tB, acc_ntB = get_transf_notransf('freeze_backbone')
acc_tC, acc_ntC = get_transf_notransf('freeze_all_but_one')
labels = ['Pretrained', 'Pretrained Transformed']
plot_bar_values([acc_ntA, acc_tA], labels)
plt.title('Scrambled Fc', size=20)
plot_bar_values([acc_ntB, acc_tB], labels)
plt.title('freeze_backbone', size=20)
plot_bar_values([acc_ntC, acc_tC], labels)
plt.title('freeze_all_but_one', size=20)

##

##
plt.close('all')
acc_tA, acc_ntA = get_transf_notransf('freeze_all_but_one')
acc_tB, acc_ntB = get_transf_notransf('freeze_backbone')
acc_tC, acc_ntC = get_transf_notransf('scramble_fc1')  # scramble_fc1 also has freeze_backbone

plot_bar_values([acc_tA, acc_tB, acc_tC, acc_vanilla], ['Freeze all but last linear', 'Freeze Backbone', 'Freeze Backbone + Scrambled Fc', 'No Pretraining'])
##
plt.close('all')
acc_tA, acc_ntA = get_transf_notransf('freeze_all_but_one')
acc_tC, acc_ntC = get_transf_notransf('scramble_fc1')  # scramble_fc1 also has freeze_backbone

plot_bar_values([acc_tA, acc_tC, acc_vanilla], ['Freeze all but last linear', 'Freeze Backbone + Scrambled Fc', 'No Pretraining'])

##
plt.close('all')
acc_tA, acc_ntA = get_transf_notransf('freeze_all_but_one')
acc_tC, acc_ntC = get_transf_notransf('add_cat_module')  # scramble_fc1 also has freeze_backbone

plot_bar_values([acc_tA, acc_tC, acc_vanilla], ['Freeze all but last linear', 'Add CONV+FC Module', 'No Pretraining'])

##
plt.close('all')
acc_tA, acc_ntA = get_transf_notransf('freeze_all_but_one', objs=500)
acc_tC, acc_ntC = get_transf_notransf('add_cat_module', objs=500)  # scramble_fc1 also has freeze_backbone

plot_bar_values([acc_tA, acc_tC, acc_vanilla], ['Freeze all but last linear', 'Add CONV+FC Module', 'No Pretraining'])


##

