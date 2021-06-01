import matplotlib.pyplot as plt
import numpy as np
import pickle
result_folder = './results/transformations/ShapeNet//'

get_path = lambda transf, objs, vp:  f'{result_folder}/{network_name}/fulltrain/FULLTRAIN_T{transf}_objs{objs}_vp{vp}_mat0_trset1_test.pickle'


def get_acc(transf, objs):
    vp = 1 if transf != 'v' else 50
    results = pickle.load(open(get_path(transf, objs, vp), 'rb'))
    acc = results[0]['test diff objs set1']['total_accuracy']
    return acc

network_name = 'vgg11bn'
transf = ['v', 't', 'r', 's']
acc_vanilla = [get_acc(pt=0, transf=t) for t in transf]
acc_lineval = [get_acc(pt=1, transf=t) for t in transf]

##

values = [acc_vanilla, acc_lineval]
width_bar = 0.15
span = width_bar * len(values)

fig, ax = plt.subplots(1, 1, figsize=(8, 8), sharex=True, sharey=True)
labels = ['vanilla', 'lineval']
x = np.arange(np.max([len(i) for i in values]))
for i in range(len(values)):
    ax.bar(x - span / 2 + width_bar * i, values[i], width_bar, yerr=None, label=labels, error_kw=dict(lw=3, capsize=6, capthick=3))
ax.set_xticks(x)
ax.set_xticklabels(transf, rotation=0)
plt.axhline(0, color='k', linewidth=2)
plt.show()
##

