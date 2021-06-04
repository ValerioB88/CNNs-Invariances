import argparse
from experiments.transformations.utils.misc import get_fulltrain_strings_shapenet
import numpy as np

import pickle


def get_filename(pt_transf, transf, pt_objs, seed):
    pt_num_v = 50 if 'v' in pt_transf else 1

    _, _, _, net_name = get_fulltrain_strings_shapenet(pt_transf, pt_objs, pt_num_v, 'set1', 0, 'samediffnet', seed)
    filename = f'./results/transformations/ETH/samediffnet/only_tests/{net_name}_test_testT{transf}.pickle'
    f = pickle.load(open(filename, 'rb'))[0]['test ETH']['total_accuracy']
    return f


def get_mean_std(pt_transf, transf, pt_objs):
    m = [get_filename(pt_transf, transf, 250, s) for s in [1, 2, 3]]
    return np.mean(m), np.std(m)


print("Trained on Transformed Objects [mean, std]")
[print(f"{t}->{t}{get_mean_std(t, t, 250)}") for t in ['v', 't', 'r', 's', 'b', 'c']]

print("\nTrained on un-Transformed Objects [mean, std]")
[print(f"->{t}{get_mean_std('', t, 250)}") for t in ['v', 't', 'r', 's', 'b', 'c']]