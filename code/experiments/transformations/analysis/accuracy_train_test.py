import matplotlib.pyplot as plt
import numpy as np
from experiments.transformations.utils.misc import get_fulltrain_strings_shapenet
import pickle
import sty
##

def average_all_seeds(pt_t, pt_obj):
    num_v = 1 if 'v' not in pt_t else 50
    def average_test(test_name):
        all_t = []
        for s in range(1, 4):
            _, _, test, name = get_fulltrain_strings_shapenet(pt_t, pt_obj, num_v, 'set1', 0, network_name, s)
            try:
                t = pickle.load(open(test, 'rb'))
            except FileNotFoundError:
                print(sty.fg.yellow + f"File not found: {test}" + sty.rs.fg)
                all_t = [np.nan]
                continue
            # all_t.append(t[0][test_name]['total_accuracy'])

            # Delta acc
            all_t.append(t[0][test_name]['total_accuracy'])

        return np.nanmean(all_t), np.nanstd(all_t)

    diff_objs = average_test('test diff objs set1')

    print(sty.fg.yellow + "TEST ON DIFFERENT OBJECTS, SAME CLASSES:")
    print(sty.fg.red + f"Pretrained on Transformation " + sty.fg.blue + f"{pt_t}" + sty.fg.red+ " with " +sty.fg.blue + f"{pt_obj}" + sty.fg.red+ " objects: " + sty.fg.blue + f"{diff_objs[0]}+-{diff_objs[1]}"  + sty.rs.fg)

    diff_vp = None
    if 'v' in pt_t:
        diff_vp = (average_test('test same objs diff vp set1')) if 'v' in pt_t else None
        print(sty.fg.yellow + "TEST ON DIFFERENT VIEWPOINTS, SAME CLASSES:")
        print(sty.fg.red + f"Pretrained on Transformation " + sty.fg.blue + f"{pt_t}" + sty.fg.red + " with " + sty.fg.blue + f"{pt_obj}" + sty.fg.red + " objects: " + sty.fg.blue + f"{diff_vp[0]}+-{diff_vp[1]}" + sty.rs.fg)
    return diff_objs, diff_vp
##

objs = 250
network_name = 'vgg11bn'
m = average_all_seeds('t', objs)
m = average_all_seeds('r', objs)
m = average_all_seeds('s', objs)
m = average_all_seeds('c', objs)
m = average_all_seeds('b', objs)
m = average_all_seeds('v', objs)
m = average_all_seeds('', objs)

##

